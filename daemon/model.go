package daemon

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

// ErrNoModel is returned when a chat is attempted with no model loaded.
var ErrNoModel = errors.New("no model loaded")

// LoadOpts contains options for model loading.
type LoadOpts struct {
	NVLink       bool // Enable tensor parallelism across GPUs (requires NVLink interconnect)
	EnforceEager bool // Disable CUDA graph capturing for faster model loading
}

// ModelBackend abstracts the LLM engine for testability.
// The real implementation uses vLLM; tests use a mock.
type ModelBackend interface {
	// LoadModel loads a model from the given path onto the specified GPUs.
	// gpus is the list of GPU device IDs (e.g. [0] or [0,1]).
	LoadModel(path string, gpus []int, opts LoadOpts) error
	// UnloadModel frees the currently loaded model.
	UnloadModel() error
	// IsLoaded returns true if a model is currently loaded.
	IsLoaded() bool
	// ModelPath returns the path of the currently loaded model.
	ModelPath() string
	// RunChat streams a chat response, calling onDelta for each token.
	RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error
	// GetStatus returns model and GPU status info.
	GetStatus() (*protocol.ModelStatus, error)
}

// ModelSlot is a per-model container that manages lifecycle and inactivity timeout.
type ModelSlot struct {
	mu       sync.RWMutex
	name     string
	backend  ModelBackend
	gpus     []int
	timeout  time.Duration
	lastUsed time.Time
	timer    *time.Timer
	onExpire func(name string)
}

// newModelSlot creates a new slot. If timeout > 0, starts an inactivity timer.
func newModelSlot(name string, backend ModelBackend, gpus []int, timeout time.Duration, onExpire func(string)) *ModelSlot {
	s := &ModelSlot{
		name:     name,
		backend:  backend,
		gpus:     gpus,
		timeout:  timeout,
		lastUsed: time.Now(),
		onExpire: onExpire,
	}
	if timeout > 0 {
		s.timer = time.AfterFunc(timeout, func() {
			if s.onExpire != nil {
				s.onExpire(s.name)
			}
		})
	}
	return s
}

// Chat runs a chat request on this slot. Resets the inactivity timer.
func (s *ModelSlot) Chat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.backend.IsLoaded() {
		return ErrNoModel
	}

	s.lastUsed = time.Now()
	if s.timer != nil {
		s.timer.Reset(s.timeout)
	}

	return s.backend.RunChat(ctx, msgs, opts, onDelta)
}

// Unload stops the timer and unloads the model.
func (s *ModelSlot) Unload() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	return s.backend.UnloadModel()
}

// Status returns the slot's status info.
func (s *ModelSlot) Status() protocol.SlotInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	info := protocol.SlotInfo{
		Name:      s.name,
		ModelPath: s.backend.ModelPath(),
		GPUs:      s.gpus,
		LastUsed:  s.lastUsed,
	}
	if s.timeout > 0 {
		info.Timeout = s.timeout.String()
	}
	if status, err := s.backend.GetStatus(); err == nil && status != nil {
		info.GPUInfo = status.GPUs
	}
	return info
}

// ModelRegistry manages a collection of model slots with GPU conflict detection.
type ModelRegistry struct {
	mu         sync.RWMutex
	slots      map[string]*ModelSlot // keyed by model name
	gpuInUse   map[int]string        // gpu_id → model name
	newBackend func(*config.Config) ModelBackend
	cfg        *config.Config
}

// NewModelRegistry creates a new registry.
func NewModelRegistry(cfg *config.Config, newBackend func(*config.Config) ModelBackend) *ModelRegistry {
	return &ModelRegistry{
		slots:      make(map[string]*ModelSlot),
		gpuInUse:   make(map[int]string),
		newBackend: newBackend,
		cfg:        cfg,
	}
}

// Load loads a model onto the specified GPUs with an optional inactivity timeout.
func (r *ModelRegistry) Load(name, path string, gpus []int, timeout time.Duration, opts LoadOpts) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Multi-GPU requires NVLink for tensor parallelism.
	if len(gpus) > 1 && !opts.NVLink {
		return fmt.Errorf("multiple GPUs require --nvlink for tensor parallelism")
	}

	// Check GPU conflicts before doing anything irreversible.
	for _, gpu := range gpus {
		if owner, ok := r.gpuInUse[gpu]; ok {
			return fmt.Errorf("GPU %d already in use by model %q", gpu, owner)
		}
	}

	// Reject duplicate names — caller must unload before reloading.
	if _, ok := r.slots[name]; ok {
		return fmt.Errorf("model %q is already loaded; unload it first", name)
	}

	// Create backend and load
	backend := r.newBackend(r.cfg)
	if err := backend.LoadModel(path, gpus, opts); err != nil {
		return err
	}

	// Create slot
	slot := newModelSlot(name, backend, gpus, timeout, r.removeExpired)
	r.slots[name] = slot
	for _, gpu := range gpus {
		r.gpuInUse[gpu] = name
	}

	return nil
}

// UnloadByGPU unloads whichever model is currently on the given GPU.
func (r *ModelRegistry) UnloadByGPU(gpu int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	owner, ok := r.gpuInUse[gpu]
	if !ok {
		return fmt.Errorf("no model loaded on GPU %d", gpu)
	}
	return r.unloadLocked(owner)
}

// Unload unloads a specific model by name. If name is empty and only one model
// is loaded, unloads that one.
func (r *ModelRegistry) Unload(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if name == "" {
		if len(r.slots) == 1 {
			for n := range r.slots {
				name = n
			}
		} else if len(r.slots) == 0 {
			return errors.New("no model loaded")
		} else {
			names := make([]string, 0, len(r.slots))
			for n := range r.slots {
				names = append(names, n)
			}
			return fmt.Errorf("multiple models loaded, specify which to unload: %v", names)
		}
	}

	if _, ok := r.slots[name]; !ok {
		return fmt.Errorf("model %q not loaded", name)
	}
	return r.unloadLocked(name)
}

// unloadLocked tears down the named slot. Caller must hold r.mu (write lock).
func (r *ModelRegistry) unloadLocked(name string) error {
	slot := r.slots[name]
	if err := slot.Unload(); err != nil {
		return err
	}
	for _, gpu := range slot.gpus {
		delete(r.gpuInUse, gpu)
	}
	delete(r.slots, name)
	return nil
}

// Chat routes a chat request to the appropriate model slot.
//
// Routing priority:
//  1. name != ""  → route by model name
//  2. gpu >= 0    → route to whichever model is on that GPU
//  3. single model loaded → auto-route
//  4. multiple models → route to model on cfg.DefaultGPU
func (r *ModelRegistry) Chat(ctx context.Context, name string, gpu int, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	r.mu.RLock()

	if len(r.slots) == 0 {
		r.mu.RUnlock()
		return ErrNoModel
	}

	var slot *ModelSlot

	if name != "" {
		s, ok := r.slots[name]
		if !ok {
			r.mu.RUnlock()
			return fmt.Errorf("model %q not loaded", name)
		}
		slot = s
	} else if gpu >= 0 {
		owner, ok := r.gpuInUse[gpu]
		if !ok {
			r.mu.RUnlock()
			return fmt.Errorf("no model loaded on GPU %d", gpu)
		}
		slot = r.slots[owner]
	} else if len(r.slots) == 1 {
		for _, s := range r.slots {
			slot = s
		}
	} else {
		defaultGPU := r.cfg.DefaultGPU
		if owner, ok := r.gpuInUse[defaultGPU]; ok {
			slot = r.slots[owner]
		} else {
			names := make([]string, 0, len(r.slots))
			for n := range r.slots {
				names = append(names, n)
			}
			r.mu.RUnlock()
			return fmt.Errorf("no model on default GPU %d, available models: %v", defaultGPU, names)
		}
	}

	r.mu.RUnlock()
	return slot.Chat(ctx, msgs, opts, onDelta)
}

// Status returns info for all loaded models.
func (r *ModelRegistry) Status() []protocol.SlotInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	infos := make([]protocol.SlotInfo, 0, len(r.slots))
	for _, slot := range r.slots {
		infos = append(infos, slot.Status())
	}
	return infos
}

// Shutdown unloads all models.
func (r *ModelRegistry) Shutdown() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, slot := range r.slots {
		slot.Unload()
		delete(r.slots, name)
	}
	for gpu := range r.gpuInUse {
		delete(r.gpuInUse, gpu)
	}
}

// removeExpired is the callback for timer expiry. It acquires the write lock and cleans up.
func (r *ModelRegistry) removeExpired(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	slot, ok := r.slots[name]
	if !ok {
		return
	}
	slot.Unload()
	for _, gpu := range slot.gpus {
		delete(r.gpuInUse, gpu)
	}
	delete(r.slots, name)
}
