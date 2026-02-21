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

// ModelBackend abstracts the LLM engine for testability.
// The real implementation wraps llama-go; tests use a mock.
type ModelBackend interface {
	// LoadModel loads a model from the given path onto the specified GPUs.
	// gpus is the list of GPU device IDs (e.g. [0] or [0,1]).
	LoadModel(path string, gpus []int) error
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
func (r *ModelRegistry) Load(name, path string, gpus []int, timeout time.Duration) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check GPU conflicts
	for _, gpu := range gpus {
		if owner, ok := r.gpuInUse[gpu]; ok && owner != name {
			return fmt.Errorf("GPU %d already in use by model %q", gpu, owner)
		}
	}

	// If model with same name is already loaded, unload it first
	if existing, ok := r.slots[name]; ok {
		existing.Unload()
		for _, gpu := range existing.gpus {
			delete(r.gpuInUse, gpu)
		}
		delete(r.slots, name)
	}

	// Create backend and load
	backend := r.newBackend(r.cfg)
	if err := backend.LoadModel(path, gpus); err != nil {
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

	slot, ok := r.slots[name]
	if !ok {
		return fmt.Errorf("model %q not loaded", name)
	}

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
// If name is empty: auto-route to single model, or to model on default GPU.
func (r *ModelRegistry) Chat(ctx context.Context, name string, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	r.mu.RLock()

	if len(r.slots) == 0 {
		r.mu.RUnlock()
		return ErrNoModel
	}

	var slot *ModelSlot

	if name != "" {
		// Explicit model requested
		s, ok := r.slots[name]
		if !ok {
			r.mu.RUnlock()
			return fmt.Errorf("model %q not loaded", name)
		}
		slot = s
	} else if len(r.slots) == 1 {
		// Auto-route: only one model loaded
		for _, s := range r.slots {
			slot = s
		}
	} else {
		// Multiple models: route to model on default GPU
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
