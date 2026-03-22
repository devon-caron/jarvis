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

// ModelSlot is a per-model container that manages lifecycle and inactivity timeout.
type ModelSlot struct {
	mu        sync.RWMutex
	name      string
	backend   ModelBackend
	gpus      []int
	timeout   time.Duration
	lastUsed  time.Time
	timer     *time.Timer
	onExpire  func(name string)
	historyMu sync.Mutex
	history   map[int][]protocol.ChatMessage // keyed by shell PID
}

// ModelRegistry manages a collection of model slots with GPU conflict detection.
type ModelRegistry struct {
	mu         sync.RWMutex
	slots      map[string]*ModelSlot // keyed by model name
	gpuInUse   map[int]string        // gpu_id → model name
	loading    map[string]bool       // models currently being loaded
	newBackend func(*config.Config) ModelBackend
	cfg        *config.Config
}

// NewModelRegistry creates a new registry.
func NewModelRegistry(cfg *config.Config, newBackend func(*config.Config) ModelBackend) *ModelRegistry {
	return &ModelRegistry{
		slots:      make(map[string]*ModelSlot),
		gpuInUse:   make(map[int]string),
		loading:    make(map[string]bool),
		newBackend: newBackend,
		cfg:        cfg,
	}
}

func (r *ModelRegistry) Chat(ctx context.Context, name string, gpu int, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onToken func(string), shellPID int, clearContext bool) error {
	return nil
}

func (r *ModelRegistry) Load(ctx context.Context, name, path string, gpus []int, timeout time.Duration, opts LoadOpts) error {

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

	if _, ok := r.slots[name]; !ok {
		return fmt.Errorf("model %q not loaded", name)
	}
	return r.unloadLocked(name)
}

func (r *ModelRegistry) unloadLocked(name string) error {
	slot := r.slots[name]
	if err := slot.Unload(); err != nil {
		return err
	}
	delete(r.slots, name)
	for _, gpu := range slot.gpus {
		delete(r.gpuInUse, gpu)
	}
	return nil
}

func (r *ModelRegistry) Shutdown() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, slot := range r.slots {
		slot.backend.UnloadModel()
		delete(r.slots, name)
	}
	for gpu := range r.gpuInUse {
		delete(r.gpuInUse, gpu)
	}
}

func (s *ModelSlot) Unload() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	return s.backend.UnloadModel()
}
