package daemon

import (
	"context"
	"errors"
	"fmt"
	"log"
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
	mu          sync.RWMutex
	modelLoaded bool
	slots       map[string]*ModelSlot // keyed by model name
	gpuInUse    map[int]string        // gpu_id → model name
	loading     map[string]bool       // models currently being loaded
	newBackend  func(*config.Config) ModelBackend
	cfg         *config.Config
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
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.modelLoaded {
		return errors.New("only one model can be loaded at a time")
	}

	r.modelLoaded = true

	return nil
}

// Unload unloads a specific model by name. If name is empty and only one model
// is loaded, unloads that one.
func (r *ModelRegistry) Unload(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.modelLoaded = false

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

	var status *protocol.ModelStatus
	var err error
	if status, err = s.backend.GetStatus(); err == nil && status != nil {
		info.GPUInfo = status.GPUs
	} else if err != nil {
		log.Printf("failed to get backend status for model %s: %v", s.name, err)
	}

	return info
}
