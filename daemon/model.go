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

// ModelRegistry manages loading and unloading a single model at a time.
type ModelRegistry struct {
	mu         sync.RWMutex
	loaded     bool
	name       string
	path       string
	gpus       []int
	backend    ModelBackend
	newBackend func(*config.Config) ModelBackend
	cfg        *config.Config
}

// NewModelRegistry creates a new registry.
func NewModelRegistry(cfg *config.Config, newBackend func(*config.Config) ModelBackend) *ModelRegistry {
	return &ModelRegistry{
		newBackend: newBackend,
		cfg:        cfg,
	}
}

func (r *ModelRegistry) Chat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onToken func(string), shellPID int, clearContext bool) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.loaded || r.backend == nil {
		return errors.New("no model loaded")
	}

	return r.backend.RunChat(ctx, msgs, opts, onToken)
}

func (r *ModelRegistry) Load(ctx context.Context, name, path string, gpus []int, timeout time.Duration, opts LoadOpts) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.loaded {
		return errors.New("a model is already loaded; unload it first")
	}

	backend := r.newBackend(r.cfg)
	if err := backend.LoadModel(ctx, path, gpus, opts); err != nil {
		return fmt.Errorf("backend failed to load model: %w", err)
	}

	r.backend = backend
	r.name = name
	r.path = path
	r.gpus = gpus
	r.loaded = true

	return nil
}

// Unload unloads the currently loaded model.
func (r *ModelRegistry) Unload(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.loaded {
		return errors.New("no model loaded")
	}

	if name != "" && name != r.name {
		return fmt.Errorf("model %q is not loaded (loaded: %q)", name, r.name)
	}

	if err := r.backend.UnloadModel(); err != nil {
		return err
	}

	r.backend = nil
	r.name = ""
	r.path = ""
	r.gpus = nil
	r.loaded = false

	return nil
}

// Status returns info about the currently loaded model, if any.
func (r *ModelRegistry) Status() *protocol.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.loaded || r.backend == nil {
		return nil
	}

	info := &protocol.ModelInfo{
		Name:      r.name,
		ModelPath: r.backend.ModelPath(),
		GPUs:      r.gpus,
	}

	if status, err := r.backend.GetStatus(); err == nil && status != nil {
		info.GPUInfo = status.GPUs
	} else if err != nil {
		log.Printf("failed to get backend status for model %s: %v", r.name, err)
	}

	return info
}

// IsLoaded returns whether a model is currently loaded.
func (r *ModelRegistry) IsLoaded() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.loaded
}

func (r *ModelRegistry) Shutdown() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.loaded && r.backend != nil {
		r.backend.UnloadModel()
		r.backend = nil
		r.name = ""
		r.path = ""
		r.gpus = nil
		r.loaded = false
	}
}
