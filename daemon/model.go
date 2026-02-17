package daemon

import (
	"context"
	"errors"
	"sync"

	"github.com/devon-caron/jarvis/protocol"
)

// ErrNoModel is returned when a chat is attempted with no model loaded.
var ErrNoModel = errors.New("no model loaded")

// ModelBackend abstracts the LLM engine for testability.
// The real implementation wraps llama-go; tests use a mock.
type ModelBackend interface {
	// LoadModel loads a model from the given path with GPU layer config.
	LoadModel(path string, gpuLayers int) error
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

// ModelManager adds concurrency control around a ModelBackend.
// Chat requests hold a read lock; load/unload acquires a write lock.
// The RWMutex ensures load/unload waits for in-flight chat requests to finish.
type ModelManager struct {
	mu      sync.RWMutex
	backend ModelBackend
}

// NewModelManager creates a ModelManager wrapping the given backend.
func NewModelManager(backend ModelBackend) *ModelManager {
	return &ModelManager{backend: backend}
}

// Load loads a model, waiting for any in-flight chat requests to complete first.
// If a model is already loaded, it is unloaded before loading the new one.
func (m *ModelManager) Load(path string, gpuLayers int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.backend.IsLoaded() {
		if err := m.backend.UnloadModel(); err != nil {
			return err
		}
	}
	return m.backend.LoadModel(path, gpuLayers)
}

// Unload frees the loaded model, waiting for in-flight requests first.
func (m *ModelManager) Unload() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.backend.IsLoaded() {
		return errors.New("no model loaded")
	}
	return m.backend.UnloadModel()
}

// Chat runs a chat request. Holds a read lock so load/unload must wait.
func (m *ModelManager) Chat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.backend.IsLoaded() {
		return ErrNoModel
	}
	return m.backend.RunChat(ctx, msgs, opts, onDelta)
}

// Status returns current model status. Holds a read lock.
func (m *ModelManager) Status() (bool, *protocol.ModelStatus) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.backend.IsLoaded() {
		return false, nil
	}
	status, err := m.backend.GetStatus()
	if err != nil {
		return true, &protocol.ModelStatus{ModelPath: m.backend.ModelPath()}
	}
	return true, status
}

// IsLoaded returns whether a model is currently loaded.
func (m *ModelManager) IsLoaded() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.backend.IsLoaded()
}

// ModelPath returns the path of the currently loaded model.
func (m *ModelManager) ModelPath() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.backend.ModelPath()
}

// Shutdown unloads the model if loaded. Used during daemon shutdown.
func (m *ModelManager) Shutdown() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.backend.IsLoaded() {
		return m.backend.UnloadModel()
	}
	return nil
}
