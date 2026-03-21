package daemon

import (
	"context"
	"fmt"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

// LoadOpts holds llama-server tuning parameters for model loading.
type LoadOpts struct {
	ContextSize    int
	SplitMode      string
	Parallel       int // num concurrent inference requests per server
	FlashAttention bool
	BatchSize      int    // micro-batch size → llama-server -ub
	TensorSplit    string // GPU weight distribution → llama-server -ts
}

// ModelBackend abstracts the LLM engine for testability.
// The real implementation wraps llama-server; tests use a mock.
type ModelBackend interface {
	// LoadModel loads a model from the given path onto the specified GPUs.
	// The context allows cancellation (e.g. when the client disconnects).
	LoadModel(ctx context.Context, path string, gpus []int, opts LoadOpts) error
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

type ServerBackend struct {
	config *config.Config
}

func NewServerBackend(config *config.Config) ModelBackend {
	return &ServerBackend{config: config}
}

func (s *ServerBackend) LoadModel(ctx context.Context, modelPath string, gpus []int, opts LoadOpts) error {
	return fmt.Errorf("unimplemented")
}

func (s *ServerBackend) UnloadModel() error {
	return fmt.Errorf("unimplemented")
}

func (s *ServerBackend) IsLoaded() bool {
	return false
}

func (s *ServerBackend) ModelPath() string {
	return "unimplemented"
}

func (s *ServerBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	return nil
}

func (s *ServerBackend) GetStatus() (*protocol.ModelStatus, error) {
	return nil, fmt.Errorf("unimplemented")
}
