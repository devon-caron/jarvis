package daemon

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
	llama "github.com/tcpipuk/llama-go"
)

// LlamaBackend implements ModelBackend using the llama-go library.
type LlamaBackend struct {
	model *llama.Model
	path  string
	cfg   *config.Config
}

// NewLlamaBackend creates a new LlamaBackend with the given config.
func NewLlamaBackend(cfg *config.Config) *LlamaBackend {
	return &LlamaBackend{cfg: cfg}
}

// buildTensorSplit returns a comma-separated weight string for llama.cpp's
// tensor_split parameter. GPUs in the specified list get weight 1; all others
// get 0. Returns "" if totalGPUs is 0 or gpus is empty.
func buildTensorSplit(gpus []int, totalGPUs int) string {
	if totalGPUs == 0 || len(gpus) == 0 {
		return ""
	}
	gpuSet := make(map[int]bool, len(gpus))
	for _, g := range gpus {
		gpuSet[g] = true
	}
	parts := make([]string, totalGPUs)
	for i := range parts {
		if gpuSet[i] {
			parts[i] = "1"
		} else {
			parts[i] = "0"
		}
	}
	return strings.Join(parts, ",")
}

func (e *LlamaBackend) LoadModel(path string, gpus []int) error {
	opts := []llama.ModelOption{
		llama.WithGPULayers(-1),
	}

	if len(gpus) > 0 {
		opts = append(opts, llama.WithMainGPU(strconv.Itoa(gpus[0])))
		if split := buildTensorSplit(gpus, llama.GPUCount()); split != "" {
			opts = append(opts, llama.WithTensorSplit(split))
		}
	}

	if e.cfg.ModelOptions.MLock {
		opts = append(opts, llama.WithMLock())
	}

	model, err := llama.LoadModel(path, opts...)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	e.model = model
	e.path = path
	return nil
}

func (e *LlamaBackend) UnloadModel() error {
	if e.model == nil {
		return nil
	}
	err := e.model.Close()
	e.model = nil
	e.path = ""
	return err
}

func (e *LlamaBackend) IsLoaded() bool {
	return e.model != nil
}

func (e *LlamaBackend) ModelPath() string {
	return e.path
}

func (e *LlamaBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	if e.model == nil {
		return fmt.Errorf("no model loaded")
	}

	// Determine context size
	ctxSize := opts.ContextSize
	if ctxSize == 0 {
		ctxSize = e.cfg.Inference.ContextSize
	}

	llamaCtx, err := e.model.NewContext(llama.WithContext(ctxSize))
	if err != nil {
		return fmt.Errorf("failed to create context: %w", err)
	}
	defer llamaCtx.Close()

	// Convert protocol messages to llama messages
	llamaMsgs := make([]llama.ChatMessage, len(msgs))
	for i, m := range msgs {
		llamaMsgs[i] = llama.ChatMessage{Role: m.Role, Content: m.Content}
	}

	// Build chat options from inference opts, falling back to config defaults
	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = e.cfg.Inference.MaxTokens
	}
	temp := opts.Temperature
	if temp == 0 {
		temp = e.cfg.Inference.Temperature
	}
	topP := opts.TopP
	if topP == 0 {
		topP = e.cfg.Inference.TopP
	}
	topK := opts.TopK
	if topK == 0 {
		topK = e.cfg.Inference.TopK
	}

	chatOpts := llama.ChatOptions{
		MaxTokens:   llama.Int(maxTokens),
		Temperature: llama.Float32(float32(temp)),
		TopP:        llama.Float32(float32(topP)),
		TopK:        llama.Int(topK),
	}

	// Stream chat
	deltaCh, errCh := llamaCtx.ChatStream(ctx, llamaMsgs, chatOpts)
	for delta := range deltaCh {
		onDelta(delta.Content)
	}
	if err := <-errCh; err != nil {
		return fmt.Errorf("chat failed: %w", err)
	}
	return nil
}

func (e *LlamaBackend) GetStatus() (*protocol.ModelStatus, error) {
	if e.model == nil {
		return nil, fmt.Errorf("no model loaded")
	}
	stats, err := e.model.Stats()
	if err != nil {
		return &protocol.ModelStatus{ModelPath: e.path}, nil
	}

	gpus := make([]protocol.GPUInfo, len(stats.GPUs))
	for i, g := range stats.GPUs {
		gpus[i] = protocol.GPUInfo{
			DeviceID:      g.DeviceID,
			DeviceName:    g.DeviceName,
			FreeMemoryMB:  g.FreeMemoryMB,
			TotalMemoryMB: g.TotalMemoryMB,
		}
	}

	return &protocol.ModelStatus{
		ModelPath: e.path,
		GPULayers: stats.Runtime.GPULayersLoaded,
		GPUs:      gpus,
	}, nil
}
