package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
	"github.com/devon-caron/jarvis/search"
)

// ResponseWriter writes NDJSON responses to a connection.
type ResponseWriter struct {
	w io.Writer
}

// NewResponseWriter creates a ResponseWriter that writes to w.
func NewResponseWriter(w io.Writer) *ResponseWriter {
	return &ResponseWriter{w: w}
}

// Write serializes and writes a single response as NDJSON (JSON + newline).
func (rw *ResponseWriter) Write(resp *protocol.Response) error {
	data, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = rw.w.Write(data)
	return err
}

// Handler processes incoming requests and writes responses.
type Handler struct {
	Registry *ModelRegistry
	Config   *config.Config
	Searcher search.Searcher
	StopCh   chan struct{}
}

// NewHandler creates a Handler with the given dependencies.
func NewHandler(registry *ModelRegistry, cfg *config.Config, searcher search.Searcher, stopCh chan struct{}) *Handler {
	return &Handler{
		Registry: registry,
		Config:   cfg,
		Searcher: searcher,
		StopCh:   stopCh,
	}
}

// Handle routes a request to the appropriate handler and writes responses.
// The context is cancelled when the client disconnects, allowing long-running
// operations (like model loading) to be interrupted.
func (h *Handler) Handle(ctx context.Context, req *protocol.Request, rw *ResponseWriter) {
	switch req.Type {
	case protocol.ReqChat:
		h.handleChat(ctx, req.Chat, rw)
	case protocol.ReqLoad:
		h.handleLoad(ctx, req.Load, rw)
	case protocol.ReqUnload:
		h.handleUnload(req.Unload, rw)
	case protocol.ReqStatus:
		h.handleStatus(rw)
	case protocol.ReqStop:
		h.handleStop(rw)
	default:
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("unknown request type: %s", req.Type)))
	}
}

func (h *Handler) handleChat(ctx context.Context, req *protocol.ChatRequest, rw *ResponseWriter) {
	if req == nil {
		rw.Write(protocol.ErrorResponse("missing chat request payload"))
		return
	}

	msgs := req.Messages

	// Prepend system prompt if configured
	if req.SystemPrompt != "" {
		msgs = append([]protocol.ChatMessage{{Role: "system", Content: req.SystemPrompt}}, msgs...)
	} else if h.Config.SystemPrompt != "" {
		msgs = append([]protocol.ChatMessage{{Role: "system", Content: h.Config.SystemPrompt}}, msgs...)
	}

	// Inject terminal context if provided
	if req.TerminalContext != "" {
		ctxMsg := protocol.ChatMessage{
			Role:    "system",
			Content: "Recent terminal output (for context):\n```\n" + req.TerminalContext + "\n```",
		}
		// Insert before user messages but after system prompt
		insertIdx := 0
		for insertIdx < len(msgs) && msgs[insertIdx].Role == "system" {
			insertIdx++
		}
		msgs = append(msgs[:insertIdx], append([]protocol.ChatMessage{ctxMsg}, msgs[insertIdx:]...)...)
	}

	// Web search augmentation
	if req.WebSearch && h.Searcher != nil {
		userPrompt := ""
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == "user" {
				userPrompt = msgs[i].Content
				break
			}
		}
		if userPrompt != "" {
			results, err := h.Searcher.Search(ctx, userPrompt)
			if err == nil && len(results) > 0 {
				searchCtx := search.FormatResults(results)
				// Insert search context as system message before user messages
				msgs = append([]protocol.ChatMessage{{Role: "system", Content: searchCtx}}, msgs...)
			} else if err == nil && len(results) == 0 {
				rw.Write(protocol.ErrorResponse("zero results from search"))
				return
			} else if err != nil {
				rw.Write(protocol.ErrorResponse(fmt.Sprintf("unexpected error encountered: %v", err)))
			}
		}
	}

	// Merge inference opts with config defaults
	opts := req.Opts
	if opts.MaxTokens == 0 {
		opts.MaxTokens = h.Config.Inference.MaxTokens
	}
	if opts.Temperature == 0 {
		opts.Temperature = h.Config.Inference.Temperature
	}
	if opts.TopP == 0 {
		opts.TopP = h.Config.Inference.TopP
	}
	if opts.TopK == 0 {
		opts.TopK = h.Config.Inference.TopK
	}

	// Resolve GPU pointer to int (-1 = not specified).
	gpu := -1
	if req.GPU != nil {
		gpu = *req.GPU
	}

	// Route to model by name, GPU, or auto-route.
	err := h.Registry.Chat(ctx, req.Model, gpu, msgs, opts, func(token string) {
		rw.Write(protocol.DeltaResponse(token))
	}, req.ShellPID, req.ClearContext)
	if err != nil {
		rw.Write(protocol.ErrorResponse(err.Error()))
		return
	}
	rw.Write(protocol.DoneResponse())
}

func (h *Handler) handleLoad(ctx context.Context, req *protocol.LoadRequest, rw *ResponseWriter) {
	if req == nil {
		rw.Write(protocol.ErrorResponse("missing load request payload"))
		return
	}

	// Resolve model path, context size, and split mode from registry or request.
	var path string
	var contextSize int
	var splitMode string
	var entry config.ModelEntry
	if req.ModelPath != "" {
		path = req.ModelPath
	} else if req.Name != "" {
		// Reload config from disk so newly registered models are visible.
		cfg, err := config.Load()
		if err == nil {
			h.Config = cfg
		}
		var ok bool
		entry, ok = h.Config.Models[req.Name]
		if !ok {
			rw.Write(protocol.ErrorResponse(fmt.Sprintf(
				"model %q not found in registry; use 'jarvis models register' to add it or '-p' to load by path", req.Name)))
			return
		}
		path = entry.Path
		contextSize = entry.ContextSize
		splitMode = entry.SplitMode
	} else {
		rw.Write(protocol.ErrorResponse("must specify a model name or path"))
		return
	}

	// Request-level context size overrides the registry entry.
	if req.ContextSize > 0 {
		contextSize = req.ContextSize
	}
	// Fall back to global default if still unset.
	if contextSize == 0 {
		contextSize = h.Config.Inference.ContextSize
	}

	// Request-level split mode overrides the registry entry.
	if req.SplitMode != "" {
		splitMode = req.SplitMode
	}

	// Flash attention: request > model entry > global config.
	flashAttention := req.FlashAttention
	if !flashAttention {
		if entry.FlashAttention {
			flashAttention = true
		} else {
			flashAttention = h.Config.ModelOptions.FlashAttention
		}
	}

	// Batch size: request > model entry > global config.
	batchSize := req.BatchSize
	if batchSize == 0 {
		batchSize = entry.BatchSize
	}
	if batchSize == 0 {
		batchSize = h.Config.ModelOptions.BatchSize
	}

	// Tensor split: request > model entry > global config.
	tensorSplit := req.TensorSplit
	if tensorSplit == "" {
		tensorSplit = entry.TensorSplit
	}
	if tensorSplit == "" {
		tensorSplit = h.Config.ModelOptions.TensorSplit
	}

	// Determine model name
	name := req.Name
	if name == "" {
		name = req.ModelPath
	}

	// Determine GPUs.
	// When a split mode is set and no GPUs are explicitly specified, leave gpus
	// empty so CUDA_VISIBLE_DEVICES is not set and all GPUs are visible to
	// llama-server (required for multi-GPU split modes).
	gpus := req.GPUs
	if len(gpus) == 0 && splitMode == "" {
		gpus = []int{h.Config.DefaultGPU}
	}

	// Determine timeout
	var timeout time.Duration
	if req.Timeout != "" {
		var err error
		timeout, err = time.ParseDuration(req.Timeout)
		if err != nil {
			rw.Write(protocol.ErrorResponse(fmt.Sprintf("invalid timeout %q: %v", req.Timeout, err)))
			return
		}
	} else if h.Config.DefaultTimeout != "" && h.Config.DefaultTimeout != "0" {
		timeout, _ = time.ParseDuration(h.Config.DefaultTimeout)
	}

	opts := LoadOpts{
		ContextSize:    contextSize,
		SplitMode:      splitMode,
		Parallel:       req.Parallel,
		FlashAttention: flashAttention,
		BatchSize:      batchSize,
		TensorSplit:    tensorSplit,
	}

	if err := h.Registry.Load(ctx, name, path, gpus, timeout, opts); err != nil {
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("load handler model load failed: %v", err.Error())))
		return
	}
	rw.Write(protocol.OKResponse())
}

func (h *Handler) handleUnload(req *protocol.UnloadRequest, rw *ResponseWriter) {
	var err error
	if req != nil && req.GPU != nil {
		err = h.Registry.UnloadByGPU(*req.GPU)
	} else {
		name := ""
		if req != nil {
			name = req.Name
		}
		err = h.Registry.Unload(name)
	}
	if err != nil {
		rw.Write(protocol.ErrorResponse(err.Error()))
		return
	}
	rw.Write(protocol.OKResponse())
}

func (h *Handler) handleStatus(rw *ResponseWriter) {
	models := h.Registry.Status()
	loaded := len(models) > 0

	// Backwards compat: populate single-model fields if exactly one model
	var modelPath string
	var modelStatus *protocol.ModelStatus
	if len(models) == 1 {
		modelPath = models[0].ModelPath
		modelStatus = &protocol.ModelStatus{
			ModelPath: models[0].ModelPath,
			GPUs:      models[0].GPUInfo,
		}
	}

	rw.Write(protocol.StatusResponse(&protocol.StatusPayload{
		Running:     true,
		ModelLoaded: loaded,
		ModelPath:   modelPath,
		PID:         os.Getpid(),
		Model:       modelStatus,
		Models:      models,
	}))
}

func (h *Handler) handleStop(rw *ResponseWriter) {
	rw.Write(protocol.OKResponse())
	// Signal the daemon to stop
	select {
	case h.StopCh <- struct{}{}:
	default:
	}
}
