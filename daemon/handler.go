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
func (h *Handler) Handle(req *protocol.Request, rw *ResponseWriter) {
	switch req.Type {
	case protocol.ReqChat:
		h.handleChat(req.Chat, rw)
	case protocol.ReqLoad:
		h.handleLoad(req.Load, rw)
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

func (h *Handler) handleChat(req *protocol.ChatRequest, rw *ResponseWriter) {
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
			results, err := h.Searcher.Search(context.Background(), userPrompt)
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
	if opts.ContextSize == 0 {
		opts.ContextSize = h.Config.Inference.ContextSize
	}
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
	err := h.Registry.Chat(context.Background(), req.Model, gpu, msgs, opts, func(token string) {
		rw.Write(protocol.DeltaResponse(token))
	})
	if err != nil {
		rw.Write(protocol.ErrorResponse(err.Error()))
		return
	}
	rw.Write(protocol.DoneResponse())
}

func (h *Handler) handleLoad(req *protocol.LoadRequest, rw *ResponseWriter) {
	if req == nil {
		rw.Write(protocol.ErrorResponse("missing load request payload"))
		return
	}

	path := h.Config.ResolveModel(req.ModelPath)

	// Determine model name
	name := req.Name
	if name == "" {
		name = req.ModelPath
	}

	// Determine GPUs
	gpus := req.GPUs
	if len(gpus) == 0 {
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

	if err := h.Registry.Load(name, path, gpus, timeout); err != nil {
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("failed to load model: %v", err)))
		return
	}
	rw.Write(protocol.OKResponse())
}

func (h *Handler) handleUnload(req *protocol.UnloadRequest, rw *ResponseWriter) {
	name := ""
	if req != nil {
		name = req.Name
	}
	if err := h.Registry.Unload(name); err != nil {
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
