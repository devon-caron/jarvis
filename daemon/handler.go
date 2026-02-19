package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

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
	Manager  *ModelManager
	Config   *config.Config
	Searcher search.Searcher
	StopCh   chan struct{}
}

// NewHandler creates a Handler with the given dependencies.
func NewHandler(manager *ModelManager, cfg *config.Config, searcher search.Searcher, stopCh chan struct{}) *Handler {
	return &Handler{
		Manager:  manager,
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
		h.handleUnload(rw)
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

	err := h.Manager.Chat(context.Background(), msgs, opts, func(token string) {
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
	gpuLayers := req.GPULayers
	if gpuLayers == 0 {
		gpuLayers = h.Config.ModelOptions.GPULayers
	}

	if err := h.Manager.Load(path, gpuLayers); err != nil {
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("failed to load model: %v", err)))
		return
	}
	rw.Write(protocol.OKResponse())
}

func (h *Handler) handleUnload(rw *ResponseWriter) {
	if err := h.Manager.Unload(); err != nil {
		rw.Write(protocol.ErrorResponse(err.Error()))
		return
	}
	rw.Write(protocol.OKResponse())
}

func (h *Handler) handleStatus(rw *ResponseWriter) {
	loaded, modelStatus := h.Manager.Status()
	rw.Write(protocol.StatusResponse(&protocol.StatusPayload{
		Running:     true,
		ModelLoaded: loaded,
		ModelPath:   h.Manager.ModelPath(),
		PID:         os.Getpid(),
		Model:       modelStatus,
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
