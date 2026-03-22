package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
	// "github.com/devon-caron/jarvis/search"
)

type ResponseWriter struct {
	w io.Writer
}

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
	// Searcher search.Searcher
	StopCh chan struct{}
}

// NewHandler creates a Handler with the given dependencies.
func NewHandler(registry *ModelRegistry, cfg *config.Config, stopCh chan struct{}) *Handler {
	return &Handler{
		Registry: registry,
		Config:   cfg,
		// Searcher: searcher,
		StopCh: stopCh,
	}
}

func (h *Handler) Handle(ctx context.Context, req *protocol.Request, rw *ResponseWriter) {
	log.Printf("handling request, type: %s", req.Type)
	switch req.Type {
	case protocol.ReqLoad:
		h.handleLoad(ctx, req.Load, rw)
	case protocol.ReqUnload:
		h.handleUnload(req.Unload, rw)
	case protocol.ReqChat:
		h.handleChat(ctx, req.Chat, rw)
	case protocol.ReqStatus:
		h.handleStatus(rw)
	case protocol.ReqStop:
		h.handleStop(rw)
	default:
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("unknown request type: %s", req.Type)))
	}
}

func (h *Handler) handleLoad(ctx context.Context, req *protocol.LoadRequest, rw *ResponseWriter) {
	if req == nil {
		rw.Write(protocol.ErrorResponse("request payload missing"))
		return
	}

	var path, splitMode, name, tensorSplit string
	var timeout time.Duration
	var entry config.ModelEntry
	var contextSize, batchSize int
	var flashAttention bool

	log.Printf("handling load request")

	if req.ModelPath != "" {
		path = req.ModelPath
	} else if req.Name != "" {
		updatedCfg, err := config.Load()
		if err == nil {
			h.Config = updatedCfg
		}
		var ok bool
		entry, ok = h.Config.Models[req.Name]
		if !ok {
			log.Printf("model %q not found in registry", req.Name)
			rw.Write(protocol.ErrorResponse(fmt.Sprintf(
				"model %q not found in registry; use 'jarvis models register' to add it or '-p' to load by path", req.Name)))
			return
		}
		path = entry.Path
		contextSize = entry.ContextSize
		splitMode = entry.SplitMode
	} else {
		log.Printf("either model path or name must be provided")
		rw.Write(protocol.ErrorResponse("either model path or name must be provided"))
		return
	}

	log.Printf("model path ascertained: %s", path)

	// request contextSize is a command flag, takes precedence over registry value
	if req.ContextSize > 0 {
		contextSize = req.ContextSize
	}

	// Fall back to global default if not specified
	if contextSize == 0 {
		contextSize = h.Config.Inference.ContextSize
	}

	log.Printf("context size: %d", contextSize)

	// request splitMode is a command flag, takes precedence over registry value
	if req.SplitMode != "" {
		splitMode = req.SplitMode
	}

	// request flashAttention is a command flag, takes precedence over registry value
	// if not set, check registry, otherwise use global default
	flashAttention = req.FlashAttention
	if !flashAttention {
		if entry.FlashAttention {
			flashAttention = true
		} else {
			flashAttention = h.Config.ModelOptions.FlashAttention
		}
	}

	log.Printf("flash attention: %v", flashAttention)

	// request batchSize is a command flag, takes precedence over registry value
	// if not set, check registry, otherwise use global default
	batchSize = req.BatchSize
	if batchSize == 0 {
		batchSize = entry.BatchSize
	}
	if batchSize == 0 {
		batchSize = h.Config.ModelOptions.BatchSize
	}

	log.Printf("batch size: %d", batchSize)

	// request tensorSplit is a command flag, takes precedence over registry value
	// if not set, check registry, otherwise use global default
	tensorSplit = req.TensorSplit
	if tensorSplit == "" {
		tensorSplit = entry.TensorSplit
	}
	if tensorSplit == "" {
		tensorSplit = h.Config.ModelOptions.TensorSplit
	}

	log.Printf("tensor split: %s", tensorSplit)

	// Determine model name
	name = req.Name
	if name == "" {
		name = req.ModelPath
	}

	log.Printf("model name: %s", name)

	// Determine GPUs.
	// When a split mode is set and no GPUs are explicitly specified, leave gpus
	// empty so CUDA_VISIBLE_DEVICES is not set and all GPUs are visible to
	// llama-server (required for multi-GPU split modes).
	gpus := req.GPUs
	if len(gpus) == 0 && splitMode == "" {
		gpus = []int{h.Config.DefaultGPU}
	}

	log.Printf("gpus: %v", gpus)

	// Determine timeout
	if req.Timeout != "" {
		var err error
		timeout, err = time.ParseDuration(req.Timeout)
		if err != nil {
			log.Printf("invalid timeout %q: %v", req.Timeout, err)
			rw.Write(protocol.ErrorResponse(fmt.Sprintf("invalid timeout %q: %v", req.Timeout, err)))
			return
		}
	} else if h.Config.DefaultTimeout != "" && h.Config.DefaultTimeout != "0" {
		timeout, _ = time.ParseDuration(h.Config.DefaultTimeout)
	}

	log.Printf("timeout: %v", timeout)

	opts := LoadOpts{
		ContextSize:    contextSize,
		SplitMode:      splitMode,
		Parallel:       req.Parallel,
		FlashAttention: flashAttention,
		BatchSize:      batchSize,
		TensorSplit:    tensorSplit,
	}

	log.Printf("loading model with options: %+v", opts)

	if err := h.Registry.Load(ctx, name, path, gpus, timeout, opts); err != nil {
		log.Printf("load handler model load failed: %v", err)
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("load handler model load failed: %v", err)))
		return
	}

	log.Printf("model loaded successfully")

	rw.Write(protocol.OKResponse())
}

func (h *Handler) handleUnload(req *protocol.UnloadRequest, rw *ResponseWriter) {
	log.Println("unload request received")

	// if the request is nil, there is an error and we must not continue.
	if req == nil {
		log.Printf("unload handler failed: unload request is nil")
		rw.Write(protocol.ErrorResponse("unload handler failed: unload request is nil"))
		return
	}

	if err := h.Registry.Unload(req.Name); err != nil {
		log.Printf("unload handler failed: %v", err)
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("unload handler failed: %v", err)))
		return
	}

	log.Println("unload request processed")
}

func (h *Handler) handleChat(ctx context.Context, req *protocol.ChatRequest, rw *ResponseWriter) {
	if req == nil {
		rw.Write(protocol.ErrorResponse("missing chat request payload"))
		return
	}

	log.Printf("chat request received: model=%s, messages=%d, web_search=%v, system_prompt=%v, opts=%+v, shell_pid=%d, clear_context=%v",
		req.Model, len(req.Messages), req.WebSearch, req.SystemPrompt != "", req.Opts, req.ShellPID, req.ClearContext)

	msgs := req.Messages

	if req.SystemPrompt != "" {
		msgs = append([]protocol.ChatMessage{
			{
				Role:    "system",
				Content: req.SystemPrompt,
			},
		}, msgs...)
	} else if h.Config.SystemPrompt != "" {
		msgs = append([]protocol.ChatMessage{
			{
				Role:    "system",
				Content: h.Config.SystemPrompt,
			},
		}, msgs...)
	}

	// Inject terminal context if provided
	// if req.TerminalContext != "" {
	// 	ctxMsg := protocol.ChatMessage{
	// 		Role:    "system",
	// 		Content: "Recent terminal output (for context):\n```\n" + req.TerminalContext + "\n```",
	// 	}
	// 	// Insert before user messages but after system prompt
	// 	insertIdx := 0
	// 	for insertIdx < len(msgs) && msgs[insertIdx].Role == "system" {
	// 		insertIdx++
	// 	}
	// 	msgs = append(msgs[:insertIdx], append([]protocol.ChatMessage{ctxMsg}, msgs[insertIdx:]...)...)
	// }

	// // Web search augmentation
	// if req.WebSearch && h.Searcher != nil {
	// 	userPrompt := ""
	// 	for i := len(msgs) - 1; i >= 0; i-- {
	// 		if msgs[i].Role == "user" {
	// 			userPrompt = msgs[i].Content
	// 			break
	// 		}
	// 	}
	// 	if userPrompt != "" {
	// 		results, err := h.Searcher.Search(ctx, userPrompt)
	// 		if err == nil && len(results) > 0 {
	// 			searchCtx := search.FormatResults(results)
	// 			// Insert search context as system message before user messages
	// 			msgs = append([]protocol.ChatMessage{{Role: "system", Content: searchCtx}}, msgs...)
	// 		} else if err == nil && len(results) == 0 {
	// 			rw.Write(protocol.ErrorResponse("zero results from search"))
	// 			return
	// 		} else if err != nil {
	// 			rw.Write(protocol.ErrorResponse(fmt.Sprintf("unexpected error encountered: %v", err)))
	// 		}
	// 	}
	// }

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
		log.Printf("chat function received token %v", token)
		rw.Write(protocol.DeltaTokenResponse(token))
	}, req.ShellPID, req.ClearContext)
	if err != nil {
		log.Printf("chat function received error: %v", err)
		rw.Write(protocol.ErrorResponse(err.Error()))
		return
	}

	log.Printf("chat function received end token")
	rw.Write(protocol.EndTokenResponse())
}

// dc: Written post one model load update
func (h *Handler) handleStatus(rw *ResponseWriter) {
	log.Printf("handling status request")
	models := h.Registry.Status()
	if len(models) < 1 {
		rw.Write(protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: false,
			PID:         os.Getpid(),
		}))
		return
	}

	myModel := models[0]

	rw.Write(protocol.StatusResponse(&protocol.StatusPayload{
		Running:     true,
		ModelLoaded: h.Registry.modelLoaded,
		ModelPath:   myModel.ModelPath,
		PID:         os.Getpid(),
		Model: &protocol.ModelStatus{
			ModelPath: myModel.ModelPath,
			GPUs:      myModel.GPUInfo,
		},
		Models: models,
	}))
}

func (h *Handler) handleStop(rw *ResponseWriter) {
	// TODO: Implement stop model logic
	rw.Write(protocol.OKResponse())
}
