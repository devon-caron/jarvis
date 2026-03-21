package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
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
	StopCh   chan struct{}
}

// NewHandler creates a Handler with the given dependencies.
func NewHandler(registry *ModelRegistry, cfg *config.Config, stopCh chan struct{}) *Handler {
	return &Handler{
		Registry: registry,
		Config:   cfg,
		StopCh:   stopCh,
	}
}

func (h *Handler) Handle(ctx context.Context, req *protocol.Request, rw *ResponseWriter) {
	switch req.Type {
	case protocol.ReqLoad:
		h.handleLoad(ctx, req.Load, rw)
	case protocol.ReqUnload:
		// TODO: Implement unload model logic
		rw.Write(protocol.OKResponse())
	case protocol.ReqStop:
		// TODO: Implement stop model logic
		rw.Write(protocol.OKResponse())
	case protocol.ReqStatus:
		// TODO: Implement status check logic
		rw.Write(protocol.OKResponse())
	case protocol.ReqChat:
		// TODO: Implement chat logic
		rw.Write(protocol.OKResponse())
	default:
		rw.Write(protocol.ErrorResponse("unknown request type"))
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
