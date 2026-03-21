package daemon

import (
	"context"
	"encoding/json"
	"io"

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
		// TODO: Implement load model logic
		rw.Write(protocol.OKResponse())
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
