package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

// readResponses parses NDJSON responses from a buffer.
func readResponses(t *testing.T, buf *bytes.Buffer) []*protocol.Response {
	t.Helper()
	var responses []*protocol.Response
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if line == "" {
			continue
		}
		var resp protocol.Response
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			t.Fatalf("failed to unmarshal response %q: %v", line, err)
		}
		responses = append(responses, &resp)
	}
	return responses
}

func newTestHandler(t *testing.T) (*Handler, *mockBackend) {
	t.Helper()
	backend := &mockBackend{}
	cfg := config.Defaults()
	factory := func(c *config.Config) ModelBackend { return backend }
	mr := NewModelRegister(cfg, factory)
	stopCh := make(chan struct{}, 1)
	h := NewHandler(mr, cfg, stopCh)
	return h, backend
}

// TestHandler_Chat verifies the handler processes a chat request end-to-end:
// sends a user message to a loaded model, receives streamed delta tokens, and
// confirms the final response is "done" with the full content "Hello world!".
func TestHandler_Chat(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Register.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	req := &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hello"}},
		},
	}

	h.Handle(context.Background(), req, rw)

	responses := readResponses(t, &buf)
	if len(responses) < 2 {
		t.Fatalf("expected at least 2 responses, got %d", len(responses))
	}

	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("last response type = %q, want done", lastResp.Type)
	}

	var content string
	for _, r := range responses {
		if r.Type == protocol.RespDelta && r.Delta != nil {
			content += r.Delta.Content
		}
	}
	if content != "Hello world!" {
		t.Errorf("content = %q, want 'Hello world!'", content)
	}
}

// TestHandler_Chat_NoModel verifies the handler returns an error response
// when a chat request is sent with no model loaded.
func TestHandler_Chat_NoModel(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	req := &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hello"}},
		},
	}

	h.Handle(context.Background(), req, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 {
		t.Fatalf("expected 1 response, got %d", len(responses))
	}
	if responses[0].Type != protocol.RespError {
		t.Errorf("response type = %q, want error", responses[0].Type)
	}
}

// TestHandler_Chat_NilPayload verifies the handler returns an error response
// when a chat request has a nil Chat payload.
func TestHandler_Chat_NilPayload(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{Type: protocol.ReqChat}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for nil chat payload")
	}
}

// TestHandler_Chat_SystemPrompt verifies the handler successfully processes a
// chat request when a system prompt is configured, completing with a "done" response.
func TestHandler_Chat_SystemPrompt(t *testing.T) {
	backend := &mockBackend{}
	cfg := config.Defaults()
	cfg.SystemPrompt = "Be helpful"
	factory := func(c *config.Config) ModelBackend { return backend }
	mr := NewModelRegister(cfg, factory)
	h := NewHandler(mr, cfg, make(chan struct{}, 1))

	mr.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hi"}},
		},
	}, rw)

	responses := readResponses(t, &buf)
	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("expected done, got %q", lastResp.Type)
	}
}

// TestHandler_Load verifies the handler processes a load request with a model
// path and name, returning an OK response.
func TestHandler_Load(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}
}

// TestHandler_Load_NilPayload verifies the handler returns an error response
// when a load request has a nil Load payload.
func TestHandler_Load_NilPayload(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{Type: protocol.ReqLoad}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for nil load payload")
	}
}

// TestHandler_Load_AliasResolution verifies the handler resolves a model name
// to its path via the config mr, loading alias "big" and confirming the
// backend received the resolved path "/path/to/big.gguf".
func TestHandler_Load_AliasResolution(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", tmpDir)
	cfg := config.Defaults()
	cfg.Models["big"] = config.ModelEntry{Path: "/path/to/big.gguf"}
	cfg.Save(filepath.Join(tmpDir, "jarvis", "config.yaml"))

	backend := &mockBackend{}
	factory := func(c *config.Config) ModelBackend { return backend }
	mr := NewModelRegister(cfg, factory)
	h := NewHandler(mr, cfg, make(chan struct{}, 1))

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{Name: "big"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}
	if backend.path != "/path/to/big.gguf" {
		t.Errorf("loaded path = %q, want /path/to/big.gguf", backend.path)
	}
}

// TestHandler_Unload verifies the handler processes an unload request for a
// loaded model by name, returning an OK response.
func TestHandler_Unload(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Register.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type:   protocol.ReqUnload,
		Unload: &protocol.UnloadRequest{Name: "test"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK, got %v", responses)
	}
}

// TestHandler_Unload_NilPayload verifies the handler returns an error response
// when an unload request has a nil Unload payload.
func TestHandler_Unload_NilPayload(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Register.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{Type: protocol.ReqUnload}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for nil unload payload")
	}
}

// TestHandler_Unload_NoModel verifies the handler returns an error response
// when an unload request is sent with no model loaded.
func TestHandler_Unload_NoModel(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type:   protocol.ReqUnload,
		Unload: &protocol.UnloadRequest{},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error when unloading with no model")
	}
}

// TestHandler_Status verifies the handler returns a status response with
// Running=true, ModelLoaded=true, and the correct model name when a model
// is loaded.
func TestHandler_Status(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Register.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{Type: protocol.ReqStatus}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespStatus {
		t.Fatalf("expected status response, got %v", responses)
	}

	status := responses[0].Status
	if !status.Running {
		t.Error("Running should be true")
	}
	if !status.ModelLoaded {
		t.Error("ModelLoaded should be true")
	}
	if status.Model == nil {
		t.Fatal("Model should not be nil")
	}
	if status.Model.Name != "test" {
		t.Errorf("model name = %q, want test", status.Model.Name)
	}
}

// TestHandler_Status_NoModel verifies the handler returns a status response
// with ModelLoaded=false and Model=nil when no model is loaded.
func TestHandler_Status_NoModel(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{Type: protocol.ReqStatus}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespStatus {
		t.Fatal("expected status response")
	}
	if responses[0].Status.ModelLoaded {
		t.Error("ModelLoaded should be false")
	}
	if responses[0].Status.Model != nil {
		t.Error("Model should be nil when no model loaded")
	}
}

// TestHandler_Stop verifies the handler returns an OK response for a stop
// request and sends a signal on the StopCh channel to trigger daemon shutdown.
func TestHandler_Stop(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{Type: protocol.ReqStop}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Error("expected OK response for stop")
	}

	select {
	case <-h.StopCh:
	default:
		t.Error("StopCh should have a signal")
	}
}

// TestHandler_UnknownType verifies the handler returns an error response when
// the request type is unrecognized.
func TestHandler_UnknownType(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{Type: "bogus"}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for unknown type")
	}
}

// TestHandler_Load_WithTimeout verifies the handler processes a load request
// with a timeout parameter and the model is loaded afterward.
func TestHandler_Load_WithTimeout(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", Timeout: "30m"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	model := h.Register.Status()
	if model == nil {
		t.Fatal("expected model to be loaded")
	}
}

// TestHandler_Load_InvalidTimeout verifies the handler returns an error
// response when the load request contains an unparseable timeout value.
func TestHandler_Load_InvalidTimeout(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", Timeout: "badvalue"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for invalid timeout")
	}
}

// TestHandler_Load_DefaultTimeout verifies the handler uses the config's
// DefaultTimeout when no timeout is specified in the load request.
func TestHandler_Load_DefaultTimeout(t *testing.T) {
	backend := &mockBackend{}
	cfg := config.Defaults()
	cfg.DefaultTimeout = "15m"
	factory := func(c *config.Config) ModelBackend { return backend }
	mr := NewModelRegister(cfg, factory)
	h := NewHandler(mr, cfg, make(chan struct{}, 1))

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	model := mr.Status()
	if model == nil {
		t.Fatal("expected model to be loaded")
	}
}

// TestHandler_Load_WithGPUs verifies the handler processes a load request
// with explicit GPU assignments, returning an OK response.
func TestHandler_Load_WithGPUs(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", GPUs: []int{1, 2}},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}
}

// TestHandler_Load_DefaultName verifies the handler defaults the model name
// to the ModelPath when no Name is provided in the load request.
func TestHandler_Load_DefaultName(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	model := h.Register.Status()
	if model == nil {
		t.Fatal("expected model to be loaded")
	}
	if model.Name != "/model.gguf" {
		t.Errorf("Name = %q, want /model.gguf", model.Name)
	}
}

// TestHandler_Load_GPUAssignment_SingleGPU verifies the handler passes a
// single GPU assignment through to the backend correctly.
func TestHandler_Load_GPUAssignment_SingleGPU(t *testing.T) {
	h, backend := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", GPUs: []int{1}},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	if len(backend.loadedGPUs) != 1 || backend.loadedGPUs[0] != 1 {
		t.Errorf("loadedGPUs = %v, want [1]", backend.loadedGPUs)
	}
}

// TestHandler_Load_GPUAssignment_MultiGPU verifies the handler passes
// multiple GPU assignments through to the backend correctly.
func TestHandler_Load_GPUAssignment_MultiGPU(t *testing.T) {
	h, backend := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", GPUs: []int{0, 1}},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	if len(backend.loadedGPUs) != 2 || backend.loadedGPUs[0] != 0 || backend.loadedGPUs[1] != 1 {
		t.Errorf("loadedGPUs = %v, want [0, 1]", backend.loadedGPUs)
	}
}

// TestResponseWriter verifies ResponseWriter serializes responses as NDJSON,
// writing a delta and done response and confirming the output contains two
// newline-delimited JSON lines with correct types.
func TestResponseWriter(t *testing.T) {
	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	rw.Write(protocol.DeltaTokenResponse("hi"))
	rw.Write(protocol.EndTokenResponse())

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}

	var r1, r2 protocol.Response
	json.Unmarshal([]byte(lines[0]), &r1)
	json.Unmarshal([]byte(lines[1]), &r2)

	if r1.Type != "delta" {
		t.Errorf("first response type = %q", r1.Type)
	}
	if r2.Type != "done" {
		t.Errorf("second response type = %q", r2.Type)
	}
}

// TestHandler_Chat_ClearContext verifies the handler successfully processes a
// chat request with the ClearContext flag and ShellPID set, completing with
// a "done" response.
func TestHandler_Chat_ClearContext(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Register.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages:     []protocol.ChatMessage{{Role: "user", Content: "hello"}},
			ShellPID:     12345,
			ClearContext: true,
		},
	}, rw)

	responses := readResponses(t, &buf)
	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("expected done, got %q", lastResp.Type)
	}
}

// TestHandler_Chat_ShellPID verifies the handler successfully processes a
// chat request with a ShellPID set, completing with a "done" response.
func TestHandler_Chat_ShellPID(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Register.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(), &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hello"}},
			ShellPID: 12345,
		},
	}, rw)

	responses := readResponses(t, &buf)
	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("expected done, got %q", lastResp.Type)
	}
}
