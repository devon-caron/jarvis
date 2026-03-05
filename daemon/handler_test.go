package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
	"github.com/devon-caron/jarvis/search"
)

// mockSearcher implements search.Searcher for testing.
type mockSearcher struct {
	results []search.Result
	err     error
}

func (m *mockSearcher) Search(ctx context.Context, query string) ([]search.Result, error) {
	return m.results, m.err
}

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
	registry := NewModelRegistry(cfg, factory)
	stopCh := make(chan struct{}, 1)
	h := NewHandler(registry, cfg, nil, stopCh)
	return h, backend
}

func TestHandler_Chat(t *testing.T) {
	h, backend := newTestHandler(t)
	backend.LoadModel(context.Background(), "/model.gguf", []int{0}, LoadOpts{})
	h.Registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	req := &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hello"}},
		},
	}

	h.Handle(context.Background(),req, rw)

	responses := readResponses(t, &buf)
	if len(responses) < 2 {
		t.Fatalf("expected at least 2 responses, got %d", len(responses))
	}

	// Should have delta responses followed by done
	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("last response type = %q, want done", lastResp.Type)
	}

	// Check deltas
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

	h.Handle(context.Background(),req, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 {
		t.Fatalf("expected 1 response, got %d", len(responses))
	}
	if responses[0].Type != protocol.RespError {
		t.Errorf("response type = %q, want error", responses[0].Type)
	}
}

func TestHandler_Chat_NilPayload(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqChat}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for nil chat payload")
	}
}

func TestHandler_Chat_SystemPrompt(t *testing.T) {
	backend := &mockBackend{}
	cfg := config.Defaults()
	cfg.SystemPrompt = "Be helpful"
	factory := func(c *config.Config) ModelBackend { return backend }
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, nil, make(chan struct{}, 1))

	registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
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

func TestHandler_Chat_WebSearch(t *testing.T) {
	backend := &mockBackend{}
	cfg := config.Defaults()
	searcher := &mockSearcher{
		results: []search.Result{
			{Title: "Test", URL: "http://test.com", Description: "A test result"},
		},
	}
	factory := func(c *config.Config) ModelBackend { return backend }
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, searcher, make(chan struct{}, 1))

	registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages:  []protocol.ChatMessage{{Role: "user", Content: "search test"}},
			WebSearch: true,
		},
	}, rw)

	responses := readResponses(t, &buf)
	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("expected done, got %q", lastResp.Type)
	}
}

func TestHandler_Chat_WebSearch_ZeroResults(t *testing.T) {
	backend := &mockBackend{}
	cfg := config.Defaults()
	searcher := &mockSearcher{results: []search.Result{}}
	factory := func(c *config.Config) ModelBackend { return backend }
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, searcher, make(chan struct{}, 1))

	registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages:  []protocol.ChatMessage{{Role: "user", Content: "search test"}},
			WebSearch: true,
		},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for zero search results")
	}
}

func TestHandler_Chat_WebSearch_Error(t *testing.T) {
	backend := &mockBackend{}
	cfg := config.Defaults()
	searcher := &mockSearcher{err: errors.New("search failed")}
	factory := func(c *config.Config) ModelBackend { return backend }
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, searcher, make(chan struct{}, 1))

	registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages:  []protocol.ChatMessage{{Role: "user", Content: "search test"}},
			WebSearch: true,
		},
	}, rw)

	responses := readResponses(t, &buf)
	// Should get an error response followed by chat continuing (error is non-fatal in this path)
	hasError := false
	for _, r := range responses {
		if r.Type == protocol.RespError {
			hasError = true
		}
	}
	if !hasError {
		t.Error("expected error response for search failure")
	}
}

func TestHandler_Chat_WithModelName(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	cfg := config.Defaults()
	idx := 0
	backends := []*mockBackend{b1, b2}
	factory := func(c *config.Config) ModelBackend {
		b := backends[idx]
		idx++
		return b
	}
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, nil, make(chan struct{}, 1))

	registry.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	registry.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hi"}},
			Model:    "m2",
		},
	}, rw)

	responses := readResponses(t, &buf)
	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("expected done, got %q", lastResp.Type)
	}
	if b2.chatCalls.Load() != 1 {
		t.Errorf("m2 chat calls = %d, want 1", b2.chatCalls.Load())
	}
}

func TestHandler_Load(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}
}

func TestHandler_Load_NilPayload(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqLoad}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for nil load payload")
	}
}

func TestHandler_Load_AliasResolution(t *testing.T) {
	// Write a test config with the "big" model so the handler reload finds it.
	tmpDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", tmpDir)
	cfg := config.Defaults()
	cfg.Models["big"] = config.ModelEntry{Path: "/path/to/big.gguf"}
	cfg.Save(filepath.Join(tmpDir, "jarvis", "config.yaml"))

	backend := &mockBackend{}
	factory := func(c *config.Config) ModelBackend { return backend }
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, nil, make(chan struct{}, 1))

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
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

func TestHandler_Unload(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type:   protocol.ReqUnload,
		Unload: &protocol.UnloadRequest{Name: "test"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK, got %v", responses)
	}
}

func TestHandler_Unload_NoModel(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqUnload}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error when unloading with no model")
	}
}

func TestHandler_Status(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqStatus}, rw)

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
	if len(status.Models) != 1 {
		t.Fatalf("expected 1 model in status, got %d", len(status.Models))
	}
	if status.Models[0].Name != "test" {
		t.Errorf("model name = %q, want test", status.Models[0].Name)
	}
}

func TestHandler_Status_NoModel(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqStatus}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespStatus {
		t.Fatal("expected status response")
	}
	if responses[0].Status.ModelLoaded {
		t.Error("ModelLoaded should be false")
	}
}

func TestHandler_Stop(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqStop}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Error("expected OK response for stop")
	}

	// StopCh should have a signal
	select {
	case <-h.StopCh:
		// Good
	default:
		t.Error("StopCh should have a signal")
	}
}

func TestHandler_UnknownType(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: "bogus"}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for unknown type")
	}
}

func TestHandler_Load_WithTimeout(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", Timeout: "30m"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	// Verify model is loaded with timeout
	models := h.Registry.Status()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].Timeout != "30m0s" {
		t.Errorf("Timeout = %q, want 30m0s", models[0].Timeout)
	}
}

func TestHandler_Load_InvalidTimeout(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", Timeout: "badvalue"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for invalid timeout")
	}
}

func TestHandler_Load_DefaultTimeout(t *testing.T) {
	backend := &mockBackend{}
	cfg := config.Defaults()
	cfg.DefaultTimeout = "15m"
	factory := func(c *config.Config) ModelBackend { return backend }
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, nil, make(chan struct{}, 1))

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	models := registry.Status()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].Timeout != "15m0s" {
		t.Errorf("Timeout = %q, want 15m0s", models[0].Timeout)
	}
}

func TestHandler_Load_WithGPUs(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", Name: "test", GPUs: []int{1, 2}},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}
}

func TestHandler_Load_DefaultName(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	// No Name field — should default to ModelPath
	h.Handle(context.Background(),&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf"},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK response, got %v", responses)
	}

	models := h.Registry.Status()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].Name != "/model.gguf" {
		t.Errorf("Name = %q, want /model.gguf", models[0].Name)
	}
}

func TestHandler_Unload_ByGPU(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	cfg := config.Defaults()
	idx := 0
	backends := []*mockBackend{b1, b2}
	factory := func(c *config.Config) ModelBackend {
		b := backends[idx]
		idx++
		return b
	}
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, nil, make(chan struct{}, 1))

	registry.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	registry.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	gpu := 0
	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)
	h.Handle(context.Background(),&protocol.Request{
		Type:   protocol.ReqUnload,
		Unload: &protocol.UnloadRequest{GPU: &gpu},
	}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK, got %v", responses)
	}

	models := registry.Status()
	if len(models) != 1 || models[0].Name != "m2" {
		t.Errorf("expected only m2 remaining after unloading GPU 0, got %v", models)
	}
}

func TestHandler_Unload_NilPayload(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	// nil unload payload — should still work with single model
	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqUnload}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK, got %v", responses)
	}
}

func TestHandler_Status_MultiModel(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	cfg := config.Defaults()
	idx := 0
	backends := []*mockBackend{b1, b2}
	factory := func(c *config.Config) ModelBackend {
		b := backends[idx]
		idx++
		return b
	}
	registry := NewModelRegistry(cfg, factory)
	h := NewHandler(registry, cfg, nil, make(chan struct{}, 1))

	registry.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	registry.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{Type: protocol.ReqStatus}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespStatus {
		t.Fatalf("expected status response, got %v", responses)
	}

	status := responses[0].Status
	if len(status.Models) != 2 {
		t.Errorf("expected 2 models, got %d", len(status.Models))
	}
	// With multiple models, single-model compat fields should be empty
	if status.ModelPath != "" {
		t.Errorf("ModelPath should be empty for multi-model, got %q", status.ModelPath)
	}
}

func TestHandler_Load_GPUAssignment_SingleGPU(t *testing.T) {
	h, backend := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
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

func TestHandler_Load_GPUAssignment_MultiGPU(t *testing.T) {
	h, backend := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(context.Background(),&protocol.Request{
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

func TestResponseWriter(t *testing.T) {
	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	rw.Write(protocol.DeltaResponse("hi"))
	rw.Write(protocol.DoneResponse())

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

func TestHandler_Chat_ClearContext(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

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

func TestHandler_Chat_ShellPID(t *testing.T) {
	h, _ := newTestHandler(t)
	h.Registry.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

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
