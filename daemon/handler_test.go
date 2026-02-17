package daemon

import (
	"bytes"
	"context"
	"encoding/json"
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
	backend := &mockBackend{}
	mgr := NewModelManager(backend)
	cfg := config.Defaults()
	stopCh := make(chan struct{}, 1)
	h := NewHandler(mgr, cfg, nil, stopCh)
	return h, backend
}

func TestHandler_Chat(t *testing.T) {
	h, backend := newTestHandler(t)
	backend.LoadModel("/model.gguf", -1)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	req := &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hello"}},
		},
	}

	h.Handle(req, rw)

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

	h.Handle(req, rw)

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

	h.Handle(&protocol.Request{Type: protocol.ReqChat}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for nil chat payload")
	}
}

func TestHandler_Chat_SystemPrompt(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)
	cfg := config.Defaults()
	cfg.SystemPrompt = "Be helpful"
	h := NewHandler(mgr, cfg, nil, make(chan struct{}, 1))

	backend.LoadModel("/model.gguf", -1)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{
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
	mgr := NewModelManager(backend)
	cfg := config.Defaults()
	searcher := &mockSearcher{
		results: []search.Result{
			{Title: "Test", URL: "http://test.com", Description: "A test result"},
		},
	}
	h := NewHandler(mgr, cfg, searcher, make(chan struct{}, 1))

	backend.LoadModel("/model.gguf", -1)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{
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

func TestHandler_Load(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "/model.gguf", GPULayers: -1},
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

	h.Handle(&protocol.Request{Type: protocol.ReqLoad}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for nil load payload")
	}
}

func TestHandler_Load_AliasResolution(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)
	cfg := config.Defaults()
	cfg.Models["big"] = "/path/to/big.gguf"
	h := NewHandler(mgr, cfg, nil, make(chan struct{}, 1))

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{ModelPath: "big", GPULayers: -1},
	}, rw)

	if backend.path != "/path/to/big.gguf" {
		t.Errorf("loaded path = %q, want /path/to/big.gguf", backend.path)
	}
}

func TestHandler_Unload(t *testing.T) {
	h, backend := newTestHandler(t)
	backend.LoadModel("/model.gguf", -1)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{Type: protocol.ReqUnload}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespOK {
		t.Errorf("expected OK, got %v", responses)
	}
}

func TestHandler_Unload_NoModel(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{Type: protocol.ReqUnload}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error when unloading with no model")
	}
}

func TestHandler_Status(t *testing.T) {
	h, backend := newTestHandler(t)
	backend.LoadModel("/model.gguf", -1)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{Type: protocol.ReqStatus}, rw)

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
	if status.ModelPath != "/model.gguf" {
		t.Errorf("ModelPath = %q", status.ModelPath)
	}
}

func TestHandler_Status_NoModel(t *testing.T) {
	h, _ := newTestHandler(t)

	var buf bytes.Buffer
	rw := NewResponseWriter(&buf)

	h.Handle(&protocol.Request{Type: protocol.ReqStatus}, rw)

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

	h.Handle(&protocol.Request{Type: protocol.ReqStop}, rw)

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

	h.Handle(&protocol.Request{Type: "bogus"}, rw)

	responses := readResponses(t, &buf)
	if len(responses) != 1 || responses[0].Type != protocol.RespError {
		t.Error("expected error for unknown type")
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
