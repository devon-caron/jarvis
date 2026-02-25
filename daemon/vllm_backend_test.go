package daemon

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

func TestParseSSEStream(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: {"choices":[{"delta":{"content":"!"}}]}

data: [DONE]

`
	var tokens []string
	err := parseSSEStream(strings.NewReader(sseData), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}

	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d: %v", len(tokens), tokens)
	}
	if tokens[0] != "Hello" || tokens[1] != " world" || tokens[2] != "!" {
		t.Errorf("tokens = %v, want [Hello, ' world', !]", tokens)
	}
}

func TestParseSSEStream_EmptyContent(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":""}}]}

data: {"choices":[{"delta":{"content":"tok"}}]}

data: [DONE]

`
	var tokens []string
	err := parseSSEStream(strings.NewReader(sseData), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}

	// Empty content should be skipped
	if len(tokens) != 1 || tokens[0] != "tok" {
		t.Errorf("tokens = %v, want [tok]", tokens)
	}
}

func TestParseSSEStream_MalformedJSON(t *testing.T) {
	sseData := `data: not-json

data: {"choices":[{"delta":{"content":"ok"}}]}

data: [DONE]

`
	var tokens []string
	err := parseSSEStream(strings.NewReader(sseData), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}

	// Malformed line should be skipped, valid token received
	if len(tokens) != 1 || tokens[0] != "ok" {
		t.Errorf("tokens = %v, want [ok]", tokens)
	}
}

func TestParseSSEStream_NoData(t *testing.T) {
	sseData := `data: [DONE]

`
	var tokens []string
	err := parseSSEStream(strings.NewReader(sseData), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}

	if len(tokens) != 0 {
		t.Errorf("expected 0 tokens, got %v", tokens)
	}
}

func TestParseSSEStream_FinishReason(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}

data: [DONE]

`
	var tokens []string
	err := parseSSEStream(strings.NewReader(sseData), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSEStream: %v", err)
	}

	if len(tokens) != 1 || tokens[0] != "Hi" {
		t.Errorf("tokens = %v, want [Hi]", tokens)
	}
}

func TestVLLMBackend_RunChat_SSE(t *testing.T) {
	// Mock vLLM server that returns SSE chat completions
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
			`data: {"choices":[{"delta":{"content":" world"}}]}`,
			`data: [DONE]`,
		}
		for _, chunk := range chunks {
			fmt.Fprintln(w, chunk)
			fmt.Fprintln(w) // blank line between events
		}
	}))
	defer server.Close()

	// Extract port from test server URL
	parts := strings.Split(server.URL, ":")
	port := 0
	fmt.Sscanf(parts[len(parts)-1], "%d", &port)

	cfg := config.Defaults()
	v := &VLLMBackend{
		path:       "/test/model.gguf",
		port:       port,
		loaded:     true,
		cfg:        cfg,
		httpClient: server.Client(),
	}

	var tokens []string
	err := v.RunChat(
		t.Context(),
		[]protocol.ChatMessage{{Role: "user", Content: "hello"}},
		protocol.InferenceOpts{},
		func(token string) { tokens = append(tokens, token) },
	)
	if err != nil {
		t.Fatalf("RunChat: %v", err)
	}

	if len(tokens) != 2 || tokens[0] != "Hello" || tokens[1] != " world" {
		t.Errorf("tokens = %v, want [Hello, ' world']", tokens)
	}
}

func TestVLLMBackend_RunChat_NotLoaded(t *testing.T) {
	cfg := config.Defaults()
	v := &VLLMBackend{cfg: cfg, httpClient: &http.Client{}}

	err := v.RunChat(
		t.Context(),
		[]protocol.ChatMessage{{Role: "user", Content: "hello"}},
		protocol.InferenceOpts{},
		func(string) {},
	)
	if err == nil {
		t.Error("expected error when not loaded")
	}
}

func TestVLLMBackend_IsLoaded(t *testing.T) {
	v := &VLLMBackend{}
	if v.IsLoaded() {
		t.Error("should not be loaded initially")
	}

	v.loaded = true
	if !v.IsLoaded() {
		t.Error("should be loaded")
	}
}

func TestVLLMBackend_ModelPath(t *testing.T) {
	v := &VLLMBackend{path: "/test/model.gguf"}
	if v.ModelPath() != "/test/model.gguf" {
		t.Errorf("ModelPath() = %q", v.ModelPath())
	}
}

func TestVLLMBackend_GetStatus_NotLoaded(t *testing.T) {
	v := &VLLMBackend{}
	_, err := v.GetStatus()
	if err == nil {
		t.Error("expected error when not loaded")
	}
}

func TestVLLMBackend_GetStatus_Loaded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `{"data":[]}`)
	}))
	defer server.Close()

	parts := strings.Split(server.URL, ":")
	port := 0
	fmt.Sscanf(parts[len(parts)-1], "%d", &port)

	v := &VLLMBackend{
		path:       "/test/model.gguf",
		port:       port,
		loaded:     true,
		httpClient: server.Client(),
	}

	status, err := v.GetStatus()
	if err != nil {
		t.Fatalf("GetStatus: %v", err)
	}
	if status.ModelPath != "/test/model.gguf" {
		t.Errorf("ModelPath = %q", status.ModelPath)
	}
}

func TestIsVRAMError(t *testing.T) {
	tests := []struct {
		stderr string
		want   bool
	}{
		{"CUDA out of memory", true},
		{"torch.OutOfMemoryError: CUDA out of memory", true},
		{"cudaMalloc failed: out of memory", true},
		{"failed to allocate memory", true},
		{"normal startup message", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isVRAMError(tt.stderr); got != tt.want {
			t.Errorf("isVRAMError(%q) = %v, want %v", tt.stderr, got, tt.want)
		}
	}
}

func TestVllmErrorMsg(t *testing.T) {
	tests := []struct {
		stderr string
		want   string
	}{
		{"", ""},
		{"error line\n", "error line"},
		{"first\nsecond\nthird", "third"},
		{"first\nsecond\n\n", "second"},
	}
	for _, tt := range tests {
		if got := vllmErrorMsg(tt.stderr); got != tt.want {
			t.Errorf("vllmErrorMsg(%q) = %q, want %q", tt.stderr, got, tt.want)
		}
	}
}

func TestFindFreePort(t *testing.T) {
	port, err := findFreePort()
	if err != nil {
		t.Fatalf("findFreePort: %v", err)
	}
	if port <= 0 || port > 65535 {
		t.Errorf("invalid port: %d", port)
	}
}
