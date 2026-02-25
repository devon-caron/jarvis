package daemon

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

func TestNewLlamaServerBackend(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaServerBackend(cfg)
	if b == nil {
		t.Fatal("NewLlamaServerBackend returned nil")
	}
}

func TestLlamaServerBackend_IsLoaded_Initially(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaServerBackend(cfg).(*LlamaServerBackend)
	if b.IsLoaded() {
		t.Error("should not be loaded initially")
	}
}

func TestLlamaServerBackend_ModelPath_Initially(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaServerBackend(cfg).(*LlamaServerBackend)
	if b.ModelPath() != "" {
		t.Errorf("ModelPath = %q, want empty", b.ModelPath())
	}
}

func TestLlamaServerBackend_UnloadModel_WhenNotLoaded(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaServerBackend(cfg).(*LlamaServerBackend)
	if err := b.UnloadModel(); err != nil {
		t.Errorf("UnloadModel on unloaded backend should not error: %v", err)
	}
}

func TestLlamaServerBackend_GetStatus_WhenNotLoaded(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaServerBackend(cfg).(*LlamaServerBackend)
	_, err := b.GetStatus()
	if err == nil {
		t.Error("GetStatus should error when no model loaded")
	}
}

func TestLlamaServerBackend_LoadModel_BadPath(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaServerBackend(cfg).(*LlamaServerBackend)
	err := b.LoadModel("/nonexistent/model.gguf", []int{0}, 8192)
	if err == nil {
		t.Error("LoadModel should error for nonexistent file")
	}
	if !strings.Contains(err.Error(), "model file not found") {
		t.Errorf("expected 'model file not found' error, got: %v", err)
	}
	if b.IsLoaded() {
		t.Error("should not be loaded after error")
	}
}

func TestServerErrorMsg(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"empty", "", ""},
		{"whitespace only", "   \n  ", ""},
		{"cobra prefix stripped", "Error: failed to load model: no such file", "failed to load model: no such file"},
		{"no prefix unchanged", "some error occurred", "some error occurred"},
		{"multiline takes last", "line one\nError: last error\n", "last error"},
		{"multiline with trailing blank", "line one\nError: last error\n\n  \n", "last error"},
		{"nested wrapping returns last line", "Error: outer\nError: inner detail", "inner detail"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := serverErrorMsg(tc.input)
			if got != tc.want {
				t.Errorf("serverErrorMsg(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestIsVRAMError(t *testing.T) {
	tests := []struct {
		name   string
		stderr string
		want   bool
	}{
		{"empty", "", false},
		{"unrelated error", "Error: failed to load model: bad file\n", false},
		{
			"cudaMalloc OOM",
			"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1024.00 MiB on device 0: cudaMalloc failed: out of memory\nError: failed to load\n",
			true,
		},
		{
			"generic out of memory",
			"some context\nout of memory\nError: load failed\n",
			true,
		},
		{
			"failed to allocate buffer",
			"alloc_tensor_range: failed to allocate CUDA0 buffer of size 2147483648\nError: load failed\n",
			true,
		},
		{
			"unable to allocate",
			"unable to allocate CUDA0 buffer of size 2147483648\n",
			true,
		},
		{
			"case insensitive",
			"CUDAMALLOC FAILED: OUT OF MEMORY\n",
			true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := isVRAMError(tc.stderr)
			if got != tc.want {
				t.Errorf("isVRAMError(...) = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestValidateGGUF(t *testing.T) {
	dir := t.TempDir()

	tests := []struct {
		name    string
		content []byte
		wantErr string
	}{
		{
			name:    "valid v3 header",
			content: append([]byte("GGUF"), 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			wantErr: "",
		},
		{
			name:    "valid v2 header",
			content: append([]byte("GGUF"), 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			wantErr: "",
		},
		{
			name:    "bad magic",
			content: append([]byte("NOPE"), 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			wantErr: "not a valid GGUF model file",
		},
		{
			name:    "unsupported version",
			content: append([]byte("GGUF"), 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
			wantErr: "unsupported GGUF version",
		},
		{
			name:    "too small",
			content: []byte("GGU"),
			wantErr: "file too small",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".gguf")
			os.WriteFile(path, tc.content, 0644)
			err := validateGGUF(path)
			if tc.wantErr == "" {
				if err != nil {
					t.Errorf("expected no error, got: %v", err)
				}
			} else {
				if err == nil {
					t.Error("expected error, got nil")
				} else if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("expected error containing %q, got: %v", tc.wantErr, err)
				}
			}
		})
	}

	t.Run("nonexistent file", func(t *testing.T) {
		err := validateGGUF("/nonexistent/file.gguf")
		if err == nil || !strings.Contains(err.Error(), "model file not found") {
			t.Errorf("expected 'model file not found', got: %v", err)
		}
	})
}

func TestAllocatePort(t *testing.T) {
	port, err := allocatePort()
	if err != nil {
		t.Fatalf("allocatePort: %v", err)
	}
	if port <= 0 {
		t.Errorf("expected positive port, got %d", port)
	}
}

func TestParseSSE(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: {"choices":[{"delta":{"content":"!"}}]}
data: [DONE]
`
	var tokens []string
	err := parseSSE(strings.NewReader(input), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSE: %v", err)
	}
	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d: %v", len(tokens), tokens)
	}
	if tokens[0] != "Hello" || tokens[1] != " world" || tokens[2] != "!" {
		t.Errorf("tokens = %v, want [Hello, world, !]", tokens)
	}
}

func TestParseSSE_EmptyContent(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":""}}]}
data: {"choices":[{"delta":{"content":"hi"}}]}
data: [DONE]
`
	var tokens []string
	err := parseSSE(strings.NewReader(input), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSE: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != "hi" {
		t.Errorf("tokens = %v, want [hi]", tokens)
	}
}

func TestParseSSE_NonDataLines(t *testing.T) {
	input := `: comment
event: message
data: {"choices":[{"delta":{"content":"ok"}}]}

data: [DONE]
`
	var tokens []string
	err := parseSSE(strings.NewReader(input), func(token string) {
		tokens = append(tokens, token)
	})
	if err != nil {
		t.Fatalf("parseSSE: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != "ok" {
		t.Errorf("tokens = %v, want [ok]", tokens)
	}
}

func TestRunChat_SSE(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"Hello"}}]}`)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":" world!"}}]}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer ts.Close()

	// Extract port from test server URL.
	parts := strings.Split(ts.URL, ":")
	var port int
	fmt.Sscanf(parts[len(parts)-1], "%d", &port)

	cfg := config.Defaults()
	b := &LlamaServerBackend{
		path:       "/test.gguf",
		port:       port,
		loaded:     true,
		cfg:        cfg,
		httpClient: ts.Client(),
	}

	var tokens []string
	err := b.RunChat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(token string) { tokens = append(tokens, token) },
	)
	if err != nil {
		t.Fatalf("RunChat: %v", err)
	}
	if len(tokens) != 2 || tokens[0] != "Hello" || tokens[1] != " world!" {
		t.Errorf("tokens = %v, want [Hello,  world!]", tokens)
	}
}

func TestRunChat_HTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	}))
	defer ts.Close()

	parts := strings.Split(ts.URL, ":")
	var port int
	fmt.Sscanf(parts[len(parts)-1], "%d", &port)

	cfg := config.Defaults()
	b := &LlamaServerBackend{
		path:       "/test.gguf",
		port:       port,
		loaded:     true,
		cfg:        cfg,
		httpClient: ts.Client(),
	}

	err := b.RunChat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {})
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected error to mention status code, got: %v", err)
	}
}

func TestRunChat_NotLoaded(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaServerBackend(cfg).(*LlamaServerBackend)

	err := b.RunChat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {})
	if err == nil {
		t.Error("expected error when not loaded")
	}
}

func TestGetStatus_Loaded(t *testing.T) {
	b := &LlamaServerBackend{
		path:   "/test.gguf",
		loaded: true,
		gpus:   []int{0, 1},
	}

	status, err := b.GetStatus()
	if err != nil {
		t.Fatalf("GetStatus: %v", err)
	}
	if status.ModelPath != "/test.gguf" {
		t.Errorf("ModelPath = %q, want /test.gguf", status.ModelPath)
	}
	if len(status.GPUs) != 2 {
		t.Fatalf("expected 2 GPUs, got %d", len(status.GPUs))
	}
	if status.GPUs[0].DeviceID != 0 || status.GPUs[1].DeviceID != 1 {
		t.Errorf("GPU IDs = [%d, %d], want [0, 1]", status.GPUs[0].DeviceID, status.GPUs[1].DeviceID)
	}
}
