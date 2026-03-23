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

func TestNewServerBackend(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg)
	if b == nil {
		t.Fatal("NewServerBackend returned nil")
	}
}

func TestBackend_IsLoaded_Initially(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)
	if b.IsLoaded() {
		t.Error("should not be loaded initially")
	}
}

func TestBackend_ModelPath_Initially(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)
	if b.ModelPath() != "" {
		t.Errorf("ModelPath = %q, want empty", b.ModelPath())
	}
}

func TestBackend_UnloadModel_WhenNotLoaded(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)
	if err := b.UnloadModel(); err != nil {
		t.Errorf("UnloadModel on unloaded backend should not error: %v", err)
	}
}

func TestBackend_LoadModel_BadPath(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)
	err := b.LoadModel(context.Background(), "/nonexistent/model.gguf", []int{0}, LoadOpts{ContextSize: 8192})
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
	b := &Backend{
		path:       "/test.gguf",
		port:       port,
		loaded:     true,
		config:     cfg,
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
	b := &Backend{
		path:       "/test.gguf",
		port:       port,
		loaded:     true,
		config:     cfg,
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
	b := NewServerBackend(cfg).(*Backend)

	err := b.RunChat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {})
	if err == nil {
		t.Error("expected error when not loaded")
	}
}

func TestNormalizeSplitMode(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{"", "", false},
		{"none", "", false},
		{"l", "layer", false},
		{"layer", "layer", false},
		{"LAYER", "layer", false},
		{"r", "row", false},
		{"row", "row", false},
		{"ROW", "row", false},
		{" layer ", "layer", false},
		{"invalid", "", true},
		{"x", "", true},
	}
	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got, err := NormalizeSplitMode(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Errorf("NormalizeSplitMode(%q) expected error, got %q", tc.input, got)
				}
				return
			}
			if err != nil {
				t.Errorf("NormalizeSplitMode(%q) unexpected error: %v", tc.input, err)
				return
			}
			if got != tc.want {
				t.Errorf("NormalizeSplitMode(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
