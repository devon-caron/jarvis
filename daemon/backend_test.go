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

// TestNewServerBackend verifies NewServerBackend returns a non-nil ModelBackend
// instance when given a default config.
func TestNewServerBackend(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg)
	if b == nil {
		t.Fatal("NewServerBackend returned nil")
	}
}

// TestBackend_IsLoaded_Initially verifies a freshly created Backend reports
// IsLoaded() as false before any model is loaded.
func TestBackend_IsLoaded_Initially(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)
	if b.IsLoaded() {
		t.Error("should not be loaded initially")
	}
}

// TestBackend_ModelPath_Initially verifies a freshly created Backend returns
// an empty ModelPath() before any model is loaded.
func TestBackend_ModelPath_Initially(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)
	if b.ModelPath() != "" {
		t.Errorf("ModelPath = %q, want empty", b.ModelPath())
	}
}

// TestBackend_UnloadModel_WhenNotLoaded verifies calling UnloadModel on a
// backend with no loaded model is a graceful no-op (returns no error).
func TestBackend_UnloadModel_WhenNotLoaded(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)
	if err := b.UnloadModel(); err != nil {
		t.Errorf("UnloadModel on unloaded backend should not error: %v", err)
	}
}

// TestBackend_LoadModel_BadPath verifies LoadModel returns a "model file not
// found" error for a nonexistent file path and leaves the backend unloaded.
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

// TestIsVRAMError is a table-driven test verifying isVRAMError correctly
// identifies CUDA/GPU out-of-memory patterns in stderr output (cudaMalloc,
// out of memory, failed to allocate, unable to allocate) with case-insensitive
// matching, and correctly rejects unrelated error messages.
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

// TestValidateGGUF is a table-driven test verifying validateGGUF accepts valid
// GGUF v2/v3 headers and rejects files with bad magic bytes, unsupported
// versions, files that are too small, and nonexistent paths.
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

// TestAllocatePort verifies allocatePort returns a positive ephemeral port
// number without error.
func TestAllocatePort(t *testing.T) {
	port, err := allocatePort()
	if err != nil {
		t.Fatalf("allocatePort: %v", err)
	}
	if port <= 0 {
		t.Errorf("expected positive port, got %d", port)
	}
}

// TestParseSSE verifies parseSSE correctly parses an OpenAI-compatible SSE
// stream, extracting content tokens from "data:" lines and stopping at "[DONE]".
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

// TestParseSSE_EmptyContent verifies parseSSE skips SSE chunks with empty
// content strings and only delivers non-empty tokens.
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

// TestParseSSE_NonDataLines verifies parseSSE ignores non-data lines (comments,
// event types, blank lines) and only extracts content from "data:" lines.
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

// TestRunChat_SSE is an integration test using httptest to verify Backend.RunChat
// sends a request to /v1/chat/completions and streams back tokens from the SSE response.
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

// TestRunChat_HTTPError verifies Backend.RunChat returns an error containing
// the HTTP status code when the server responds with 500 Internal Server Error.
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

// TestRunChat_NotLoaded verifies Backend.RunChat returns an error when called
// on a backend that has no model loaded.
func TestRunChat_NotLoaded(t *testing.T) {
	cfg := config.Defaults()
	b := NewServerBackend(cfg).(*Backend)

	err := b.RunChat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {})
	if err == nil {
		t.Error("expected error when not loaded")
	}
}

// TestNormalizeSplitMode is a table-driven test verifying NormalizeSplitMode
// normalizes split mode strings (layer/l, row/r, none, empty) to canonical
// forms, handles case insensitivity and whitespace, and rejects invalid values.
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
