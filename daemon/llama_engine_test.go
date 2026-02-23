package daemon

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)


func TestNewLlamaBackend(t *testing.T) {
	cfg := config.Defaults()
	b := NewLlamaBackend(cfg)
	if b == nil {
		t.Fatal("NewLlamaBackend returned nil")
	}
	if b.cfg != cfg {
		t.Error("config not set")
	}
}

func TestLlamaBackend_IsLoaded_Initially(t *testing.T) {
	b := NewLlamaBackend(config.Defaults())
	if b.IsLoaded() {
		t.Error("should not be loaded initially")
	}
}

func TestLlamaBackend_ModelPath_Initially(t *testing.T) {
	b := NewLlamaBackend(config.Defaults())
	if b.ModelPath() != "" {
		t.Errorf("ModelPath = %q, want empty", b.ModelPath())
	}
}

func TestLlamaBackend_UnloadModel_WhenNil(t *testing.T) {
	b := NewLlamaBackend(config.Defaults())
	if err := b.UnloadModel(); err != nil {
		t.Errorf("UnloadModel on nil model should not error: %v", err)
	}
}

func TestLlamaBackend_GetStatus_WhenNil(t *testing.T) {
	b := NewLlamaBackend(config.Defaults())
	_, err := b.GetStatus()
	if err == nil {
		t.Error("GetStatus should error when no model loaded")
	}
}

func TestLlamaBackend_LoadModel_BadPath(t *testing.T) {
	b := NewLlamaBackend(config.Defaults())
	err := b.LoadModel("/nonexistent/model.gguf", []int{0})
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

func TestLlamaBackend_LoadModel_BadPath_MultiGPU(t *testing.T) {
	cfg := config.Defaults()
	cfg.ModelOptions.MLock = true
	b := NewLlamaBackend(cfg)
	err := b.LoadModel("/nonexistent/model.gguf", []int{0, 1})
	if err == nil {
		t.Error("LoadModel should error for nonexistent file")
	}
}

func TestLlamaBackend_RunChat_WhenNil(t *testing.T) {
	b := NewLlamaBackend(config.Defaults())
	err := b.RunChat(nil, nil, protocol.InferenceOpts{}, func(s string) {})
	if err == nil {
		t.Error("RunChat should error when no model loaded")
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
