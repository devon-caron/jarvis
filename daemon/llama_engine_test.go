package daemon

import (
	"testing"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

func TestBuildTensorSplit(t *testing.T) {
	tests := []struct {
		name     string
		gpus     []int
		total    int
		expected string
	}{
		{"GPU 1 only (the bug)", []int{1}, 2, "0,1"},
		{"GPU 0 only", []int{0}, 2, "1,0"},
		{"Both GPUs", []int{0, 1}, 2, "1,1"},
		{"GPU 1 of 4", []int{1}, 4, "0,1,0,0"},
		{"Empty GPU list", []int{}, 2, ""},
		{"Zero total GPUs", []int{0}, 0, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildTensorSplit(tt.gpus, tt.total)
			if got != tt.expected {
				t.Errorf("buildTensorSplit(%v, %d) = %q, want %q", tt.gpus, tt.total, got, tt.expected)
			}
		})
	}
}

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
