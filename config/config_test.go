package config

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadFrom_ValidConfig writes a complete YAML config to a temp file and
// verifies LoadFrom correctly parses all fields: default_model, default_timeout,
// default_gpu, model entries (path, context_size, flash_attention), inference
// settings (max_tokens, temperature), and search config (provider, max_results).
func TestLoadFrom_ValidConfig(t *testing.T) {
	// Create a temporary file with valid YAML config
	validConfig := `
default_model: "llama-3.1-8b"
default_timeout: "45m"
default_gpu: 0
models:
  llama-3.1-8b:
    path: "/models/llama-3.1-8b.gguf"
    context_size: 4096
    flash_attention: true
    batch_size: 512
model_options:
  gpu_layers: -1
  tensor_split: ""
  mlock: false
  flash_attention: false
  batch_size: 0
inference:
  context_size: 8192
  max_tokens: 2048
  temperature: 0.8
  top_p: 0.95
  top_k: 50
  timeout: 180
system_prompt: "You are a helpful AI assistant."
search:
  provider: "google"
  api_key: "test-api-key"
  max_results: 10
llama_server:
  binary_path: "/usr/local/bin/llama-server"
`

	// Create temporary file
	tmpFile, err := os.CreateTemp("", "test_config_*.yaml")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	// Write valid config to temp file
	if _, err := tmpFile.WriteString(validConfig); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpFile.Close()

	// Load config from temp file
	cfg, err := LoadFrom(tmpFile.Name())
	if err != nil {
		t.Fatalf("LoadFrom failed: %v", err)
	}

	// Verify loaded config contents
	if cfg.DefaultModel != "llama-3.1-8b" {
		t.Errorf("Expected DefaultModel 'llama-3.1-8b', got '%s'", cfg.DefaultModel)
	}

	if cfg.DefaultTimeout != "45m" {
		t.Errorf("Expected DefaultTimeout '45m', got '%s'", cfg.DefaultTimeout)
	}

	if cfg.DefaultGPU != 0 {
		t.Errorf("Expected DefaultGPU 0, got %d", cfg.DefaultGPU)
	}

	// Verify model entry
	model, exists := cfg.Models["llama-3.1-8b"]
	if !exists {
		t.Fatal("Expected model 'llama-3.1-8b' not found in config")
	}

	if model.Path != "/models/llama-3.1-8b.gguf" {
		t.Errorf("Expected model path '/models/llama-3.1-8b.gguf', got '%s'", model.Path)
	}

	if model.ContextSize != 4096 {
		t.Errorf("Expected context size 4096, got %d", model.ContextSize)
	}

	if !model.FlashAttention {
		t.Error("Expected FlashAttention to be true")
	}

	// Verify inference config
	if cfg.Inference.MaxTokens != 2048 {
		t.Errorf("Expected MaxTokens 2048, got %d", cfg.Inference.MaxTokens)
	}

	if cfg.Inference.Temperature != 0.8 {
		t.Errorf("Expected Temperature 0.8, got %f", cfg.Inference.Temperature)
	}

	// Verify search config
	if cfg.Search.Provider != "google" {
		t.Errorf("Expected Search.Provider 'google', got '%s'", cfg.Search.Provider)
	}

	if cfg.Search.MaxResults != 10 {
		t.Errorf("Expected Search.MaxResults 10, got %d", cfg.Search.MaxResults)
	}
}

// TestLoadFrom_InvalidConfig writes YAML with an invalid type (string where
// int is expected for context_size) and verifies LoadFrom returns an error
// and a nil config.
func TestLoadFrom_InvalidConfig(t *testing.T) {
	// Create a temporary file with invalid YAML
	invalidConfig := `
default_model: "llama-3.1-8b"
default_timeout: "45m"
models:
  llama-3.1-8b:
    path: "/models/llama-3.1-8b.gguf"
    context_size: invalid_number  # This will cause YAML parsing to fail
    flash_attention: true
inference:
  context_size: 8192
  max_tokens: 2048
  temperature: 0.8
  top_p: 0.95
  top_k: 50
  timeout: 180
system_prompt: "You are a helpful AI assistant."
search:
  provider: "google"
  api_key: "test-api-key"
  max_results: 10
`

	// Create temporary file
	tmpFile, err := os.CreateTemp("", "invalid_config_*.yaml")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	// Write invalid config to temp file
	if _, err := tmpFile.WriteString(invalidConfig); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tmpFile.Close()

	// Attempt to load config from invalid file
	cfg, err := LoadFrom(tmpFile.Name())

	// Verify that loading failed
	if err == nil {
		t.Error("Expected LoadFrom to return an error for invalid YAML, but got nil")
	}

	if cfg != nil {
		t.Error("Expected config to be nil when error occurs, but got non-nil config")
	}
}

// TestLoadFrom_NonExistentFile verifies that LoadFrom returns the default
// config (with initialized Models map and "30m" default timeout) without
// error when the config file doesn't exist.
func TestLoadFrom_NonExistentFile(t *testing.T) {
	// Test loading from a non-existent file
	nonExistentPath := filepath.Join(os.TempDir(), "non_existent_config.yaml")

	cfg, err := LoadFrom(nonExistentPath)

	// Should return default config without error
	if err != nil {
		t.Errorf("Expected no error for non-existent file, got: %v", err)
	}

	if cfg == nil {
		t.Error("Expected default config to be returned for non-existent file, got nil")
	}

	// Verify it's the default config
	if cfg.Models == nil {
		t.Error("Expected Models map to be initialized in default config")
	}

	if cfg.DefaultTimeout != "30m" {
		t.Errorf("Expected default timeout '30m', got '%s'", cfg.DefaultTimeout)
	}
}
