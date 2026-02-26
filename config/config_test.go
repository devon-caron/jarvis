package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaults(t *testing.T) {
	cfg := Defaults()

	if cfg.Inference.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want 8192", cfg.Inference.ContextSize)
	}
	if cfg.Inference.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d, want 1024", cfg.Inference.MaxTokens)
	}
	if cfg.Inference.Temperature != 0.7 {
		t.Errorf("Temperature = %f, want 0.7", cfg.Inference.Temperature)
	}
	if cfg.Inference.TopP != 0.9 {
		t.Errorf("TopP = %f, want 0.9", cfg.Inference.TopP)
	}
	if cfg.Inference.TopK != 40 {
		t.Errorf("TopK = %d, want 40", cfg.Inference.TopK)
	}
	if cfg.Inference.Timeout != 120 {
		t.Errorf("Timeout = %d, want 120", cfg.Inference.Timeout)
	}
	if cfg.ModelOptions.GPULayers != -1 {
		t.Errorf("GPULayers = %d, want -1", cfg.ModelOptions.GPULayers)
	}
	if cfg.SystemPrompt != "You are a helpful AI assistant." {
		t.Errorf("SystemPrompt = %q", cfg.SystemPrompt)
	}
	if cfg.Search.Provider != "brave" {
		t.Errorf("Search.Provider = %q, want brave", cfg.Search.Provider)
	}
	if cfg.Search.MaxResults != 5 {
		t.Errorf("Search.MaxResults = %d, want 5", cfg.Search.MaxResults)
	}
	if cfg.Models == nil {
		t.Error("Models map should be initialized")
	}
	if cfg.DefaultGPU != 0 {
		t.Errorf("DefaultGPU = %d, want 0", cfg.DefaultGPU)
	}
	if cfg.DefaultTimeout != "30m" {
		t.Errorf("DefaultTimeout = %q, want 30m", cfg.DefaultTimeout)
	}
}

func TestDefaults_DefaultGPU(t *testing.T) {
	cfg := Defaults()
	if cfg.DefaultGPU != 0 {
		t.Errorf("DefaultGPU = %d, want 0", cfg.DefaultGPU)
	}
}

func TestLoadFrom_MissingFile(t *testing.T) {
	cfg, err := LoadFrom("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("LoadFrom should not error on missing file: %v", err)
	}
	// Should return defaults
	if cfg.Inference.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want default 8192", cfg.Inference.ContextSize)
	}
}

func TestLoadFrom_ValidYAML(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	yaml := `
default_model: mymodel
default_timeout: "30m"
default_gpu: 1
models:
  mymodel:
    path: /path/to/model.gguf
    context_size: 16384
  small:
    path: /path/to/small.gguf
model_options:
  gpu_layers: 40
  tensor_split: "0.6,0.4"
  mlock: true
inference:
  context_size: 4096
  max_tokens: 512
  temperature: 0.5
  top_p: 0.85
  top_k: 30
  timeout: 60
system_prompt: "Custom prompt"
search:
  provider: brave
  api_key: "test-key"
  max_results: 3
llama_server:
  binary_path: /usr/local/bin/llama-server
`
	os.WriteFile(cfgPath, []byte(yaml), 0644)

	cfg, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}

	if cfg.DefaultModel != "mymodel" {
		t.Errorf("DefaultModel = %q, want mymodel", cfg.DefaultModel)
	}
	if cfg.DefaultTimeout != "30m" {
		t.Errorf("DefaultTimeout = %q, want 30m", cfg.DefaultTimeout)
	}
	if cfg.DefaultGPU != 1 {
		t.Errorf("DefaultGPU = %d, want 1", cfg.DefaultGPU)
	}
	if cfg.Models["mymodel"].Path != "/path/to/model.gguf" {
		t.Errorf("Models[mymodel].Path = %q", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["mymodel"].ContextSize != 16384 {
		t.Errorf("Models[mymodel].ContextSize = %d, want 16384", cfg.Models["mymodel"].ContextSize)
	}
	if cfg.Models["small"].Path != "/path/to/small.gguf" {
		t.Errorf("Models[small].Path = %q", cfg.Models["small"].Path)
	}
	if cfg.Models["small"].ContextSize != 0 {
		t.Errorf("Models[small].ContextSize = %d, want 0 (unset)", cfg.Models["small"].ContextSize)
	}
	if cfg.ModelOptions.GPULayers != 40 {
		t.Errorf("GPULayers = %d, want 40", cfg.ModelOptions.GPULayers)
	}
	if cfg.ModelOptions.TensorSplit != "0.6,0.4" {
		t.Errorf("TensorSplit = %q", cfg.ModelOptions.TensorSplit)
	}
	if !cfg.ModelOptions.MLock {
		t.Error("MLock should be true")
	}
	if cfg.Inference.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096", cfg.Inference.ContextSize)
	}
	if cfg.Inference.MaxTokens != 512 {
		t.Errorf("MaxTokens = %d, want 512", cfg.Inference.MaxTokens)
	}
	if cfg.Inference.Temperature != 0.5 {
		t.Errorf("Temperature = %f, want 0.5", cfg.Inference.Temperature)
	}
	if cfg.SystemPrompt != "Custom prompt" {
		t.Errorf("SystemPrompt = %q", cfg.SystemPrompt)
	}
	if cfg.Search.APIKey != "test-key" {
		t.Errorf("Search.APIKey = %q", cfg.Search.APIKey)
	}
	if cfg.Search.MaxResults != 3 {
		t.Errorf("Search.MaxResults = %d, want 3", cfg.Search.MaxResults)
	}
	if cfg.LlamaServer.BinaryPath != "/usr/local/bin/llama-server" {
		t.Errorf("LlamaServer.BinaryPath = %q, want /usr/local/bin/llama-server", cfg.LlamaServer.BinaryPath)
	}
}

func TestLoadFrom_InvalidYAML(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	os.WriteFile(cfgPath, []byte("invalid: yaml: ["), 0644)

	_, err := LoadFrom(cfgPath)
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestLoadFrom_PartialYAML(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	os.WriteFile(cfgPath, []byte("default_model: test\n"), 0644)

	cfg, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}

	if cfg.DefaultModel != "test" {
		t.Errorf("DefaultModel = %q, want test", cfg.DefaultModel)
	}
	// Unset fields should keep defaults
	if cfg.Inference.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want default 8192", cfg.Inference.ContextSize)
	}
}

func TestResolveModel_Found(t *testing.T) {
	cfg := Defaults()
	cfg.Models["big"] = ModelEntry{Path: "/path/to/big.gguf", ContextSize: 4096}

	entry, ok := cfg.ResolveModel("big")
	if !ok {
		t.Fatal("ResolveModel(big) should return true")
	}
	if entry.Path != "/path/to/big.gguf" {
		t.Errorf("Path = %q, want /path/to/big.gguf", entry.Path)
	}
	if entry.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096", entry.ContextSize)
	}
}

func TestResolveModel_NotFound(t *testing.T) {
	cfg := Defaults()
	_, ok := cfg.ResolveModel("nonexistent")
	if ok {
		t.Error("ResolveModel(nonexistent) should return false")
	}
}

func TestSearchAPIKey_FromConfig(t *testing.T) {
	cfg := Defaults()
	cfg.Search.APIKey = "config-key"
	if got := cfg.SearchAPIKey(); got != "config-key" {
		t.Errorf("SearchAPIKey() = %q, want config-key", got)
	}
}

func TestSearchAPIKey_FromEnv(t *testing.T) {
	t.Setenv("BRAVE_API_KEY", "env-key")
	cfg := Defaults()
	if got := cfg.SearchAPIKey(); got != "env-key" {
		t.Errorf("SearchAPIKey() = %q, want env-key", got)
	}
}

func TestSearchAPIKey_ConfigOverridesEnv(t *testing.T) {
	t.Setenv("BRAVE_API_KEY", "env-key")
	cfg := Defaults()
	cfg.Search.APIKey = "config-key"
	if got := cfg.SearchAPIKey(); got != "config-key" {
		t.Errorf("SearchAPIKey() = %q, want config-key", got)
	}
}

func TestLoadFrom_NilModelsMap(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	// YAML that explicitly doesn't set models
	os.WriteFile(cfgPath, []byte("system_prompt: test\n"), 0644)

	cfg, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}
	if cfg.Models == nil {
		t.Error("Models should be initialized even when not in config")
	}
}

func TestSave(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	cfg := Defaults()
	cfg.DefaultModel = "testmodel"
	cfg.DefaultGPU = 1
	cfg.DefaultTimeout = "30m"
	cfg.Models["test"] = ModelEntry{Path: "/path/to/test.gguf", ContextSize: 8192}

	if err := cfg.Save(cfgPath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Reload and verify
	loaded, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom after save: %v", err)
	}
	if loaded.DefaultModel != "testmodel" {
		t.Errorf("DefaultModel = %q, want testmodel", loaded.DefaultModel)
	}
	if loaded.DefaultGPU != 1 {
		t.Errorf("DefaultGPU = %d, want 1", loaded.DefaultGPU)
	}
	if loaded.DefaultTimeout != "30m" {
		t.Errorf("DefaultTimeout = %q, want 30m", loaded.DefaultTimeout)
	}
	if loaded.Models["test"].Path != "/path/to/test.gguf" {
		t.Errorf("Models[test].Path = %q", loaded.Models["test"].Path)
	}
	if loaded.Models["test"].ContextSize != 8192 {
		t.Errorf("Models[test].ContextSize = %d, want 8192", loaded.Models["test"].ContextSize)
	}
}

func TestSave_CreatesDirectories(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "subdir", "config.yaml")

	cfg := Defaults()
	if err := cfg.Save(cfgPath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	if _, err := os.Stat(cfgPath); err != nil {
		t.Errorf("config file should exist: %v", err)
	}
}

func TestAddModel(t *testing.T) {
	cfg := Defaults()
	cfg.AddModel("mymodel", "/path/to/model.gguf", 8192, false)

	if cfg.Models["mymodel"].Path != "/path/to/model.gguf" {
		t.Errorf("Models[mymodel].Path = %q, want /path/to/model.gguf", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["mymodel"].ContextSize != 8192 {
		t.Errorf("Models[mymodel].ContextSize = %d, want 8192", cfg.Models["mymodel"].ContextSize)
	}

	// Update existing
	cfg.AddModel("mymodel", "/new/path.gguf", 16384, false)
	if cfg.Models["mymodel"].Path != "/new/path.gguf" {
		t.Errorf("Models[mymodel].Path = %q, want /new/path.gguf", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["mymodel"].ContextSize != 16384 {
		t.Errorf("Models[mymodel].ContextSize = %d, want 16384", cfg.Models["mymodel"].ContextSize)
	}
}

func TestAddModel_NilMap(t *testing.T) {
	cfg := &Config{}
	cfg.AddModel("test", "/path.gguf", 4096, false)

	if cfg.Models["test"].Path != "/path.gguf" {
		t.Errorf("Models[test].Path = %q, want /path.gguf", cfg.Models["test"].Path)
	}
	if cfg.Models["test"].ContextSize != 4096 {
		t.Errorf("Models[test].ContextSize = %d, want 4096", cfg.Models["test"].ContextSize)
	}
}

func TestLoad_UsesDefaultPath(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	// No config file at the XDG path — should return defaults
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Inference.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want default 8192", cfg.Inference.ContextSize)
	}
}

func TestWriteDefault(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	path, err := WriteDefault()
	if err != nil {
		t.Fatalf("WriteDefault: %v", err)
	}
	if path == "" {
		t.Error("WriteDefault should return the config path")
	}

	// File should exist
	if _, err := os.Stat(path); err != nil {
		t.Errorf("config file should exist: %v", err)
	}

	// Calling again should return ErrExist
	_, err = WriteDefault()
	if err == nil {
		t.Error("WriteDefault should error when file exists")
	}
}

func TestLoadFrom_DefaultGPU_Unset(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	os.WriteFile(cfgPath, []byte("default_model: test\n"), 0644)

	cfg, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}
	// DefaultGPU should be 0 (Go zero value) when not set
	if cfg.DefaultGPU != 0 {
		t.Errorf("DefaultGPU = %d, want 0", cfg.DefaultGPU)
	}
}

func TestSave_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	cfg := Defaults()
	cfg.DefaultTimeout = "1h"
	cfg.DefaultGPU = 2
	cfg.Models["big"] = ModelEntry{Path: "/models/big.gguf", ContextSize: 16384}
	cfg.Models["small"] = ModelEntry{Path: "/models/small.gguf"}

	if err := cfg.Save(cfgPath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}

	if loaded.DefaultTimeout != "1h" {
		t.Errorf("DefaultTimeout = %q, want 1h", loaded.DefaultTimeout)
	}
	if loaded.DefaultGPU != 2 {
		t.Errorf("DefaultGPU = %d, want 2", loaded.DefaultGPU)
	}
	if loaded.Models["big"].Path != "/models/big.gguf" {
		t.Errorf("Models[big].Path = %q", loaded.Models["big"].Path)
	}
	if loaded.Models["big"].ContextSize != 16384 {
		t.Errorf("Models[big].ContextSize = %d, want 16384", loaded.Models["big"].ContextSize)
	}
	if loaded.Models["small"].Path != "/models/small.gguf" {
		t.Errorf("Models[small].Path = %q", loaded.Models["small"].Path)
	}
}
