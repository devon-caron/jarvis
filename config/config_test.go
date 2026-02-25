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
	if cfg.ModelOptions.GPUMemoryUtilization != 0.9 {
		t.Errorf("GPUMemoryUtilization = %f, want 0.9", cfg.ModelOptions.GPUMemoryUtilization)
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
	if cfg.VLLM.BinaryPath != "vllm" {
		t.Errorf("VLLM.BinaryPath = %q, want vllm", cfg.VLLM.BinaryPath)
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

	yamlData := `
default_model: mymodel
default_timeout: "30m"
default_gpu: 1
models:
  mymodel: /path/to/model.gguf
  small: /path/to/small.gguf
model_options:
  gpu_memory_utilization: 0.8
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
vllm:
  binary_path: /usr/bin/vllm
`
	os.WriteFile(cfgPath, []byte(yamlData), 0644)

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
	// Bare string format (backward compat): mymodel: /path/to/model.gguf
	if cfg.Models["mymodel"].Path != "/path/to/model.gguf" {
		t.Errorf("Models[mymodel].Path = %q", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["small"].Path != "/path/to/small.gguf" {
		t.Errorf("Models[small].Path = %q", cfg.Models["small"].Path)
	}
	if cfg.ModelOptions.GPUMemoryUtilization != 0.8 {
		t.Errorf("GPUMemoryUtilization = %f, want 0.8", cfg.ModelOptions.GPUMemoryUtilization)
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
	if cfg.VLLM.BinaryPath != "/usr/bin/vllm" {
		t.Errorf("VLLM.BinaryPath = %q, want /usr/bin/vllm", cfg.VLLM.BinaryPath)
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

func TestResolveModel_Alias(t *testing.T) {
	cfg := Defaults()
	cfg.Models["big"] = ModelEntry{Path: "/path/to/big.gguf"}

	got := cfg.ResolveModel("big")
	if got != "/path/to/big.gguf" {
		t.Errorf("ResolveModel(big) = %q, want /path/to/big.gguf", got)
	}
}

func TestResolveModel_Path(t *testing.T) {
	cfg := Defaults()
	got := cfg.ResolveModel("/direct/path.gguf")
	if got != "/direct/path.gguf" {
		t.Errorf("ResolveModel = %q, want /direct/path.gguf", got)
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
	cfg.Models["test"] = ModelEntry{Path: "/path/to/test.gguf"}

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
	cfg.AddModel("mymodel", "/path/to/model.gguf", false)

	if cfg.Models["mymodel"].Path != "/path/to/model.gguf" {
		t.Errorf("Models[mymodel].Path = %q, want /path/to/model.gguf", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["mymodel"].NVLink {
		t.Error("NVLink should be false")
	}

	// Update existing with nvlink
	cfg.AddModel("mymodel", "/new/path.gguf", true)
	if cfg.Models["mymodel"].Path != "/new/path.gguf" {
		t.Errorf("Models[mymodel].Path = %q, want /new/path.gguf", cfg.Models["mymodel"].Path)
	}
	if !cfg.Models["mymodel"].NVLink {
		t.Error("NVLink should be true")
	}
}

func TestAddModel_NilMap(t *testing.T) {
	cfg := &Config{}
	cfg.AddModel("test", "/path.gguf", false)

	if cfg.Models["test"].Path != "/path.gguf" {
		t.Errorf("Models[test].Path = %q, want /path.gguf", cfg.Models["test"].Path)
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
	cfg.Models["big"] = ModelEntry{Path: "/models/big.gguf", NVLink: true}
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
	if !loaded.Models["big"].NVLink {
		t.Error("Models[big].NVLink should be true")
	}
	if loaded.Models["small"].Path != "/models/small.gguf" {
		t.Errorf("Models[small].Path = %q", loaded.Models["small"].Path)
	}
}

func TestLoadFrom_BackwardCompatBareString(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	// Old format: models as bare strings
	yamlData := `
models:
  mymodel: /path/to/model.gguf
  other: /path/to/other.gguf
`
	os.WriteFile(cfgPath, []byte(yamlData), 0644)

	cfg, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}

	if cfg.Models["mymodel"].Path != "/path/to/model.gguf" {
		t.Errorf("Models[mymodel].Path = %q, want /path/to/model.gguf", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["mymodel"].NVLink {
		t.Error("NVLink should default to false for bare string format")
	}
	if cfg.Models["other"].Path != "/path/to/other.gguf" {
		t.Errorf("Models[other].Path = %q", cfg.Models["other"].Path)
	}
}

func TestLoadFrom_NewModelEntryFormat(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	yamlData := `
models:
  mymodel:
    path: /path/to/model.gguf
    nvlink: true
  other:
    path: /path/to/other.gguf
`
	os.WriteFile(cfgPath, []byte(yamlData), 0644)

	cfg, err := LoadFrom(cfgPath)
	if err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}

	if cfg.Models["mymodel"].Path != "/path/to/model.gguf" {
		t.Errorf("Models[mymodel].Path = %q", cfg.Models["mymodel"].Path)
	}
	if !cfg.Models["mymodel"].NVLink {
		t.Error("Models[mymodel].NVLink should be true")
	}
	if cfg.Models["other"].Path != "/path/to/other.gguf" {
		t.Errorf("Models[other].Path = %q", cfg.Models["other"].Path)
	}
	if cfg.Models["other"].NVLink {
		t.Error("Models[other].NVLink should be false")
	}
}

func TestGetModelEntry(t *testing.T) {
	cfg := Defaults()
	cfg.Models["test"] = ModelEntry{Path: "/path.gguf", NVLink: true}

	entry, ok := cfg.GetModelEntry("test")
	if !ok {
		t.Fatal("expected model entry to exist")
	}
	if entry.Path != "/path.gguf" {
		t.Errorf("Path = %q", entry.Path)
	}
	if !entry.NVLink {
		t.Error("NVLink should be true")
	}

	_, ok = cfg.GetModelEntry("nonexistent")
	if ok {
		t.Error("expected nonexistent model to return false")
	}
}
