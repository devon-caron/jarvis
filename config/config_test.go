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
	if len(cfg.Search.ZimPaths) != 0 {
		t.Errorf("Search.ZimPaths = %v, want empty", cfg.Search.ZimPaths)
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
  zim_paths:
    - /path/to/wiki.zim
  max_results: 3
llama_server:
  ik_binary_path: /opt/ik/llama-server
  vanilla_binary_path: /opt/vanilla/llama-server
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
	if len(cfg.Search.ZimPaths) != 1 || cfg.Search.ZimPaths[0] != "/path/to/wiki.zim" {
		t.Errorf("Search.ZimPaths = %v, want [/path/to/wiki.zim]", cfg.Search.ZimPaths)
	}
	if cfg.Search.MaxResults != 3 {
		t.Errorf("Search.MaxResults = %d, want 3", cfg.Search.MaxResults)
	}
	if cfg.LlamaServer.IKBinaryPath != "/opt/ik/llama-server" {
		t.Errorf("LlamaServer.IKBinaryPath = %q, want /opt/ik/llama-server", cfg.LlamaServer.IKBinaryPath)
	}
	if cfg.LlamaServer.VanillaBinaryPath != "/opt/vanilla/llama-server" {
		t.Errorf("LlamaServer.VanillaBinaryPath = %q, want /opt/vanilla/llama-server", cfg.LlamaServer.VanillaBinaryPath)
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

func TestZimPaths_DefaultEmpty(t *testing.T) {
	cfg := Defaults()
	if len(cfg.Search.ZimPaths) != 0 {
		t.Errorf("default ZimPaths should be empty, got %v", cfg.Search.ZimPaths)
	}
}

func TestZimPaths_SetAndRetrieve(t *testing.T) {
	cfg := Defaults()
	cfg.Search.ZimPaths = []string{"/path/to/wiki.zim", "/path/to/other.zim"}
	if len(cfg.Search.ZimPaths) != 2 {
		t.Errorf("ZimPaths length = %d, want 2", len(cfg.Search.ZimPaths))
	}
	if cfg.Search.ZimPaths[0] != "/path/to/wiki.zim" {
		t.Errorf("ZimPaths[0] = %q", cfg.Search.ZimPaths[0])
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
	cfg.AddModel("mymodel", ModelEntry{Path: "/path/to/model.gguf", ContextSize: 8192})

	if cfg.Models["mymodel"].Path != "/path/to/model.gguf" {
		t.Errorf("Models[mymodel].Path = %q, want /path/to/model.gguf", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["mymodel"].ContextSize != 8192 {
		t.Errorf("Models[mymodel].ContextSize = %d, want 8192", cfg.Models["mymodel"].ContextSize)
	}

	// Update existing
	cfg.AddModel("mymodel", ModelEntry{Path: "/new/path.gguf", ContextSize: 16384})
	if cfg.Models["mymodel"].Path != "/new/path.gguf" {
		t.Errorf("Models[mymodel].Path = %q, want /new/path.gguf", cfg.Models["mymodel"].Path)
	}
	if cfg.Models["mymodel"].ContextSize != 16384 {
		t.Errorf("Models[mymodel].ContextSize = %d, want 16384", cfg.Models["mymodel"].ContextSize)
	}
}

func TestAddModel_NilMap(t *testing.T) {
	cfg := &Config{}
	cfg.AddModel("test", ModelEntry{Path: "/path.gguf", ContextSize: 4096})

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

func TestResolveBinary(t *testing.T) {
	tests := []struct {
		name      string
		cfg       LlamaServerConfig
		splitMode string
		want      string
	}{
		{
			name:      "all empty returns default",
			cfg:       LlamaServerConfig{},
			splitMode: "",
			want:      "llama-server",
		},
		{
			name:      "no split mode uses vanilla_binary_path",
			cfg:       LlamaServerConfig{VanillaBinaryPath: "/vanilla"},
			splitMode: "",
			want:      "/vanilla",
		},
		{
			name:      "graph uses ik_binary_path",
			cfg:       LlamaServerConfig{IKBinaryPath: "/ik"},
			splitMode: "graph",
			want:      "/ik",
		},
		{
			name:      "graph falls back to default when ik not set",
			cfg:       LlamaServerConfig{},
			splitMode: "graph",
			want:      "llama-server",
		},
		{
			name:      "layer uses vanilla_binary_path",
			cfg:       LlamaServerConfig{VanillaBinaryPath: "/vanilla"},
			splitMode: "layer",
			want:      "/vanilla",
		},
		{
			name:      "row uses vanilla_binary_path",
			cfg:       LlamaServerConfig{VanillaBinaryPath: "/vanilla"},
			splitMode: "row",
			want:      "/vanilla",
		},
		{
			name:      "layer falls back to default when vanilla not set",
			cfg:       LlamaServerConfig{},
			splitMode: "layer",
			want:      "llama-server",
		},
		{
			name:      "both set, graph picks ik",
			cfg:       LlamaServerConfig{IKBinaryPath: "/ik", VanillaBinaryPath: "/vanilla"},
			splitMode: "graph",
			want:      "/ik",
		},
		{
			name:      "both set, row picks vanilla",
			cfg:       LlamaServerConfig{IKBinaryPath: "/ik", VanillaBinaryPath: "/vanilla"},
			splitMode: "row",
			want:      "/vanilla",
		},
		{
			name:      "both set, no split mode picks vanilla",
			cfg:       LlamaServerConfig{IKBinaryPath: "/ik", VanillaBinaryPath: "/vanilla"},
			splitMode: "",
			want:      "/vanilla",
		},
		{
			name:      "only ik set, no split mode falls back to default",
			cfg:       LlamaServerConfig{IKBinaryPath: "/ik"},
			splitMode: "",
			want:      "llama-server",
		},
		{
			name:      "only vanilla set, graph falls back to default",
			cfg:       LlamaServerConfig{VanillaBinaryPath: "/vanilla"},
			splitMode: "graph",
			want:      "llama-server",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.cfg.ResolveBinary(tt.splitMode)
			if got != tt.want {
				t.Errorf("ResolveBinary(%q) = %q, want %q", tt.splitMode, got, tt.want)
			}
		})
	}
}
