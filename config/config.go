package config

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	"github.com/devon-caron/jarvis/internal"
)

// ModelEntry stores a registered model's path and settings.
type ModelEntry struct {
	Path           string `yaml:"path"`
	ContextSize    int    `yaml:"context_size,omitempty"`    // 0 = use default (8192)
	SplitMode      string `yaml:"split_mode,omitempty"`      // multi-GPU split mode: layer, row, or graph
	FlashAttention bool   `yaml:"flash_attention,omitempty"` // enable flash attention
	BatchSize      int    `yaml:"batch_size,omitempty"`      // micro-batch size (0 = server default)
	TensorSplit    string `yaml:"tensor_split,omitempty"`    // GPU weight distribution (e.g. "1,1")
}

// LlamaServerConfig configures the llama-server binary.
type LlamaServerConfig struct {
	BinaryPath string `yaml:"binary_path"`
}

// Config holds all jarvis configuration.
type Config struct {
	DefaultModel   string                  `yaml:"default_model"`
	DefaultTimeout string                  `yaml:"default_timeout"`
	DefaultGPU     int                     `yaml:"default_gpu"`
	Models         map[string]ModelEntry   `yaml:"models"`
	ModelOptions   ModelOptions            `yaml:"model_options"`
	Inference      InferenceConfig         `yaml:"inference"`
	SystemPrompt   string                  `yaml:"system_prompt"`
	Search         SearchConfig            `yaml:"search"`
	LlamaServer    LlamaServerConfig       `yaml:"llama_server"`
}

// ModelOptions configures how models are loaded.
type ModelOptions struct {
	GPULayers      int    `yaml:"gpu_layers"`
	TensorSplit    string `yaml:"tensor_split"`
	MLock          bool   `yaml:"mlock"`
	FlashAttention bool   `yaml:"flash_attention"`
	BatchSize      int    `yaml:"batch_size"` // micro-batch size (0 = server default)
}

// InferenceConfig holds default inference parameters.
type InferenceConfig struct {
	ContextSize int     `yaml:"context_size"`
	MaxTokens   int     `yaml:"max_tokens"`
	Temperature float64 `yaml:"temperature"`
	TopP        float64 `yaml:"top_p"`
	TopK        int     `yaml:"top_k"`
	Timeout     int     `yaml:"timeout"`
}

// SearchConfig configures web search.
type SearchConfig struct {
	Provider   string `yaml:"provider"`
	APIKey     string `yaml:"api_key"`
	MaxResults int    `yaml:"max_results"`
}

// Defaults returns a Config with sensible default values.
func Defaults() *Config {
	return &Config{
		Models:         make(map[string]ModelEntry),
		DefaultTimeout: "30m",
		ModelOptions: ModelOptions{
			GPULayers: -1,
		},
		Inference: InferenceConfig{
			ContextSize: 8192,
			MaxTokens:   1024,
			Temperature: 0.7,
			TopP:        0.9,
			TopK:        40,
			Timeout:     120,
		},
		SystemPrompt: "You are a helpful AI assistant.",
		Search: SearchConfig{
			Provider:   "brave",
			MaxResults: 5,
		},
		LlamaServer: LlamaServerConfig{
			BinaryPath: "llama-server",
		},
	}
}

// Load reads the config file from the default path.
// If the file doesn't exist, returns defaults.
func Load() (*Config, error) {
	return LoadFrom(internal.ConfigPath())
}

// LoadFrom reads a config file from the given path.
// If the file doesn't exist, returns defaults.
func LoadFrom(path string) (*Config, error) {
	cfg := Defaults()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	// Ensure maps are initialized
	if cfg.Models == nil {
		cfg.Models = make(map[string]ModelEntry)
	}

	return cfg, nil
}

// WriteDefault writes a default config file to the standard config path.
// Returns an error if the file already exists.
func WriteDefault() (string, error) {
	path := internal.ConfigPath()
	if _, err := os.Stat(path); err == nil {
		return path, os.ErrExist
	}

	if err := os.MkdirAll(internal.ConfigDir(), 0755); err != nil {
		return "", err
	}

	cfg := Defaults()
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return "", err
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return "", err
	}
	return path, nil
}

// ResolveModel resolves a model name to its registry entry.
// Returns the entry and true if found, or a zero entry and false if not.
func (c *Config) ResolveModel(name string) (ModelEntry, bool) {
	entry, ok := c.Models[name]
	return entry, ok
}

// SearchAPIKey returns the search API key, checking the env var as fallback.
func (c *Config) SearchAPIKey() string {
	if c.Search.APIKey != "" {
		return c.Search.APIKey
	}
	return os.Getenv("BRAVE_API_KEY")
}

// Save writes the config to the given path as YAML.
func (c *Config) Save(path string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// AddModel registers a named model entry.
func (c *Config) AddModel(name string, entry ModelEntry) {
	if c.Models == nil {
		c.Models = make(map[string]ModelEntry)
	}
	c.Models[name] = entry
}

// RemoveModel deletes a named model alias. Returns false if the name was not found.
func (c *Config) RemoveModel(name string) bool {
	if _, ok := c.Models[name]; !ok {
		return false
	}
	delete(c.Models, name)
	return true
}
