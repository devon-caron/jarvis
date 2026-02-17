package config

import (
	"os"

	"gopkg.in/yaml.v3"

	"github.com/devon-caron/jarvis/internal"
)

// Config holds all jarvis configuration.
type Config struct {
	DefaultModel string            `yaml:"default_model"`
	Models       map[string]string `yaml:"models"`
	ModelOptions ModelOptions      `yaml:"model_options"`
	Inference    InferenceConfig   `yaml:"inference"`
	SystemPrompt string            `yaml:"system_prompt"`
	Search       SearchConfig      `yaml:"search"`
}

// ModelOptions configures how models are loaded.
type ModelOptions struct {
	GPULayers   int    `yaml:"gpu_layers"`
	TensorSplit string `yaml:"tensor_split"`
	MLock       bool   `yaml:"mlock"`
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
		Models: make(map[string]string),
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
		cfg.Models = make(map[string]string)
	}

	return cfg, nil
}

// ResolveModel resolves a model path or alias to an absolute path.
// If the input matches a configured alias, it returns the alias's path.
// Otherwise it returns the input as-is (assumed to be a path).
func (c *Config) ResolveModel(nameOrPath string) string {
	if path, ok := c.Models[nameOrPath]; ok {
		return path
	}
	return nameOrPath
}

// SearchAPIKey returns the search API key, checking the env var as fallback.
func (c *Config) SearchAPIKey() string {
	if c.Search.APIKey != "" {
		return c.Search.APIKey
	}
	return os.Getenv("BRAVE_API_KEY")
}
