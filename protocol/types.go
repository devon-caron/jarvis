package protocol

import "time"

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// InferenceOpts holds inference parameters for chat requests.
type InferenceOpts struct {
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
}

// ModelStatus describes the current state of a loaded model.
type ModelStatus struct {
	ModelPath string    `json:"model_path,omitempty"`
	GPULayers int       `json:"gpu_layers,omitempty"`
	GPUs      []GPUInfo `json:"gpus,omitempty"`
}

// GPUInfo describes a single GPU device.
type GPUInfo struct {
	DeviceID      int    `json:"device_id"`
	DeviceName    string `json:"device_name"`
	FreeMemoryMB  int    `json:"free_memory_mb"`
	TotalMemoryMB int    `json:"total_memory_mb"`
}

// SlotInfo describes a loaded model slot for multi-model status reporting.
type SlotInfo struct {
	Name      string    `json:"name"`
	ModelPath string    `json:"model_path"`
	GPUs      []int     `json:"gpus"`
	Timeout   string    `json:"timeout,omitempty"`
	LastUsed  time.Time `json:"last_used"`
	GPUInfo   []GPUInfo `json:"gpu_info,omitempty"`
}
