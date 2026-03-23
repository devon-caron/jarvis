package protocol

import (
	"encoding/json"
	"fmt"
)

// Request is the envelope for all client-to-daemon messages.
type Request struct {
	Type   string         `json:"type"`
	Chat   *ChatRequest   `json:"chat,omitempty"`
	Load   *LoadRequest   `json:"load,omitempty"`
	Unload *UnloadRequest `json:"unload,omitempty"`
}

// ChatRequest holds the payload for a chat request.
type ChatRequest struct {
	Messages     []ChatMessage `json:"messages"`
	WebSearch    bool          `json:"web_search"`
	SystemPrompt string        `json:"system_prompt"`
	Opts         InferenceOpts `json:"opts"`
	ShellPID     int           `json:"shell_pid"`
	ClearContext bool          `json:"clear_context"`
}

// LoadRequest holds the payload for a model load request.
type LoadRequest struct {
	ModelPath      string `json:"model_path"`
	Name           string `json:"name,omitempty"`
	GPUs           []int  `json:"gpus,omitempty"`
	GPULayers      int    `json:"gpu_layers"`
	Timeout        string `json:"timeout,omitempty"`
	ContextSize    int    `json:"context_size,omitempty"`
	SplitMode      string `json:"split_mode,omitempty"`
	Parallel       int    `json:"parallel,omitempty"`
	FlashAttention bool   `json:"flash_attention,omitempty"`
	BatchSize      int    `json:"batch_size,omitempty"`
	TensorSplit    string `json:"tensor_split,omitempty"`
}

// UnloadRequest holds the payload for a model unload request.
type UnloadRequest struct {
	Name string `json:"name,omitempty"`
	GPU  *int   `json:"gpu,omitempty"` // nil = not specified; unload by GPU index
}

// Response is the envelope for all daemon-to-client messages.
type Response struct {
	Type   string         `json:"type"`
	Delta  *DeltaPayload  `json:"delta,omitempty"`
	Error  *ErrorPayload  `json:"error,omitempty"`
	Status *StatusPayload `json:"status,omitempty"`
}

// DeltaPayload carries a single token/chunk of streamed text.
type DeltaPayload struct {
	Content string `json:"content"`
}

// ErrorPayload carries an error message.
type ErrorPayload struct {
	Message string `json:"message"`
}

// StatusPayload carries daemon and model status info.
type StatusPayload struct {
	Running     bool       `json:"running"`
	ModelLoaded bool       `json:"model_loaded"`
	ModelPath   string     `json:"model_path,omitempty"`
	PID         int        `json:"pid"`
	Model       *ModelInfo `json:"model,omitempty"`
}

// MarshalRequest serializes a request to JSON bytes (no trailing newline).
func MarshalRequest(r *Request) ([]byte, error) {
	return json.Marshal(r)
}

// UnmarshalRequest deserializes a request from JSON bytes.
func UnmarshalRequest(data []byte) (*Request, error) {
	var r Request
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}
	if r.Type == "" {
		return nil, fmt.Errorf("request missing type field")
	}
	return &r, nil
}

// MarshalResponse serializes a response to JSON bytes (no trailing newline).
func MarshalResponse(r *Response) ([]byte, error) {
	return json.Marshal(r)
}

// UnmarshalResponse deserializes a response from JSON bytes.
func UnmarshalResponse(data []byte) (*Response, error) {
	var r Response
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("invalid response: %w", err)
	}
	if r.Type == "" {
		return nil, fmt.Errorf("response missing type field")
	}
	return &r, nil
}

// Helper constructors for common responses.

func DeltaTokenResponse(content string) *Response {
	return &Response{Type: RespDelta, Delta: &DeltaPayload{Content: content}}
}

func EndTokenResponse() *Response {
	return &Response{Type: RespDone}
}

func ErrorResponse(msg string) *Response {
	return &Response{Type: RespError, Error: &ErrorPayload{Message: msg}}
}

func OKResponse() *Response {
	return &Response{Type: RespOK}
}

func StatusResponse(payload *StatusPayload) *Response {
	return &Response{Type: RespStatus, Status: payload}
}
