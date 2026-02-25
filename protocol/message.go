package protocol

import (
	"encoding/json"
	"fmt"
)

// Request types sent from client to daemon.
const (
	ReqChat   = "chat"
	ReqLoad   = "load"
	ReqUnload = "unload"
	ReqStatus = "status"
	ReqStop   = "stop"
)

// Response types sent from daemon to client.
const (
	RespDelta  = "delta"
	RespDone   = "done"
	RespError  = "error"
	RespStatus = "status"
	RespOK     = "ok"
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
	Model        string        `json:"model,omitempty"`
	GPU          *int          `json:"gpu,omitempty"` // nil = auto-route; set to route to specific GPU
	WebSearch    bool          `json:"web_search,omitempty"`
	SystemPrompt string        `json:"system_prompt,omitempty"`
	Opts         InferenceOpts `json:"opts,omitempty"`
}

// LoadRequest holds the payload for a model load request.
type LoadRequest struct {
	ModelPath    string `json:"model_path"`
	Name         string `json:"name,omitempty"`
	GPUs         []int  `json:"gpus,omitempty"`
	NVLink       bool   `json:"nvlink,omitempty"`
	EnforceEager bool   `json:"enforce_eager,omitempty"`
	Timeout      string `json:"timeout,omitempty"`
}

// UnloadRequest holds the payload for a model unload request.
type UnloadRequest struct {
	Name string `json:"name,omitempty"`
	GPU  *int   `json:"gpu,omitempty"` // nil = not specified; unload by GPU index
}

// Response is the envelope for all daemon-to-client messages.
type Response struct {
	Type   string          `json:"type"`
	Delta  *DeltaPayload   `json:"delta,omitempty"`
	Error  *ErrorPayload   `json:"error,omitempty"`
	Status *StatusPayload  `json:"status,omitempty"`
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
	Running     bool         `json:"running"`
	ModelLoaded bool         `json:"model_loaded"`
	ModelPath   string       `json:"model_path,omitempty"`
	PID         int          `json:"pid"`
	Model       *ModelStatus `json:"model,omitempty"`
	Models      []SlotInfo   `json:"models,omitempty"`
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

func DeltaResponse(content string) *Response {
	return &Response{Type: RespDelta, Delta: &DeltaPayload{Content: content}}
}

func DoneResponse() *Response {
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
