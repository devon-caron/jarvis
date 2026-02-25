package protocol

import (
	"encoding/json"
	"testing"
	"time"
)

func TestMarshalUnmarshalRequest_Chat(t *testing.T) {
	req := &Request{
		Type: ReqChat,
		Chat: &ChatRequest{
			Messages:  []ChatMessage{{Role: "user", Content: "hello"}},
			WebSearch: true,
		},
	}

	data, err := MarshalRequest(req)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	got, err := UnmarshalRequest(data)
	if err != nil {
		t.Fatalf("UnmarshalRequest: %v", err)
	}

	if got.Type != ReqChat {
		t.Errorf("Type = %q, want %q", got.Type, ReqChat)
	}
	if got.Chat == nil {
		t.Fatal("Chat payload is nil")
	}
	if len(got.Chat.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(got.Chat.Messages))
	}
	if got.Chat.Messages[0].Content != "hello" {
		t.Errorf("Content = %q, want %q", got.Chat.Messages[0].Content, "hello")
	}
	if !got.Chat.WebSearch {
		t.Error("WebSearch should be true")
	}
}

func TestMarshalUnmarshalRequest_ChatWithModel(t *testing.T) {
	req := &Request{
		Type: ReqChat,
		Chat: &ChatRequest{
			Messages: []ChatMessage{{Role: "user", Content: "hello"}},
			Model:    "llama70b",
		},
	}

	data, err := MarshalRequest(req)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	got, err := UnmarshalRequest(data)
	if err != nil {
		t.Fatalf("UnmarshalRequest: %v", err)
	}

	if got.Chat.Model != "llama70b" {
		t.Errorf("Model = %q, want llama70b", got.Chat.Model)
	}
}

func TestMarshalUnmarshalRequest_Load(t *testing.T) {
	req := &Request{
		Type: ReqLoad,
		Load: &LoadRequest{
			ModelPath: "/path/to/model.gguf",
			Name:      "mymodel",
			GPUs:      []int{0, 1},
			NVLink:    true,
			Timeout:   "30m",
		},
	}

	data, err := MarshalRequest(req)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	got, err := UnmarshalRequest(data)
	if err != nil {
		t.Fatalf("UnmarshalRequest: %v", err)
	}

	if got.Type != ReqLoad {
		t.Errorf("Type = %q, want %q", got.Type, ReqLoad)
	}
	if got.Load == nil {
		t.Fatal("Load payload is nil")
	}
	if got.Load.ModelPath != "/path/to/model.gguf" {
		t.Errorf("ModelPath = %q", got.Load.ModelPath)
	}
	if got.Load.Name != "mymodel" {
		t.Errorf("Name = %q, want mymodel", got.Load.Name)
	}
	if len(got.Load.GPUs) != 2 || got.Load.GPUs[0] != 0 || got.Load.GPUs[1] != 1 {
		t.Errorf("GPUs = %v, want [0, 1]", got.Load.GPUs)
	}
	if !got.Load.NVLink {
		t.Error("NVLink should be true")
	}
	if got.Load.Timeout != "30m" {
		t.Errorf("Timeout = %q, want 30m", got.Load.Timeout)
	}
}

func TestMarshalUnmarshalRequest_Unload(t *testing.T) {
	req := &Request{
		Type:   ReqUnload,
		Unload: &UnloadRequest{Name: "mymodel"},
	}

	data, err := MarshalRequest(req)
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}

	got, err := UnmarshalRequest(data)
	if err != nil {
		t.Fatalf("UnmarshalRequest: %v", err)
	}

	if got.Type != ReqUnload {
		t.Errorf("Type = %q, want %q", got.Type, ReqUnload)
	}
	if got.Unload == nil {
		t.Fatal("Unload payload is nil")
	}
	if got.Unload.Name != "mymodel" {
		t.Errorf("Name = %q, want mymodel", got.Unload.Name)
	}
}

func TestMarshalUnmarshalRequest_Simple(t *testing.T) {
	for _, typ := range []string{ReqStatus, ReqStop} {
		req := &Request{Type: typ}
		data, err := MarshalRequest(req)
		if err != nil {
			t.Fatalf("MarshalRequest(%s): %v", typ, err)
		}
		got, err := UnmarshalRequest(data)
		if err != nil {
			t.Fatalf("UnmarshalRequest(%s): %v", typ, err)
		}
		if got.Type != typ {
			t.Errorf("Type = %q, want %q", got.Type, typ)
		}
	}
}

func TestUnmarshalRequest_InvalidJSON(t *testing.T) {
	_, err := UnmarshalRequest([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestUnmarshalRequest_MissingType(t *testing.T) {
	_, err := UnmarshalRequest([]byte(`{"chat":{}}`))
	if err == nil {
		t.Error("expected error for missing type")
	}
}

func TestMarshalUnmarshalResponse_Delta(t *testing.T) {
	resp := DeltaResponse("Hello")
	data, err := MarshalResponse(resp)
	if err != nil {
		t.Fatalf("MarshalResponse: %v", err)
	}

	got, err := UnmarshalResponse(data)
	if err != nil {
		t.Fatalf("UnmarshalResponse: %v", err)
	}

	if got.Type != RespDelta {
		t.Errorf("Type = %q, want %q", got.Type, RespDelta)
	}
	if got.Delta == nil || got.Delta.Content != "Hello" {
		t.Errorf("Delta.Content = %v, want Hello", got.Delta)
	}
}

func TestMarshalUnmarshalResponse_Done(t *testing.T) {
	resp := DoneResponse()
	data, err := MarshalResponse(resp)
	if err != nil {
		t.Fatalf("MarshalResponse: %v", err)
	}

	got, err := UnmarshalResponse(data)
	if err != nil {
		t.Fatalf("UnmarshalResponse: %v", err)
	}

	if got.Type != RespDone {
		t.Errorf("Type = %q, want %q", got.Type, RespDone)
	}
}

func TestMarshalUnmarshalResponse_Error(t *testing.T) {
	resp := ErrorResponse("something failed")
	data, err := MarshalResponse(resp)
	if err != nil {
		t.Fatalf("MarshalResponse: %v", err)
	}

	got, err := UnmarshalResponse(data)
	if err != nil {
		t.Fatalf("UnmarshalResponse: %v", err)
	}

	if got.Type != RespError {
		t.Errorf("Type = %q, want %q", got.Type, RespError)
	}
	if got.Error == nil || got.Error.Message != "something failed" {
		t.Errorf("Error = %v, want 'something failed'", got.Error)
	}
}

func TestMarshalUnmarshalResponse_OK(t *testing.T) {
	resp := OKResponse()
	data, _ := MarshalResponse(resp)
	got, _ := UnmarshalResponse(data)
	if got.Type != RespOK {
		t.Errorf("Type = %q, want %q", got.Type, RespOK)
	}
}

func TestMarshalUnmarshalResponse_Status(t *testing.T) {
	now := time.Now().Truncate(time.Second)
	resp := StatusResponse(&StatusPayload{
		Running:     true,
		ModelLoaded: true,
		ModelPath:   "/model.gguf",
		PID:         12345,
		Model: &ModelStatus{
			ModelPath: "/model.gguf",
			GPULayers: 80,
			GPUs: []GPUInfo{
				{DeviceID: 0, DeviceName: "RTX 4090", FreeMemoryMB: 20000, TotalMemoryMB: 24000},
			},
		},
		Models: []SlotInfo{
			{
				Name:      "test",
				ModelPath: "/model.gguf",
				GPUs:      []int{0},
				Timeout:   "30m0s",
				LastUsed:  now,
			},
		},
	})

	data, err := MarshalResponse(resp)
	if err != nil {
		t.Fatalf("MarshalResponse: %v", err)
	}

	got, err := UnmarshalResponse(data)
	if err != nil {
		t.Fatalf("UnmarshalResponse: %v", err)
	}

	if got.Type != RespStatus {
		t.Errorf("Type = %q, want %q", got.Type, RespStatus)
	}
	if got.Status == nil {
		t.Fatal("Status is nil")
	}
	if !got.Status.Running {
		t.Error("Running should be true")
	}
	if !got.Status.ModelLoaded {
		t.Error("ModelLoaded should be true")
	}
	if got.Status.PID != 12345 {
		t.Errorf("PID = %d, want 12345", got.Status.PID)
	}
	if got.Status.Model == nil {
		t.Fatal("Model is nil")
	}
	if got.Status.Model.GPULayers != 80 {
		t.Errorf("GPULayers = %d, want 80", got.Status.Model.GPULayers)
	}
	if len(got.Status.Model.GPUs) != 1 {
		t.Fatalf("GPUs len = %d, want 1", len(got.Status.Model.GPUs))
	}
	if len(got.Status.Models) != 1 {
		t.Fatalf("Models len = %d, want 1", len(got.Status.Models))
	}
	if got.Status.Models[0].Name != "test" {
		t.Errorf("Models[0].Name = %q, want test", got.Status.Models[0].Name)
	}
	if got.Status.Models[0].Timeout != "30m0s" {
		t.Errorf("Models[0].Timeout = %q, want 30m0s", got.Status.Models[0].Timeout)
	}
}

func TestUnmarshalResponse_InvalidJSON(t *testing.T) {
	_, err := UnmarshalResponse([]byte("{bad"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestUnmarshalResponse_MissingType(t *testing.T) {
	_, err := UnmarshalResponse([]byte(`{"delta":{"content":"hi"}}`))
	if err == nil {
		t.Error("expected error for missing type")
	}
}

func TestRequestJSON_MatchesProtocol(t *testing.T) {
	// Verify the wire format matches the protocol spec
	req := &Request{
		Type: ReqChat,
		Chat: &ChatRequest{
			Messages: []ChatMessage{{Role: "user", Content: "hello"}},
		},
	}
	data, _ := json.Marshal(req)

	var raw map[string]interface{}
	json.Unmarshal(data, &raw)

	if raw["type"] != "chat" {
		t.Errorf("wire format type = %v, want 'chat'", raw["type"])
	}
	if raw["chat"] == nil {
		t.Error("wire format missing 'chat' field")
	}
}

func TestResponseHelpers(t *testing.T) {
	tests := []struct {
		name     string
		resp     *Response
		wantType string
	}{
		{"DeltaResponse", DeltaResponse("tok"), RespDelta},
		{"DoneResponse", DoneResponse(), RespDone},
		{"ErrorResponse", ErrorResponse("err"), RespError},
		{"OKResponse", OKResponse(), RespOK},
		{"StatusResponse", StatusResponse(&StatusPayload{Running: true}), RespStatus},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.resp.Type != tt.wantType {
				t.Errorf("Type = %q, want %q", tt.resp.Type, tt.wantType)
			}
		})
	}
}
