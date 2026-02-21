package daemon

import (
	"bufio"
	"encoding/json"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

func setupTestServer(t *testing.T) (*Server, string) {
	t.Helper()
	dir := t.TempDir()
	sockPath := filepath.Join(dir, "test.sock")

	backend := &mockBackend{}
	cfg := config.Defaults()
	factory := func(c *config.Config) ModelBackend { return backend }
	registry := NewModelRegistry(cfg, factory)
	registry.Load("test", "/model.gguf", []int{0}, 0)

	stopCh := make(chan struct{}, 1)
	handler := NewHandler(registry, cfg, nil, stopCh)
	server := NewServer(sockPath, handler)

	if err := server.Listen(); err != nil {
		t.Fatalf("Listen: %v", err)
	}

	go server.Serve()

	// Give server a moment to start
	time.Sleep(10 * time.Millisecond)

	return server, sockPath
}

func TestServer_Status(t *testing.T) {
	server, sockPath := setupTestServer(t)
	defer server.Close()

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()

	// Send status request
	req := &protocol.Request{Type: protocol.ReqStatus}
	data, _ := json.Marshal(req)
	data = append(data, '\n')
	conn.Write(data)

	// Read response
	scanner := bufio.NewScanner(conn)
	if !scanner.Scan() {
		t.Fatal("no response received")
	}

	resp, err := protocol.UnmarshalResponse(scanner.Bytes())
	if err != nil {
		t.Fatalf("UnmarshalResponse: %v", err)
	}

	if resp.Type != protocol.RespStatus {
		t.Errorf("response type = %q, want status", resp.Type)
	}
	if !resp.Status.Running {
		t.Error("Running should be true")
	}
	if !resp.Status.ModelLoaded {
		t.Error("ModelLoaded should be true")
	}
}

func TestServer_Chat(t *testing.T) {
	server, sockPath := setupTestServer(t)
	defer server.Close()

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()

	req := &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hi"}},
		},
	}
	data, _ := json.Marshal(req)
	data = append(data, '\n')
	conn.Write(data)

	scanner := bufio.NewScanner(conn)
	var responses []*protocol.Response
	for scanner.Scan() {
		resp, err := protocol.UnmarshalResponse(scanner.Bytes())
		if err != nil {
			t.Fatalf("UnmarshalResponse: %v", err)
		}
		responses = append(responses, resp)
		if resp.Type == protocol.RespDone || resp.Type == protocol.RespError {
			break
		}
	}

	if len(responses) < 2 {
		t.Fatalf("expected at least 2 responses, got %d", len(responses))
	}

	lastResp := responses[len(responses)-1]
	if lastResp.Type != protocol.RespDone {
		t.Errorf("last response type = %q, want done", lastResp.Type)
	}
}

func TestServer_InvalidRequest(t *testing.T) {
	server, sockPath := setupTestServer(t)
	defer server.Close()

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()

	conn.Write([]byte("not json\n"))

	scanner := bufio.NewScanner(conn)
	if !scanner.Scan() {
		t.Fatal("no response")
	}

	resp, err := protocol.UnmarshalResponse(scanner.Bytes())
	if err != nil {
		t.Fatalf("UnmarshalResponse: %v", err)
	}

	if resp.Type != protocol.RespError {
		t.Errorf("response type = %q, want error", resp.Type)
	}
}

func TestServer_Close(t *testing.T) {
	server, _ := setupTestServer(t)
	if err := server.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestServer_MultipleConnections(t *testing.T) {
	server, sockPath := setupTestServer(t)
	defer server.Close()

	for i := 0; i < 5; i++ {
		conn, err := net.Dial("unix", sockPath)
		if err != nil {
			t.Fatalf("Dial %d: %v", i, err)
		}

		req := &protocol.Request{Type: protocol.ReqStatus}
		data, _ := json.Marshal(req)
		data = append(data, '\n')
		conn.Write(data)

		scanner := bufio.NewScanner(conn)
		if !scanner.Scan() {
			t.Fatalf("no response on connection %d", i)
		}

		resp, _ := protocol.UnmarshalResponse(scanner.Bytes())
		if resp.Type != protocol.RespStatus {
			t.Errorf("connection %d: type = %q, want status", i, resp.Type)
		}
		conn.Close()
	}
}
