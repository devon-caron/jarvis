package client

import (
	"bufio"
	"encoding/json"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/devon-caron/jarvis/protocol"
)

// startMockDaemon creates a Unix socket server that responds to requests.
func startMockDaemon(t *testing.T, handler func(net.Conn)) string {
	t.Helper()
	dir := t.TempDir()
	sockPath := filepath.Join(dir, "test.sock")

	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go handler(conn)
		}
	}()

	t.Cleanup(func() { ln.Close() })
	time.Sleep(10 * time.Millisecond)
	return sockPath
}

func writeResponse(conn net.Conn, resp *protocol.Response) {
	data, _ := json.Marshal(resp)
	data = append(data, '\n')
	conn.Write(data)
}

func TestClient_ConnectTo(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	c.Close()
}

func TestClient_ConnectTo_NoServer(t *testing.T) {
	_, err := ConnectTo("/nonexistent/path.sock")
	if err == nil {
		t.Error("expected error connecting to nonexistent socket")
	}
}

func TestClient_StreamChat(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()

		scanner := bufio.NewScanner(conn)
		if !scanner.Scan() {
			return
		}

		writeResponse(conn, protocol.DeltaResponse("Hello "))
		writeResponse(conn, protocol.DeltaResponse("world!"))
		writeResponse(conn, protocol.DoneResponse())
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	defer c.Close()

	var tokens []string
	err = c.StreamChat(&protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hi"}},
		},
	}, func(token string) {
		tokens = append(tokens, token)
	})

	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}
	if len(tokens) != 2 {
		t.Fatalf("got %d tokens, want 2", len(tokens))
	}
	if tokens[0] != "Hello " || tokens[1] != "world!" {
		t.Errorf("tokens = %v", tokens)
	}
}

func TestClient_StreamChat_Error(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()

		scanner := bufio.NewScanner(conn)
		scanner.Scan()

		writeResponse(conn, protocol.ErrorResponse("no model loaded"))
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	defer c.Close()

	err = c.StreamChat(&protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{{Role: "user", Content: "hi"}},
		},
	}, func(string) {})

	if err == nil {
		t.Error("expected error")
	}
	if err.Error() != "no model loaded" {
		t.Errorf("error = %q", err.Error())
	}
}

func TestClient_SendAndWaitOK(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		scanner.Scan()
		writeResponse(conn, protocol.OKResponse())
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	defer c.Close()

	err = c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqStop})
	if err != nil {
		t.Fatalf("SendAndWaitOK: %v", err)
	}
}

func TestClient_SendAndWaitOK_Error(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		scanner.Scan()
		writeResponse(conn, protocol.ErrorResponse("denied"))
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	defer c.Close()

	err = c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqStop})
	if err == nil {
		t.Error("expected error")
	}
}

func TestClient_SendAndReadStatus(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		scanner.Scan()
		writeResponse(conn, protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: true,
			ModelPath:   "/model.gguf",
			PID:         12345,
		}))
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	defer c.Close()

	status, err := c.SendAndReadStatus(&protocol.Request{Type: protocol.ReqStatus})
	if err != nil {
		t.Fatalf("SendAndReadStatus: %v", err)
	}

	if !status.Running {
		t.Error("Running should be true")
	}
	if !status.ModelLoaded {
		t.Error("ModelLoaded should be true")
	}
	if status.PID != 12345 {
		t.Errorf("PID = %d, want 12345", status.PID)
	}
}

func TestClient_SendAndReadStatus_Error(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		scanner.Scan()
		writeResponse(conn, protocol.ErrorResponse("not running"))
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	defer c.Close()

	_, err = c.SendAndReadStatus(&protocol.Request{Type: protocol.ReqStatus})
	if err == nil {
		t.Error("expected error")
	}
}

func TestClient_ReadResponse_ConnectionClosed(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		scanner := bufio.NewScanner(conn)
		scanner.Scan()
		conn.Close() // Close without sending a response
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	defer c.Close()

	c.Send(&protocol.Request{Type: protocol.ReqStatus})

	_, err = c.ReadResponse()
	if err == nil {
		t.Error("expected error for closed connection")
	}
}
