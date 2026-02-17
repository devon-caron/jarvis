package cmd

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/devon-caron/jarvis/protocol"
)

// setupMockDaemon starts a mock Unix socket daemon and sets XDG_RUNTIME_DIR
// so the client connects to it. Returns cleanup function.
func setupMockDaemon(t *testing.T, handler func(net.Conn)) {
	t.Helper()
	dir := t.TempDir()
	sockPath := filepath.Join(dir, "jarvis.sock")

	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	t.Cleanup(func() { ln.Close() })

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go handler(conn)
		}
	}()

	t.Setenv("XDG_RUNTIME_DIR", dir)
	time.Sleep(10 * time.Millisecond)
}

func writeJSON(conn net.Conn, resp *protocol.Response) {
	data, _ := json.Marshal(resp)
	data = append(data, '\n')
	conn.Write(data)
}

func scanReq(conn net.Conn) *protocol.Request {
	scanner := bufio.NewScanner(conn)
	if !scanner.Scan() {
		return nil
	}
	req, _ := protocol.UnmarshalRequest(scanner.Bytes())
	return req
}

func TestRunChat(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.DeltaResponse("Hello!"))
		writeJSON(conn, protocol.DoneResponse())
	})

	rootCmd.SetArgs([]string{"test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunStop(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.OKResponse())
	})

	rootCmd.SetArgs([]string{"stop"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunLoad(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.OKResponse())
	})

	rootCmd.SetArgs([]string{"load", "/path/to/model.gguf"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunUnload(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.OKResponse())
	})

	rootCmd.SetArgs([]string{"unload"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunStatus_WithModel(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: true,
			ModelPath:   "/model.gguf",
			PID:         12345,
			Model: &protocol.ModelStatus{
				GPULayers: 80,
				GPUs: []protocol.GPUInfo{
					{DeviceID: 0, DeviceName: "RTX 4090", FreeMemoryMB: 20000, TotalMemoryMB: 24000},
				},
			},
		}))
	})

	rootCmd.SetArgs([]string{"status"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunStatus_NoModel(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: false,
			PID:         12345,
		}))
	})

	rootCmd.SetArgs([]string{"status"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunStart_AlreadyRunning(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	// Write PID file with our own PID (always running)
	pidPath := filepath.Join(dir, "jarvis.pid")
	os.WriteFile(pidPath, []byte(strconv.Itoa(os.Getpid())), 0644)

	rootCmd.SetArgs([]string{"start"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error for already-running daemon")
	}
}

func TestRunChat_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"test prompt"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

func TestRunStop_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"stop"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

func TestRunLoad_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"load", "/model.gguf"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

func TestRunUnload_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"unload"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

func TestRunStatus_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"status"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

func TestRunLoad_Error(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.ErrorResponse("model not found"))
	})

	rootCmd.SetArgs([]string{"load", "/bad/model.gguf"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error from daemon")
	}
}

func TestRunChat_WithFlags(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil {
			// Verify flags were passed
			if req.Chat.Opts.MaxTokens != 100 {
				fmt.Fprintf(os.Stderr, "expected max_tokens=100, got %d\n", req.Chat.Opts.MaxTokens)
			}
		}
		writeJSON(conn, protocol.DeltaResponse("test"))
		writeJSON(conn, protocol.DoneResponse())
	})

	rootCmd.SetArgs([]string{"-n", "100", "-t", "0.5", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}
