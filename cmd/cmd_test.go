package cmd

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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

func TestRunLoad_WithGPUs(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Load != nil {
			if len(req.Load.GPUs) != 2 || req.Load.GPUs[0] != 0 || req.Load.GPUs[1] != 1 {
				fmt.Fprintf(os.Stderr, "expected GPUs=[0,1], got %v\n", req.Load.GPUs)
			}
		}
		writeJSON(conn, protocol.OKResponse())
	})

	rootCmd.SetArgs([]string{"load", "-g", "0,1", "mymodel"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunLoad_WithTimeout(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Load != nil {
			if req.Load.Timeout != "30m" {
				fmt.Fprintf(os.Stderr, "expected timeout=30m, got %q\n", req.Load.Timeout)
			}
		}
		writeJSON(conn, protocol.OKResponse())
	})

	rootCmd.SetArgs([]string{"load", "-t", "30m", "mymodel"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunLoad_WithPath(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Load != nil {
			if req.Load.ModelPath != "/inline/model.gguf" {
				fmt.Fprintf(os.Stderr, "expected path=/inline/model.gguf, got %q\n", req.Load.ModelPath)
			}
		}
		writeJSON(conn, protocol.OKResponse())
	})

	rootCmd.SetArgs([]string{"load", "-p", "/inline/model.gguf"})
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

func TestRunUnload_WithName(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Unload != nil {
			if req.Unload.Name != "mymodel" {
				fmt.Fprintf(os.Stderr, "expected name=mymodel, got %q\n", req.Unload.Name)
			}
		}
		writeJSON(conn, protocol.OKResponse())
	})

	rootCmd.SetArgs([]string{"unload", "mymodel"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunStatus_WithModel(t *testing.T) {
	now := time.Now()
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
			Models: []protocol.SlotInfo{
				{
					Name:      "test",
					ModelPath: "/model.gguf",
					GPUs:      []int{0},
					LastUsed:  now,
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

	rootCmd.SetArgs([]string{"-n", "100", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunChat_WithModelFlag(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil {
			if req.Chat.Model != "llama70b" {
				fmt.Fprintf(os.Stderr, "expected model=llama70b, got %q\n", req.Chat.Model)
			}
		}
		writeJSON(conn, protocol.DeltaResponse("test"))
		writeJSON(conn, protocol.DoneResponse())
	})

	rootCmd.SetArgs([]string{"-m", "llama70b", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunChat_BatchMode(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.DeltaResponse("Hello!"))
		writeJSON(conn, protocol.DoneResponse())
	})

	rootCmd.SetArgs([]string{"-b", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunChat_WebSearchFlag(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil && !req.Chat.WebSearch {
			fmt.Fprintf(os.Stderr, "expected web_search=true\n")
		}
		writeJSON(conn, protocol.DeltaResponse("test"))
		writeJSON(conn, protocol.DoneResponse())
	})

	rootCmd.SetArgs([]string{"-w", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunChat_SystemPromptFlag(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil && req.Chat.SystemPrompt != "Be terse" {
			fmt.Fprintf(os.Stderr, "expected system_prompt='Be terse'\n")
		}
		writeJSON(conn, protocol.DeltaResponse("ok"))
		writeJSON(conn, protocol.DoneResponse())
	})

	rootCmd.SetArgs([]string{"--system", "Be terse", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunChat_TemperatureFlag(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.DeltaResponse("ok"))
		writeJSON(conn, protocol.DoneResponse())
	})

	rootCmd.SetArgs([]string{"-t", "0.5", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunRegister(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	// Create a fake model file to pass path validation
	modelPath := filepath.Join(dir, "model.gguf")
	os.WriteFile(modelPath, []byte("fake"), 0644)

	rootCmd.SetArgs([]string{"models", "register", "mymodel", modelPath})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	// Verify the config was saved
	cfgPath := filepath.Join(dir, "jarvis", "config.yaml")
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		t.Fatalf("config file should exist: %v", err)
	}
	if !strings.Contains(string(data), "mymodel") {
		t.Errorf("config should contain model name, got: %s", string(data))
	}
}

func TestRunUnregister(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	// Create a config with a model
	cfgDir := filepath.Join(dir, "jarvis")
	os.MkdirAll(cfgDir, 0755)
	os.WriteFile(filepath.Join(cfgDir, "config.yaml"), []byte("models:\n  mymodel:\n    path: /path/to/model.gguf\n"), 0644)

	rootCmd.SetArgs([]string{"models", "unregister", "mymodel"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	// Verify the model was removed
	data, _ := os.ReadFile(filepath.Join(cfgDir, "config.yaml"))
	if strings.Contains(string(data), "mymodel") {
		t.Errorf("config should not contain model name after unregister, got: %s", string(data))
	}
}

func TestRunUnregister_NotFound(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	rootCmd.SetArgs([]string{"models", "unregister", "nonexistent"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

func TestRunRegister_BadPath(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	rootCmd.SetArgs([]string{"models", "register", "mymodel", "/nonexistent/model.gguf"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error for nonexistent model path")
	}
}

func TestRunLoad_NoArgs(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"load"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when no model name or path given")
	}
}

func TestRunStatus_OldFormat(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		// Old single-model format (no Models slice)
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

func TestRunStatus_MultiModel(t *testing.T) {
	now := time.Now()
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: true,
			PID:         12345,
			Models: []protocol.SlotInfo{
				{
					Name:      "m1",
					ModelPath: "/m1.gguf",
					GPUs:      []int{0},
					Timeout:   "30m0s",
					LastUsed:  now,
					GPUInfo: []protocol.GPUInfo{
						{DeviceID: 0, DeviceName: "RTX 4090", FreeMemoryMB: 20000, TotalMemoryMB: 24000},
					},
				},
				{
					Name:      "m2",
					ModelPath: "/m2.gguf",
					GPUs:      []int{1},
					LastUsed:  now,
				},
			},
		}))
	})

	rootCmd.SetArgs([]string{"status"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunModelsLs(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	// Create a config with some models
	cfgDir := filepath.Join(dir, "jarvis")
	os.MkdirAll(cfgDir, 0755)
	cfgContent := "models:\n  llama70b:\n    path: /path/to/llama70b.gguf\n    context_size: 16384\n  gemma12b:\n    path: /path/to/gemma12b.gguf\n"
	os.WriteFile(filepath.Join(cfgDir, "config.yaml"), []byte(cfgContent), 0644)

	rootCmd.SetArgs([]string{"models", "ls"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

func TestRunModelsLs_Empty(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	rootCmd.SetArgs([]string{"models", "ls"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}
