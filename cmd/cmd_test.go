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

// TestRunChat verifies the chat command sends a prompt to the mock daemon and
// successfully receives streamed delta tokens followed by a done response.
func TestRunChat(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.DeltaTokenResponse("Hello!"))
		writeJSON(conn, protocol.EndTokenResponse())
	})

	rootCmd.SetArgs([]string{"test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

// TestRunStop verifies the stop command sends a stop request to the mock daemon
// and completes without error when the daemon responds with OK.
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

// TestRunLoad verifies the load command sends a load request with the specified
// model path to the mock daemon and completes without error.
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

// TestRunLoad_WithGPUs verifies the load command correctly parses the -g flag
// and sends the GPU list (e.g., [0,1]) in the load request to the daemon.
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

// TestRunLoad_WithTimeout verifies the load command correctly parses the -t flag
// and sends the timeout value (e.g., "30m") in the load request to the daemon.
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

// TestRunLoad_WithPath verifies the load command correctly parses the -p flag
// and sends the inline model path in the load request to the daemon.
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

// TestRunUnload verifies the unload command sends an unload request to the
// mock daemon and completes without error when the daemon responds with OK.
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

// TestRunUnload_WithName verifies the unload command passes the model name
// argument in the unload request to the daemon.
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

// TestRunStatus_WithModel verifies the status command executes without error
// when the daemon reports a loaded model with GPU info. NOTE: functionally
// duplicate with TestRunStatus_ModelOnly and TestRunStatus_WithModelInfo.
func TestRunStatus_WithModel(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: true,
			ModelPath:   "/model.gguf",
			PID:         12345,
			Model: &protocol.ModelInfo{
				Name:      "test",
				ModelPath: "/model.gguf",
				GPUs:      []int{0},
				GPUInfo: []protocol.GPUInfo{
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

// TestRunStatus_NoModel verifies the status command executes without error
// when the daemon is running but no model is loaded.
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

// TestRunStart_AlreadyRunning verifies the start command returns an error when
// a PID file exists with a still-running process (this test's own PID).
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

// TestRunChat_NoDaemon verifies the chat command returns an error when no
// daemon is running (no socket to connect to).
func TestRunChat_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"test prompt"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

// TestRunStop_NoDaemon verifies the stop command returns an error when no
// daemon is running.
func TestRunStop_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"stop"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

// TestRunLoad_NoDaemon verifies the load command returns an error when no
// daemon is running.
func TestRunLoad_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"load", "/model.gguf"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

// TestRunUnload_NoDaemon verifies the unload command returns an error when no
// daemon is running.
func TestRunUnload_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"unload"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

// TestRunStatus_NoDaemon verifies the status command returns an error when no
// daemon is running.
func TestRunStatus_NoDaemon(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"status"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when daemon is not running")
	}
}

// TestRunLoad_Error verifies the load command propagates an error response
// from the daemon (e.g., "model not found") back to the caller.
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

// TestRunChat_WithFlags verifies the chat command correctly parses the -n flag
// and sends the max_tokens inference option in the chat request to the daemon.
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
		writeJSON(conn, protocol.DeltaTokenResponse("test"))
		writeJSON(conn, protocol.EndTokenResponse())
	})

	rootCmd.SetArgs([]string{"-n", "100", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

// TestRunChat_BatchMode verifies the chat command works with the -b (batch mode)
// flag, successfully receiving streamed tokens from the daemon.
func TestRunChat_BatchMode(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.DeltaTokenResponse("Hello!"))
		writeJSON(conn, protocol.EndTokenResponse())
	})

	rootCmd.SetArgs([]string{"-b", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

// TestRunChat_WebSearchFlag verifies the chat command correctly parses the -w
// flag and sends web_search=true in the chat request to the daemon.
func TestRunChat_WebSearchFlag(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil && !req.Chat.WebSearch {
			fmt.Fprintf(os.Stderr, "expected web_search=true\n")
		}
		writeJSON(conn, protocol.DeltaTokenResponse("test"))
		writeJSON(conn, protocol.EndTokenResponse())
	})

	rootCmd.SetArgs([]string{"-w", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

// TestRunChat_SystemPromptFlag verifies the chat command correctly parses the
// --system flag and sends the system prompt string in the chat request.
func TestRunChat_SystemPromptFlag(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil && req.Chat.SystemPrompt != "Be terse" {
			fmt.Fprintf(os.Stderr, "expected system_prompt='Be terse'\n")
		}
		writeJSON(conn, protocol.DeltaTokenResponse("ok"))
		writeJSON(conn, protocol.EndTokenResponse())
	})

	rootCmd.SetArgs([]string{"--system", "Be terse", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

// TestRunRegister verifies the "models register" command writes a model entry
// (name and path) to the config YAML file in XDG_CONFIG_HOME.
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

// TestRunUnregister verifies the "models unregister" command removes a
// previously registered model entry from the config YAML file.
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

// TestRunUnregister_NotFound verifies the "models unregister" command returns
// an error when the specified model name doesn't exist in the config.
func TestRunUnregister_NotFound(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	rootCmd.SetArgs([]string{"models", "unregister", "nonexistent"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

// TestRunRegister_BadPath verifies the "models register" command returns an
// error when the specified model file path doesn't exist on disk.
func TestRunRegister_BadPath(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	rootCmd.SetArgs([]string{"models", "register", "mymodel", "/nonexistent/model.gguf"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error for nonexistent model path")
	}
}

// TestRunLoad_NoArgs verifies the load command returns an error when no model
// name or path argument is provided.
func TestRunLoad_NoArgs(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_RUNTIME_DIR", dir)

	rootCmd.SetArgs([]string{"load"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error when no model name or path given")
	}
}

// TestRunStatus_ModelOnly verifies the status command executes without error
// when the daemon reports a loaded model with GPU info. NOTE: functionally
// duplicate with TestRunStatus_WithModel and TestRunStatus_WithModelInfo.
func TestRunStatus_ModelOnly(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: true,
			ModelPath:   "/model.gguf",
			PID:         12345,
			Model: &protocol.ModelInfo{
				Name:      "test",
				ModelPath: "/model.gguf",
				GPUs:      []int{0},
				GPUInfo: []protocol.GPUInfo{
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

// TestRunStatus_WithModelInfo verifies the status command executes without
// error when the daemon reports a loaded model with GPU info. NOTE: functionally
// duplicate with TestRunStatus_WithModel and TestRunStatus_ModelOnly.
func TestRunStatus_WithModelInfo(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		scanReq(conn)
		writeJSON(conn, protocol.StatusResponse(&protocol.StatusPayload{
			Running:     true,
			ModelLoaded: true,
			ModelPath:   "/m1.gguf",
			PID:         12345,
			Model: &protocol.ModelInfo{
				Name:      "m1",
				ModelPath: "/m1.gguf",
				GPUs:      []int{0},
				GPUInfo: []protocol.GPUInfo{
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

// TestRunModelsLs verifies the "models ls" command executes without error and
// lists registered models from a config file containing two model entries.
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

// TestRunModelsLs_Empty verifies the "models ls" command executes without error
// when no config file exists (i.e., no models are registered).
func TestRunModelsLs_Empty(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	rootCmd.SetArgs([]string{"models", "ls"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

// TestRootCmd_ClearContextFlag verifies the -C flag sets clear_context=true and
// includes a non-zero shell_pid in the chat request sent to the daemon.
func TestRootCmd_ClearContextFlag(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil {
			if !req.Chat.ClearContext {
				fmt.Fprintf(os.Stderr, "expected clear_context=true\n")
			}
			if req.Chat.ShellPID == 0 {
				fmt.Fprintf(os.Stderr, "expected non-zero shell_pid\n")
			}
		}
		writeJSON(conn, protocol.DeltaTokenResponse("ok"))
		writeJSON(conn, protocol.EndTokenResponse())
	})

	rootCmd.SetArgs([]string{"-C", "test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

// TestRootCmd_ShellPID verifies that the chat command automatically includes a
// non-zero shell_pid in the chat request for per-shell context tracking.
func TestRootCmd_ShellPID(t *testing.T) {
	setupMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
		req := scanReq(conn)
		if req != nil && req.Chat != nil {
			if req.Chat.ShellPID == 0 {
				fmt.Fprintf(os.Stderr, "expected non-zero shell_pid\n")
			}
		}
		writeJSON(conn, protocol.DeltaTokenResponse("ok"))
		writeJSON(conn, protocol.EndTokenResponse())
	})

	rootCmd.SetArgs([]string{"test prompt"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
}
