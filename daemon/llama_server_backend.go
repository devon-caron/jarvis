package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

const serverReadyTimeout = 120 * time.Second

// LlamaServerBackend implements ModelBackend by spawning a llama-server process
// and communicating via HTTP/SSE.
type LlamaServerBackend struct {
	mu         sync.RWMutex
	path       string
	port       int
	process    *exec.Cmd
	loaded     bool
	gpus       []int
	cfg        *config.Config
	httpClient *http.Client
}

// NewLlamaServerBackend creates a new LlamaServerBackend. It satisfies the
// backend factory signature used by NewModelRegistry.
func NewLlamaServerBackend(cfg *config.Config) ModelBackend {
	return &LlamaServerBackend{
		cfg: cfg,
		httpClient: &http.Client{
			Timeout: 0, // no timeout for streaming
		},
	}
}

// NormalizeSplitMode validates and normalizes a split mode string.
// Accepted inputs: "", "none", "l", "layer", "r", "row", "g", "graph".
func NormalizeSplitMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "none":
		return "", nil
	case "l", "layer":
		return "layer", nil
	case "r", "row":
		return "row", nil
	case "g", "graph":
		return "graph", nil
	default:
		return "", fmt.Errorf("invalid split mode %q: must be l(ayer), r(ow), or g(raph)", mode)
	}
}

// LoadModel spawns a llama-server process and waits for it to become healthy.
func (b *LlamaServerBackend) LoadModel(path string, gpus []int, contextSize int, splitMode string, parallel int) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if err := validateGGUF(path); err != nil {
		return err
	}

	// Allocate an ephemeral port.
	port, err := allocatePort()
	if err != nil {
		return fmt.Errorf("failed to allocate port: %w", err)
	}

	// Resolve llama-server binary path.
	binary := b.cfg.LlamaServer.BinaryPath
	if binary == "" {
		binary = "llama-server"
	}

	// Build command args.
	args := []string{
		"-m", path,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"-ngl", "999",
		"-c", strconv.Itoa(contextSize),
	}
	if b.cfg.ModelOptions.MLock {
		args = append(args, "--mlock")
	}
	splitMode, err = NormalizeSplitMode(splitMode)
	if err != nil {
		return err
	}
	if splitMode != "" {
		if len(gpus) == 1 {
			return fmt.Errorf("split mode %q requires multiple GPUs; use -g 0,1 or omit -g", splitMode)
		}
		if splitMode == "graph" {
			if err := checkNCCL(); err != nil {
				return err
			}
		}
		args = append(args, "-sm", splitMode)
	}
	if parallel > 0 {
		args = append(args, "--parallel", strconv.Itoa(parallel))
	}

	cmd := exec.Command(binary, args...)

	// Set CUDA_VISIBLE_DEVICES.
	if len(gpus) > 0 {
		parts := make([]string, len(gpus))
		for i, g := range gpus {
			parts[i] = strconv.Itoa(g)
		}
		cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES="+strings.Join(parts, ","))
	} else {
		cmd.Env = os.Environ()
	}

	var stderrBuf bytes.Buffer
	cmd.Stderr = io.MultiWriter(&stderrBuf, log.Writer())

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	// Monitor process exit in a goroutine so we can detect startup failures early.
	exitCh := make(chan error, 1)
	go func() {
		exitCh <- cmd.Wait()
	}()

	// Poll /health until ready.
	healthURL := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	if err := b.waitForHealth(healthURL, exitCh, &stderrBuf); err != nil {
		// Kill the process if it's still running.
		cmd.Process.Signal(syscall.SIGKILL)
		<-exitCh
		return err
	}

	b.path = path
	b.port = port
	b.process = cmd
	b.loaded = true
	b.gpus = gpus
	return nil
}

// waitForHealth polls the health endpoint with backoff until ready or timeout.
func (b *LlamaServerBackend) waitForHealth(url string, exitCh <-chan error, stderrBuf *bytes.Buffer) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.After(serverReadyTimeout)
	interval := 100 * time.Millisecond

	for {
		select {
		case err := <-exitCh:
			// Process exited before becoming ready.
			stderr := stderrBuf.String()
			if isVRAMError(stderr) {
				return fmt.Errorf("not enough VRAM to load model")
			}
			if msg := serverErrorMsg(stderr); msg != "" {
				return fmt.Errorf("llama-server failed: %s", msg)
			}
			if err != nil {
				return fmt.Errorf("llama-server exited: %w", err)
			}
			return fmt.Errorf("llama-server exited unexpectedly")
		case <-deadline:
			stderr := stderrBuf.String()
			if isVRAMError(stderr) {
				return fmt.Errorf("not enough VRAM to load model")
			}
			return fmt.Errorf("llama-server did not become ready within %s", serverReadyTimeout)
		default:
		}

		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		time.Sleep(interval)
		if interval < 2*time.Second {
			interval = interval * 3 / 2
		}
	}
}

// UnloadModel stops the llama-server process.
func (b *LlamaServerBackend) UnloadModel() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.loaded || b.process == nil {
		b.loaded = false
		return nil
	}

	// Send SIGTERM, wait up to 5s, then SIGKILL.
	b.process.Process.Signal(syscall.SIGTERM)

	done := make(chan struct{})
	go func() {
		b.process.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		b.process.Process.Signal(syscall.SIGKILL)
		<-done
	}

	b.loaded = false
	b.path = ""
	b.port = 0
	b.process = nil
	return nil
}

// IsLoaded reports whether a llama-server process is running.
func (b *LlamaServerBackend) IsLoaded() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.loaded
}

// ModelPath returns the path of the loaded model.
func (b *LlamaServerBackend) ModelPath() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.path
}

// RunChat sends a chat completion request to llama-server and streams the response.
func (b *LlamaServerBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	b.mu.RLock()
	port := b.port
	loaded := b.loaded
	b.mu.RUnlock()

	if !loaded {
		return fmt.Errorf("no model loaded")
	}

	// Build OpenAI-compatible request body.
	type chatMsg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	apiMsgs := make([]chatMsg, len(msgs))
	for i, m := range msgs {
		apiMsgs[i] = chatMsg{Role: m.Role, Content: m.Content}
	}

	body := map[string]interface{}{
		"messages": apiMsgs,
		"stream":   true,
	}
	if opts.MaxTokens > 0 {
		body["max_tokens"] = opts.MaxTokens
	}
	if opts.Temperature > 0 {
		body["temperature"] = opts.Temperature
	}
	if opts.TopP > 0 {
		body["top_p"] = opts.TopP
	}
	if opts.TopK > 0 {
		body["top_k"] = opts.TopK
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chat request returned %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse SSE response.
	return parseSSE(resp.Body, onDelta)
}

// parseSSE reads SSE lines from r, extracting delta content and calling onDelta.
func parseSSE(r io.Reader, onDelta func(string)) error {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			onDelta(chunk.Choices[0].Delta.Content)
		}
	}
	return scanner.Err()
}

// GetStatus returns model status info.
func (b *LlamaServerBackend) GetStatus() (*protocol.ModelStatus, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if !b.loaded {
		return nil, fmt.Errorf("no model loaded")
	}

	gpus := make([]protocol.GPUInfo, len(b.gpus))
	for i, g := range b.gpus {
		gpus[i] = protocol.GPUInfo{DeviceID: g}
	}

	return &protocol.ModelStatus{
		ModelPath: b.path,
		GPUs:      gpus,
	}, nil
}

// allocatePort finds a free ephemeral port.
func allocatePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	return port, nil
}

// validateGGUF performs fast pre-checks on a model file before handing it to
// llama-server. It catches missing files, non-GGUF files, and unsupported
// GGUF versions early with clear error messages.
func validateGGUF(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("model file not found: %s", path)
		}
		return fmt.Errorf("cannot open model file: %w", err)
	}
	defer f.Close()

	// GGUF header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes
	var header [24]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return fmt.Errorf("file too small to be a valid GGUF model: %s", path)
	}

	if string(header[0:4]) != "GGUF" {
		return fmt.Errorf("not a valid GGUF model file: %s", path)
	}

	version := binary.LittleEndian.Uint32(header[4:8])
	if version < 2 || version > 3 {
		return fmt.Errorf("unsupported GGUF version %d: %s", version, path)
	}

	return nil
}

// serverErrorMsg extracts the most useful line from server stderr.
func serverErrorMsg(stderr string) string {
	s := strings.TrimSpace(stderr)
	if s == "" {
		return ""
	}
	lines := strings.Split(s, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return strings.TrimPrefix(line, "Error: ")
		}
	}
	return strings.TrimPrefix(s, "Error: ")
}

// vramErrorPatterns are substrings emitted by llama.cpp / CUDA when GPU
// memory allocation fails during model loading.
var vramErrorPatterns = []string{
	"cudamalloc failed",
	"out of memory",
	"failed to allocate",
	"unable to allocate",
}

// checkNCCL verifies that the NCCL library is available on the system.
// Required for -sm graph (NVLink tensor parallelism).
func checkNCCL() error {
	out, err := exec.Command("ldconfig", "-p").Output()
	if err != nil {
		return fmt.Errorf("failed to check for NCCL: %w", err)
	}
	if !strings.Contains(string(out), "libnccl") {
		return fmt.Errorf("NCCL library not found; install libnccl-dev for NVLink support (-sm graph)")
	}
	return nil
}

// isVRAMError scans stderr for CUDA/GPU out-of-memory indicators.
func isVRAMError(stderr string) bool {
	lower := strings.ToLower(stderr)
	for _, p := range vramErrorPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}
