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
	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/protocol"
)

const serverReadyTimeout = 30 * time.Second

type Backend struct {
	mu         sync.RWMutex
	path       string
	port       int
	process    *exec.Cmd
	loaded     bool
	gpus       []int
	httpClient *http.Client
	config     *config.Config
}

// LoadOpts holds llama-server tuning parameters for model loading.
type LoadOpts struct {
	ContextSize    int
	SplitMode      string
	Parallel       int // num concurrent inference requests per server
	MLock          bool
	FlashAttention bool
	BatchSize      int    // micro-batch size → llama-server -ub
	TensorSplit    string // GPU weight distribution → llama-server -ts
}

// ModelBackend abstracts the LLM engine for testability.
// The real implementation wraps llama-server; tests use a mock.
type ModelBackend interface {
	// LoadModel loads a model from the given path onto the specified GPUs.
	// The context allows cancellation (e.g. when the client disconnects).
	LoadModel(ctx context.Context, path string, gpus []int, opts LoadOpts) error
	// UnloadModel frees the currently loaded model.
	UnloadModel() error
	// IsLoaded returns true if a model is currently loaded.
	IsLoaded() bool
	// ModelPath returns the path of the currently loaded model.
	ModelPath() string
	// RunChat streams a chat response, calling onDelta for each token.
	RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error
	// GetStatus returns model and GPU status info.
	GetStatus() (*protocol.ModelStatus, error)
}

func NewServerBackend(config *config.Config) ModelBackend {
	return &Backend{
		config:     config,
		httpClient: &http.Client{Timeout: 0},
	}
}

// LoadModel loads a model from the given path onto the specified GPUs.
// The context allows cancellation (e.g. when the client disconnects).
func (b *Backend) LoadModel(ctx context.Context, modelPath string, gpus []int, opts LoadOpts) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Ensure file to be loaded is valid
	if err := validateGGUF(modelPath); err != nil {
		return fmt.Errorf("invalid model file: %w", err)
	}

	// allocate free ephemeral port
	port, err := allocatePort()
	if err != nil {
		return fmt.Errorf("failed to allocate port: %w", err)
	}

	splitMode, err := NormalizeSplitMode(opts.SplitMode)
	if err != nil {
		return fmt.Errorf("invalid split mode: %w", err)
	}

	binary := b.config.LlamaServer.Binary()

	// Validate binary's existence before running
	if _, err := os.Stat(binary); os.IsNotExist(err) {
		return fmt.Errorf("llama-server binary not found: %s", binary)
	}

	// Build command args.
	args := []string{
		"-m", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"-c", strconv.Itoa(opts.ContextSize),
	}

	if opts.MLock {
		args = append(args, "--mlock")
	}

	if splitMode != "" {
		args = append(args, "-sm", splitMode)
	}

	if opts.Parallel > 0 {
		args = append(args, "--parallel", strconv.Itoa(opts.Parallel))
	}

	if opts.FlashAttention {
		args = append(args, "-fa", "on")
	}

	if opts.BatchSize > 0 {
		args = append(args, "-ub", strconv.Itoa(opts.BatchSize))
	}

	if opts.TensorSplit != "" {
		args = append(args, "-ts", opts.TensorSplit)
	}

	// Build command.
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
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		exitCh <- cmd.Wait()
	}()

	// Poll /health until ready.
	healthURL := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	if err := b.waitForHealth(ctx, healthURL, exitCh, &stderrBuf); err != nil {
		// Kill the process if it's still running.
		cmd.Process.Signal(syscall.SIGKILL)
		wg.Wait()
		return err
	}

	b.path = modelPath
	b.port = port
	b.process = cmd
	b.loaded = true
	b.gpus = gpus
	return nil
}

func (b *Backend) UnloadModel() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.loaded || b.process == nil {
		b.loaded = false
		return nil
	}

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
	b.process = nil
	b.port = 0
	b.path = ""
	b.gpus = nil
	return nil
}

func (b *Backend) IsLoaded() bool {
	return b.loaded
}

func (b *Backend) ModelPath() string {
	return b.path
}

func (b *Backend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onToken func(string)) error {
	b.mu.Lock()
	port := b.port
	loaded := b.loaded
	b.mu.Unlock()

	if !loaded {
		return fmt.Errorf("no model loaded")
	}

	type chatMsg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	apiMsgs := make([]chatMsg, len(msgs))
	for i, msg := range msgs {
		apiMsgs[i] = chatMsg{
			Role:    msg.Role,
			Content: msg.Content,
		}
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
		return err
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chat request returned %d: %s", resp.StatusCode, string(respBody))
	}

	return parseSSE(resp.Body, onToken)
}

func parseSSE(r io.ReadCloser, onToken func(string)) error {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == internal.OPENAI_COMPAT_SSE_DONE {
			break
		}

		// Parse OpenAI-compatible SSE chunk
		// here we are only expecting one 'choice' from the server so we only take the first one.
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
			onToken(chunk.Choices[0].Delta.Content)
		}
	}

	return scanner.Err()
}

func (b *Backend) GetStatus() (*protocol.ModelStatus, error) {
	return nil, fmt.Errorf("unimplemented")
}

func (b *Backend) waitForHealth(ctx context.Context, url string, exitCh <-chan error, stderrBuf *bytes.Buffer) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.After(serverReadyTimeout)
	interval := 100 * time.Millisecond

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("load cancelled")
		case err := <-exitCh:
			stderr := stderrBuf.String()
			if isVRAMError(stderr) {
				return fmt.Errorf("llama-server exited due to VRAM error: %w\n%s", err, stderr)
			}
			if err != nil {
				return fmt.Errorf("llama-server exited: %w\n%s", err, stderr)
			}
			return fmt.Errorf("llama-server exited with unknown error: %s", stderr)
		case <-deadline:
			stderr := stderrBuf.String()
			if isVRAMError(stderr) {
				return fmt.Errorf("llama-server did not become ready within %s due to VRAM error: %s", serverReadyTimeout, stderr)
			}
			return fmt.Errorf("llama-server did not become ready within %s: %s", serverReadyTimeout, stderr)
		default:
		}

		req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
		resp, err := client.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("load cancelled")
		case err := <-exitCh:
			stderr := stderrBuf.String()
			if isVRAMError(stderr) {
				return fmt.Errorf("llama-server exited due to VRAM error: %w\n%s", err, stderr)
			}
			if err != nil {
				return fmt.Errorf("llama-server exited: %w\n%s", err, stderr)
			}
			return fmt.Errorf("llama-server exited with unknown error: %s", stderr)
		case <-time.After(interval):
		}

		if interval < 2*time.Second {
			interval = interval * 3 / 2
		}
	}
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

// isVRAMError scans stderr for CUDA/GPU out-of-memory indicators.
func isVRAMError(stderr string) bool {
	lower := strings.ToLower(stderr)
	for _, p := range internal.VRAM_ERROR_PATTERNS {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}
