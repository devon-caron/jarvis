package daemon

import (
	"bufio"
	"bytes"
	"context"
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
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

const vllmReadyTimeout = 120 * time.Second

// VLLMBackend implements ModelBackend by spawning a vLLM server process
// and communicating with it over HTTP using the OpenAI-compatible API.
type VLLMBackend struct {
	mu         sync.RWMutex
	path       string
	port       int
	process    *exec.Cmd
	loaded     bool
	cfg        *config.Config
	httpClient *http.Client
}

// NewVLLMBackend creates a new VLLMBackend. It satisfies the backend
// factory signature used by NewModelRegistry.
func NewVLLMBackend(cfg *config.Config) ModelBackend {
	return &VLLMBackend{
		cfg: cfg,
		httpClient: &http.Client{
			Timeout: 0, // No timeout for streaming requests
		},
	}
}

// findFreePort finds an available TCP port by briefly listening on :0.
func findFreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port, nil
}

// LoadModel spawns a vLLM server process for the given model path on the
// specified GPUs.
func (v *VLLMBackend) LoadModel(path string, gpus []int, opts LoadOpts) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	port, err := findFreePort()
	if err != nil {
		return fmt.Errorf("cannot find free port: %w", err)
	}

	// Resolve vLLM binary path
	binary := v.cfg.VLLM.BinaryPath
	if binary == "" {
		binary = "vllm"
	}

	// Build command args
	args := []string{
		"serve", path,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
	}

	// Set context size from config
	if v.cfg.Inference.ContextSize > 0 {
		args = append(args, "--max-model-len", strconv.Itoa(v.cfg.Inference.ContextSize))
	}

	// GPU memory utilization
	if v.cfg.ModelOptions.GPUMemoryUtilization > 0 {
		args = append(args, "--gpu-memory-utilization", fmt.Sprintf("%.2f", v.cfg.ModelOptions.GPUMemoryUtilization))
	}

	// Tensor parallelism for NVLink multi-GPU
	if opts.NVLink && len(gpus) > 1 {
		args = append(args, "--tensor-parallel-size", strconv.Itoa(len(gpus)))
	}

	// Enforce eager mode (disables CUDA graph capturing for faster startup)
	if opts.EnforceEager {
		args = append(args, "--enforce-eager")
	}

	cmd := exec.Command(binary, args...)

	// Restrict GPU visibility via CUDA_VISIBLE_DEVICES
	cudaDevices := ""
	if len(gpus) > 0 {
		parts := make([]string, len(gpus))
		for i, g := range gpus {
			parts[i] = strconv.Itoa(g)
		}
		cudaDevices = strings.Join(parts, ",")
	}
	cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES="+cudaDevices)

	var stderrBuf bytes.Buffer
	cmd.Stderr = io.MultiWriter(&stderrBuf, log.Writer())

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start vLLM server: %w", err)
	}

	// Poll health endpoint until ready
	healthURL := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	deadline := time.Now().Add(vllmReadyTimeout)
	healthClient := &http.Client{Timeout: 2 * time.Second}
	ready := false

	// Also monitor process exit in background
	procDone := make(chan error, 1)
	go func() {
		procDone <- cmd.Wait()
	}()

	backoff := 500 * time.Millisecond
	for time.Now().Before(deadline) {
		select {
		case <-procDone:
			stderr := stderrBuf.String()
			if isVRAMError(stderr) {
				return fmt.Errorf("not enough VRAM to load model: %s", path)
			}
			if msg := vllmErrorMsg(stderr); msg != "" {
				return fmt.Errorf("vLLM server exited during startup: %s", msg)
			}
			return fmt.Errorf("vLLM server exited before becoming ready")
		default:
		}

		resp, err := healthClient.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				ready = true
				break
			}
		}
		time.Sleep(backoff)
		if backoff < 5*time.Second {
			backoff = backoff * 3 / 2 // 1.5x backoff
		}
	}

	if !ready {
		cmd.Process.Kill()
		<-procDone
		stderr := stderrBuf.String()
		if isVRAMError(stderr) {
			return fmt.Errorf("not enough VRAM to load model: %s", path)
		}
		if msg := vllmErrorMsg(stderr); msg != "" {
			return fmt.Errorf("vLLM server startup timed out after %s; last output: %s", vllmReadyTimeout, msg)
		}
		return fmt.Errorf("vLLM server did not become ready within %s", vllmReadyTimeout)
	}

	v.path = path
	v.port = port
	v.process = cmd
	v.loaded = true
	return nil
}

// vllmErrorMsg extracts the most useful line from vLLM stderr output.
func vllmErrorMsg(stderr string) string {
	s := strings.TrimSpace(stderr)
	if s == "" {
		return ""
	}
	lines := strings.Split(s, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return line
		}
	}
	return s
}

// vramErrorPatterns are substrings emitted by CUDA/vLLM to stderr when
// GPU memory allocation fails during model loading.
var vramErrorPatterns = []string{
	"cudaMalloc failed",
	"out of memory",
	"failed to allocate",
	"torch.OutOfMemoryError",
	"CUDA out of memory",
}

// isVRAMError scans stderr for CUDA/GPU out-of-memory indicators.
func isVRAMError(stderr string) bool {
	lower := strings.ToLower(stderr)
	for _, p := range vramErrorPatterns {
		if strings.Contains(lower, strings.ToLower(p)) {
			return true
		}
	}
	return false
}

// UnloadModel stops the vLLM server process and cleans up.
func (v *VLLMBackend) UnloadModel() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if !v.loaded {
		return nil
	}

	if v.process != nil && v.process.Process != nil {
		v.process.Process.Kill()
		v.process.Wait()
		v.process = nil
	}

	v.loaded = false
	v.path = ""
	v.port = 0
	return nil
}

// IsLoaded reports whether a vLLM server is running with a model loaded.
func (v *VLLMBackend) IsLoaded() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.loaded
}

// ModelPath returns the path of the model loaded in the vLLM server.
func (v *VLLMBackend) ModelPath() string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.path
}

// openAIChatRequest is the request body for /v1/chat/completions.
type openAIChatRequest struct {
	Model       string             `json:"model"`
	Messages    []openAIChatMsg    `json:"messages"`
	Stream      bool               `json:"stream"`
	MaxTokens   int                `json:"max_tokens,omitempty"`
	Temperature float64            `json:"temperature,omitempty"`
	TopP        float64            `json:"top_p,omitempty"`
}

type openAIChatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openAIStreamChunk represents a single SSE chunk from the streaming response.
type openAIStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

// RunChat sends a chat completion request to the vLLM server and streams
// tokens back via the onDelta callback.
func (v *VLLMBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	v.mu.RLock()
	port := v.port
	modelPath := v.path
	loaded := v.loaded
	v.mu.RUnlock()

	if !loaded {
		return fmt.Errorf("no model loaded")
	}

	// Build request
	oaiMsgs := make([]openAIChatMsg, len(msgs))
	for i, m := range msgs {
		oaiMsgs[i] = openAIChatMsg{Role: m.Role, Content: m.Content}
	}

	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = v.cfg.Inference.MaxTokens
	}
	temp := opts.Temperature
	if temp == 0 {
		temp = v.cfg.Inference.Temperature
	}
	topP := opts.TopP
	if topP == 0 {
		topP = v.cfg.Inference.TopP
	}

	reqBody := openAIChatRequest{
		Model:       modelPath,
		Messages:    oaiMsgs,
		Stream:      true,
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        topP,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := v.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("vLLM returned status %d: %s", resp.StatusCode, string(errBody))
	}

	// Parse SSE stream
	return parseSSEStream(resp.Body, onDelta)
}

// parseSSEStream reads an SSE stream from vLLM and calls onDelta for each token.
func parseSSEStream(r io.Reader, onDelta func(string)) error {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			return nil
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // skip malformed chunks
		}

		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			onDelta(chunk.Choices[0].Delta.Content)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("SSE stream error: %w", err)
	}

	return nil
}

// GetStatus returns model and GPU status info from the vLLM server.
func (v *VLLMBackend) GetStatus() (*protocol.ModelStatus, error) {
	v.mu.RLock()
	loaded := v.loaded
	port := v.port
	path := v.path
	v.mu.RUnlock()

	if !loaded {
		return nil, fmt.Errorf("no model loaded")
	}

	// Query vLLM /v1/models endpoint
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/models", port)
	resp, err := v.httpClient.Get(url)
	if err != nil {
		return &protocol.ModelStatus{ModelPath: path}, nil
	}
	defer resp.Body.Close()

	return &protocol.ModelStatus{ModelPath: path}, nil
}
