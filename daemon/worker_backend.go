package daemon

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/protocol"
)

const workerReadyTimeout = 120 * time.Second

// WorkerBackend implements ModelBackend by spawning a jarvis _worker subprocess
// with CUDA_VISIBLE_DEVICES set to the requested GPUs and proxying all requests
// to the worker's Unix socket.
type WorkerBackend struct {
	mu         sync.RWMutex
	path       string
	socketPath string
	process    *exec.Cmd
	loaded     bool
}

// NewWorkerBackend creates a new WorkerBackend. It satisfies the backend
// factory signature used by NewModelRegistry.
func NewWorkerBackend(cfg *config.Config) ModelBackend {
	return &WorkerBackend{}
}

// LoadModel spawns a worker subprocess with CUDA_VISIBLE_DEVICES restricted to
// the given GPU indices and waits for it to signal readiness.
func (w *WorkerBackend) LoadModel(path string, gpus []int) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Build CUDA_VISIBLE_DEVICES from the requested physical GPU IDs.
	cudaDevices := ""
	if len(gpus) > 0 {
		parts := make([]string, len(gpus))
		for i, g := range gpus {
			parts[i] = strconv.Itoa(g)
		}
		cudaDevices = strings.Join(parts, ",")
	}

	// Determine socket path from the model file name.
	socketPath := internal.WorkerSocketPath(path)

	// Find the executable path (os.Executable resolves the running binary).
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("cannot determine executable path: %w", err)
	}

	// Create a pipe for the readiness handshake.
	// The worker writes one byte on fd 3 (ExtraFiles[0]) when ready.
	rPipe, wPipe, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("cannot create readiness pipe: %w", err)
	}

	args := []string{
		"_worker",
		"--socket", socketPath,
		"--path", path,
	}

	cmd := exec.Command(exe, args...)
	cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES="+cudaDevices)
	cmd.ExtraFiles = []*os.File{wPipe} // fd 3

	var stderrBuf bytes.Buffer
	cmd.Stderr = io.MultiWriter(&stderrBuf, log.Writer())

	if err := cmd.Start(); err != nil {
		rPipe.Close()
		wPipe.Close()
		return fmt.Errorf("failed to start worker: %w", err)
	}

	// Close write end in parent; worker owns it.
	wPipe.Close()

	// Wait for the readiness signal with a timeout.
	ready := make(chan error, 1)
	go func() {
		buf := make([]byte, 1)
		_, err := rPipe.Read(buf)
		rPipe.Close()
		if err != nil && err != io.EOF {
			ready <- fmt.Errorf("worker readiness pipe error: %w", err)
			return
		}
		// Any byte (or clean EOF after byte) means ready. EOF alone means failure.
		if err == io.EOF {
			ready <- fmt.Errorf("worker exited before signaling ready")
			return
		}
		ready <- nil
	}()

	select {
	case readyErr := <-ready:
		if readyErr != nil {
			cmd.Process.Kill()
			cmd.Wait() // flush stderr before reading stderrBuf
			stderr := stderrBuf.String()
			if isVRAMError(stderr) {
				return fmt.Errorf("not enough VRAM to load model: %s", path)
			}
			if msg := workerErrorMsg(stderr); msg != "" {
				return fmt.Errorf("worker failed: %s", msg)
			}
			return readyErr
		}
	case <-time.After(workerReadyTimeout):
		cmd.Process.Kill()
		cmd.Wait()
		stderr := stderrBuf.String()
		if isVRAMError(stderr) {
			return fmt.Errorf("not enough VRAM to load model: %s", path)
		}
		if msg := workerErrorMsg(stderr); msg != "" {
			return fmt.Errorf("worker startup timed out after %s; last output: %s", workerReadyTimeout, msg)
		}
		return fmt.Errorf("worker did not become ready within %s", workerReadyTimeout)
	}

	w.path = path
	w.socketPath = socketPath
	w.process = cmd
	w.loaded = true
	return nil
}

// workerErrorMsg extracts the most useful line from worker stderr.
// Cobra prefixes returned errors with "Error: "; this strips that prefix.
func workerErrorMsg(stderr string) string {
	s := strings.TrimSpace(stderr)
	if s == "" {
		return ""
	}
	// Take the last non-empty line (most specific in a wrapped error chain).
	lines := strings.Split(s, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return strings.TrimPrefix(line, "Error: ")
		}
	}
	return strings.TrimPrefix(s, "Error: ")
}

// vramErrorPatterns are substrings emitted by llama.cpp / CUDA to stderr when
// GPU memory allocation fails during model loading.
var vramErrorPatterns = []string{
	"cudaMalloc failed",
	"out of memory",
	"failed to allocate",
}

// isVRAMError scans worker stderr for CUDA/GPU out-of-memory indicators
// that llama.cpp prints when the model doesn't fit in VRAM.
func isVRAMError(stderr string) bool {
	lower := strings.ToLower(stderr)
	for _, p := range vramErrorPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// UnloadModel sends a stop request to the worker, waits for it to exit, and
// removes the socket file.
func (w *WorkerBackend) UnloadModel() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.loaded {
		return nil
	}

	// Best-effort stop request.
	c, err := client.ConnectTo(w.socketPath)
	if err == nil {
		c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqStop})
		c.Close()
	}

	// Reap the process.
	if w.process != nil {
		w.process.Wait()
		w.process = nil
	}

	os.Remove(w.socketPath)
	w.loaded = false
	w.path = ""
	w.socketPath = ""
	return nil
}

// IsLoaded reports whether a worker is running with a model loaded.
func (w *WorkerBackend) IsLoaded() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.loaded
}

// ModelPath returns the path of the model loaded in the worker.
func (w *WorkerBackend) ModelPath() string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.path
}

// RunChat proxies a chat request to the worker subprocess.
func (w *WorkerBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	w.mu.RLock()
	socketPath := w.socketPath
	loaded := w.loaded
	w.mu.RUnlock()

	if !loaded {
		return fmt.Errorf("no model loaded")
	}

	c, err := client.ConnectTo(socketPath)
	if err != nil {
		return fmt.Errorf("cannot connect to worker: %w", err)
	}
	defer c.Close()

	req := &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: msgs,
			Opts:     opts,
		},
	}
	return c.StreamChat(req, onDelta)
}

// GetStatus proxies a status request to the worker subprocess.
func (w *WorkerBackend) GetStatus() (*protocol.ModelStatus, error) {
	w.mu.RLock()
	socketPath := w.socketPath
	loaded := w.loaded
	w.mu.RUnlock()

	if !loaded {
		return nil, fmt.Errorf("no model loaded")
	}

	c, err := client.ConnectTo(socketPath)
	if err != nil {
		return nil, fmt.Errorf("cannot connect to worker: %w", err)
	}
	defer c.Close()

	payload, err := c.SendAndReadStatus(&protocol.Request{Type: protocol.ReqStatus})
	if err != nil {
		return nil, err
	}
	if payload.Model != nil {
		return payload.Model, nil
	}
	return &protocol.ModelStatus{ModelPath: w.path}, nil
}
