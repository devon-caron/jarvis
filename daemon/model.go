package daemon

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

// ErrNoModel is returned when a chat is attempted with no model loaded.
var ErrNoModel = errors.New("no model loaded")

// LoadOpts holds llama-server tuning parameters for model loading.
type LoadOpts struct {
	ContextSize    int
	SplitMode      string
	Parallel       int
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

// ModelSlot is a per-model container that manages lifecycle and inactivity timeout.
type ModelSlot struct {
	mu        sync.RWMutex
	name      string
	backend   ModelBackend
	gpus      []int
	timeout   time.Duration
	lastUsed  time.Time
	timer     *time.Timer
	onExpire  func(name string)
	historyMu sync.Mutex
	history   map[int][]protocol.ChatMessage // keyed by shell PID
}

// newModelSlot creates a new slot. If timeout > 0, starts an inactivity timer.
func newModelSlot(name string, backend ModelBackend, gpus []int, timeout time.Duration, onExpire func(string)) *ModelSlot {
	s := &ModelSlot{
		name:     name,
		backend:  backend,
		gpus:     gpus,
		timeout:  timeout,
		lastUsed: time.Now(),
		onExpire: onExpire,
		history:  make(map[int][]protocol.ChatMessage),
	}
	if timeout > 0 {
		s.timer = time.AfterFunc(timeout, func() {
			if s.onExpire != nil {
				s.onExpire(s.name)
			}
		})
	}
	return s
}

// Chat runs a chat request on this slot. Resets the inactivity timer.
// shellPID identifies the calling shell for per-shell history tracking.
// If clearContext is true, the shell's history is cleared before this chat.
func (s *ModelSlot) Chat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string), shellPID int, clearContext bool) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.backend.IsLoaded() {
		return ErrNoModel
	}

	s.lastUsed = time.Now()
	if s.timer != nil {
		s.timer.Reset(s.timeout)
	}

	// Split incoming messages into system-role prefix and conversation tail.
	var systemMsgs, conversationMsgs []protocol.ChatMessage
	for i, m := range msgs {
		if m.Role != "system" {
			conversationMsgs = msgs[i:]
			break
		}
		systemMsgs = append(systemMsgs, m)
	}
	if len(conversationMsgs) == 0 && len(systemMsgs) == len(msgs) {
		// All messages are system messages (unusual but handle gracefully).
		conversationMsgs = nil
	}

	// Manage per-shell history.
	s.historyMu.Lock()
	if clearContext {
		delete(s.history, shellPID)
	}
	var priorHistory []protocol.ChatMessage
	if shellPID != 0 {
		priorHistory = make([]protocol.ChatMessage, len(s.history[shellPID]))
		copy(priorHistory, s.history[shellPID])
	}
	s.historyMu.Unlock()

	// Build full message list: system + history + new conversation.
	full := make([]protocol.ChatMessage, 0, len(systemMsgs)+len(priorHistory)+len(conversationMsgs))
	full = append(full, systemMsgs...)
	full = append(full, priorHistory...)
	full = append(full, conversationMsgs...)

	// Capture the full response text.
	var responseBuilder strings.Builder
	wrappedDelta := func(token string) {
		responseBuilder.WriteString(token)
		onDelta(token)
	}

	err := s.backend.RunChat(ctx, full, opts, wrappedDelta)
	if err != nil {
		// If context length exceeded with history, retry without history.
		if IsContextLengthError(err.Error()) && len(priorHistory) > 0 {
			s.historyMu.Lock()
			delete(s.history, shellPID)
			s.historyMu.Unlock()

			onDelta("\n[context limit reached — history cleared, retrying]\n")

			responseBuilder.Reset()
			retryMsgs := make([]protocol.ChatMessage, 0, len(systemMsgs)+len(conversationMsgs))
			retryMsgs = append(retryMsgs, systemMsgs...)
			retryMsgs = append(retryMsgs, conversationMsgs...)

			err = s.backend.RunChat(ctx, retryMsgs, opts, wrappedDelta)
			if err != nil {
				return err
			}
		} else {
			return err
		}
	}

	// On success, append conversation + assistant response to history.
	if shellPID != 0 {
		s.historyMu.Lock()
		s.history[shellPID] = append(s.history[shellPID], conversationMsgs...)
		s.history[shellPID] = append(s.history[shellPID], protocol.ChatMessage{
			Role:    "assistant",
			Content: responseBuilder.String(),
		})
		s.historyMu.Unlock()
	}

	return nil
}

// GetHistory returns a copy of the conversation history for the given shell PID.
func (s *ModelSlot) GetHistory(shellPID int) []protocol.ChatMessage {
	s.historyMu.Lock()
	defer s.historyMu.Unlock()

	hist := s.history[shellPID]
	cp := make([]protocol.ChatMessage, len(hist))
	copy(cp, hist)
	return cp
}

// Unload stops the timer and unloads the model.
func (s *ModelSlot) Unload() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	return s.backend.UnloadModel()
}

// Status returns the slot's status info.
func (s *ModelSlot) Status() protocol.SlotInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	info := protocol.SlotInfo{
		Name:      s.name,
		ModelPath: s.backend.ModelPath(),
		GPUs:      s.gpus,
		LastUsed:  s.lastUsed,
	}
	if s.timeout > 0 {
		info.Timeout = s.timeout.String()
	}
	if status, err := s.backend.GetStatus(); err == nil && status != nil {
		info.GPUInfo = status.GPUs
	}
	return info
}

// ModelRegistry manages a collection of model slots with GPU conflict detection.
type ModelRegistry struct {
	mu         sync.RWMutex
	slots      map[string]*ModelSlot // keyed by model name
	gpuInUse   map[int]string        // gpu_id → model name
	loading    map[string]bool       // models currently being loaded
	newBackend func(*config.Config) ModelBackend
	cfg        *config.Config
}

// NewModelRegistry creates a new registry.
func NewModelRegistry(cfg *config.Config, newBackend func(*config.Config) ModelBackend) *ModelRegistry {
	return &ModelRegistry{
		slots:      make(map[string]*ModelSlot),
		gpuInUse:   make(map[int]string),
		loading:    make(map[string]bool),
		newBackend: newBackend,
		cfg:        cfg,
	}
}

// Load loads a model onto the specified GPUs with an optional inactivity timeout.
// The mutex is released during the actual model load so that other daemon
// operations (status, chat, unload) are not blocked.
func (r *ModelRegistry) Load(ctx context.Context, name, path string, gpus []int, timeout time.Duration, opts LoadOpts) error {
	// Phase 1: Reserve GPUs and name under lock.
	r.mu.Lock()
	for _, gpu := range gpus {
		if owner, ok := r.gpuInUse[gpu]; ok {
			r.mu.Unlock()
			return fmt.Errorf("GPU %d already in use by model %q", gpu, owner)
		}
	}
	if _, ok := r.slots[name]; ok {
		r.mu.Unlock()
		return fmt.Errorf("model %q is already loaded; unload it first", name)
	}
	if r.loading[name] {
		r.mu.Unlock()
		return fmt.Errorf("model %q is already being loaded", name)
	}
	for _, gpu := range gpus {
		r.gpuInUse[gpu] = name
	}
	r.loading[name] = true
	r.mu.Unlock()

	// Phase 2: Load model (no lock held — other operations proceed freely).
	backend := r.newBackend(r.cfg)
	err := backend.LoadModel(ctx, path, gpus, opts)

	// Phase 3: Commit or rollback under lock.
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.loading, name)
	if err != nil {
		for _, gpu := range gpus {
			delete(r.gpuInUse, gpu)
		}
		return err
	}
	slot := newModelSlot(name, backend, gpus, timeout, r.removeExpired)
	r.slots[name] = slot
	return nil
}

// UnloadByGPU unloads whichever model is currently on the given GPU.
func (r *ModelRegistry) UnloadByGPU(gpu int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	owner, ok := r.gpuInUse[gpu]
	if !ok {
		return fmt.Errorf("no model loaded on GPU %d", gpu)
	}
	return r.unloadLocked(owner)
}

// Unload unloads a specific model by name. If name is empty and only one model
// is loaded, unloads that one.
func (r *ModelRegistry) Unload(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if name == "" {
		if len(r.slots) == 1 {
			for n := range r.slots {
				name = n
			}
		} else if len(r.slots) == 0 {
			return errors.New("no model loaded")
		} else {
			names := make([]string, 0, len(r.slots))
			for n := range r.slots {
				names = append(names, n)
			}
			return fmt.Errorf("multiple models loaded, specify which to unload: %v", names)
		}
	}

	if _, ok := r.slots[name]; !ok {
		return fmt.Errorf("model %q not loaded", name)
	}
	return r.unloadLocked(name)
}

// unloadLocked tears down the named slot. Caller must hold r.mu (write lock).
func (r *ModelRegistry) unloadLocked(name string) error {
	slot := r.slots[name]
	if err := slot.Unload(); err != nil {
		return err
	}
	for _, gpu := range slot.gpus {
		delete(r.gpuInUse, gpu)
	}
	delete(r.slots, name)
	return nil
}

// Chat routes a chat request to the appropriate model slot.
//
// Routing priority:
//  1. name != ""  → route by model name
//  2. gpu >= 0    → route to whichever model is on that GPU
//  3. single model loaded → auto-route
//  4. multiple models → route to model on cfg.DefaultGPU
func (r *ModelRegistry) Chat(ctx context.Context, name string, gpu int, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string), shellPID int, clearContext bool) error {
	r.mu.RLock()

	if len(r.slots) == 0 {
		r.mu.RUnlock()
		return ErrNoModel
	}

	var slot *ModelSlot

	if name != "" {
		s, ok := r.slots[name]
		if !ok {
			r.mu.RUnlock()
			return fmt.Errorf("model %q not loaded", name)
		}
		slot = s
	} else if gpu >= 0 {
		owner, ok := r.gpuInUse[gpu]
		if !ok {
			r.mu.RUnlock()
			return fmt.Errorf("no model loaded on GPU %d", gpu)
		}
		slot = r.slots[owner]
	} else if len(r.slots) == 1 {
		for _, s := range r.slots {
			slot = s
		}
	} else {
		defaultGPU := r.cfg.DefaultGPU
		if owner, ok := r.gpuInUse[defaultGPU]; ok {
			slot = r.slots[owner]
		} else {
			names := make([]string, 0, len(r.slots))
			for n := range r.slots {
				names = append(names, n)
			}
			r.mu.RUnlock()
			return fmt.Errorf("no model on default GPU %d, available models: %v", defaultGPU, names)
		}
	}

	r.mu.RUnlock()
	return slot.Chat(ctx, msgs, opts, onDelta, shellPID, clearContext)
}

// Status returns info for all loaded models.
func (r *ModelRegistry) Status() []protocol.SlotInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	infos := make([]protocol.SlotInfo, 0, len(r.slots))
	for _, slot := range r.slots {
		infos = append(infos, slot.Status())
	}
	return infos
}

// GetHistory returns the conversation history for the specified model and shell PID.
// If name is empty and only one model is loaded, that model is used.
func (r *ModelRegistry) GetHistory(name string, shellPID int) ([]protocol.ChatMessage, string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.slots) == 0 {
		return nil, "", ErrNoModel
	}

	var slot *ModelSlot
	var resolvedName string

	if name != "" {
		s, ok := r.slots[name]
		if !ok {
			return nil, "", fmt.Errorf("model %q not loaded", name)
		}
		slot = s
		resolvedName = name
	} else if len(r.slots) == 1 {
		for n, s := range r.slots {
			slot = s
			resolvedName = n
		}
	} else {
		names := make([]string, 0, len(r.slots))
		for n := range r.slots {
			names = append(names, n)
		}
		return nil, "", fmt.Errorf("multiple models loaded, specify which: %v", names)
	}

	msgs := slot.GetHistory(shellPID)
	return msgs, resolvedName, nil
}

// Shutdown unloads all models.
func (r *ModelRegistry) Shutdown() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, slot := range r.slots {
		slot.Unload()
		delete(r.slots, name)
	}
	for gpu := range r.gpuInUse {
		delete(r.gpuInUse, gpu)
	}
}

// removeExpired is the callback for timer expiry. It acquires the write lock and cleans up.
func (r *ModelRegistry) removeExpired(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	slot, ok := r.slots[name]
	if !ok {
		return
	}
	slot.Unload()
	for _, gpu := range slot.gpus {
		delete(r.gpuInUse, gpu)
	}
	delete(r.slots, name)
}
