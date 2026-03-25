package daemon

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

// ModelRegister manages loading and unloading a single model at a time.
type ModelRegister struct {
	mu         sync.RWMutex
	loaded     bool
	name       string
	history    map[int][]protocol.ChatMessage
	historyMu  sync.Mutex
	path       string
	gpus       []int
	backend    ModelBackend
	newBackend func(*config.Config) ModelBackend
	cfg        *config.Config
}

// NewModelRegister creates a new register instance.
func NewModelRegister(cfg *config.Config, newBackend func(*config.Config) ModelBackend) *ModelRegister {
	return &ModelRegister{
		newBackend: newBackend,
		cfg:        cfg,
		history:    make(map[int][]protocol.ChatMessage),
	}
}

func (r *ModelRegister) Chat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onToken func(string), shellPID int, clearContext bool, isPty bool) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	log.Printf("Chat called with %d messages, opts: %+v, shellPID: %d, clearContext: %v, isPty: %v", len(msgs), opts, shellPID, clearContext, isPty)

	if !r.loaded || r.backend == nil {
		return errors.New("no model loaded")
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
		conversationMsgs = []protocol.ChatMessage{}
	}

	// with a pty terminal, prior context is provided via a system message appended to existing ones.
	// conversationMsgs will have the user prompt again
	if isPty {
		// remove the last system message containing prior pty terminal context
		if clearContext {
			systemMsgs = systemMsgs[:len(systemMsgs)-1]
		}
		systemMsgs = append(systemMsgs, protocol.ChatMessage{
			Role: "system",
			Content: "Please interact with the user as if your conversation history with the user is through jarvis commands + other terminal commands provided in your context." +
				" If there looks to be no context, ask the user what they need help with.",
		})
	}

	log.Printf("System messages: %d, conversation messages: %d", len(systemMsgs), len(conversationMsgs))

	var fullHistory, priorHistory []protocol.ChatMessage

	// Build history: only prepend system messages when starting fresh (no prior history or clearContext).
	r.historyMu.Lock()
	if clearContext {
		delete(r.history, shellPID)
		priorHistory = systemMsgs
	} else if len(r.history[shellPID]) == 0 {
		priorHistory = systemMsgs
	} else {
		priorHistory = r.history[shellPID]
	}

	log.Printf("History for shell %d: %d messages", shellPID, len(r.history[shellPID]))

	fullHistory = append(fullHistory, priorHistory...)
	fullHistory = append(fullHistory, conversationMsgs...)

	r.history[shellPID] = fullHistory

	r.historyMu.Unlock()

	var responseBuilder strings.Builder
	tokenWriterWrapper := func(token string) {
		responseBuilder.WriteString(token)
		onToken(token)
	}

	log.Printf("Running chat with %d messages", len(fullHistory))
	err := r.backend.RunChat(ctx, fullHistory, opts, tokenWriterWrapper)
	if err != nil {
		log.Printf("chat failed: %v", err)
		return fmt.Errorf("chat failed: %w", err)
	}

	r.historyMu.Lock()
	r.history[shellPID] = append(r.history[shellPID], protocol.ChatMessage{
		Role:    "assistant",
		Content: responseBuilder.String(),
	})
	r.historyMu.Unlock()

	log.Printf("Chat completed, response: %s", responseBuilder.String())
	return nil
}

func (r *ModelRegister) Load(ctx context.Context, name, path string, gpus []int, timeout time.Duration, opts LoadOpts) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.loaded {
		return errors.New("a model is already loaded; unload it first")
	}

	backend := r.newBackend(r.cfg)
	if err := backend.LoadModel(ctx, path, gpus, opts); err != nil {
		return fmt.Errorf("backend failed to load model: %w", err)
	}

	r.backend = backend
	r.name = name
	r.path = path
	r.gpus = gpus
	r.loaded = true

	return nil
}

// Unload unloads the currently loaded model.
func (r *ModelRegister) Unload(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.loaded {
		return errors.New("no model loaded")
	}

	if name != "" && name != r.name {
		return fmt.Errorf("model %q is not loaded (loaded: %q)", name, r.name)
	}

	if err := r.backend.UnloadModel(); err != nil {
		return err
	}

	r.backend = nil
	r.name = ""
	r.path = ""
	r.gpus = nil
	r.loaded = false

	return nil
}

// Status returns info about the currently loaded model, if any.
func (r *ModelRegister) Status() *protocol.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.loaded || r.backend == nil {
		return nil
	}

	info := &protocol.ModelInfo{
		Name:      r.name,
		ModelPath: r.backend.ModelPath(),
		GPUs:      r.gpus,
	}

	return info
}

// IsLoaded returns whether a model is currently loaded.
func (r *ModelRegister) IsLoaded() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.loaded
}

func (r *ModelRegister) Shutdown() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.loaded && r.backend != nil {
		r.backend.UnloadModel()
		r.backend = nil
		r.name = ""
		r.path = ""
		r.gpus = nil
		r.loaded = false
	}
}
