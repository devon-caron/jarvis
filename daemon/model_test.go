package daemon

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

// mockBackend implements ModelBackend for testing.
type mockBackend struct {
	loaded     bool
	path       string
	loadedGPUs []int
	loadErr    error
	loadDelay  time.Duration
	unloadErr  error
	chatErr    error
	chatDelay  time.Duration
	chatCalls  atomic.Int32
	chatFunc   func(msgs []protocol.ChatMessage) // optional hook to capture messages
}

func (m *mockBackend) LoadModel(ctx context.Context, path string, gpus []int, opts LoadOpts) error {
	if m.loadDelay > 0 {
		select {
		case <-ctx.Done():
			return fmt.Errorf("load cancelled")
		case <-time.After(m.loadDelay):
		}
	}
	if m.loadErr != nil {
		return m.loadErr
	}
	m.loaded = true
	m.path = path
	m.loadedGPUs = gpus
	return nil
}

func (m *mockBackend) UnloadModel() error {
	if m.unloadErr != nil {
		return m.unloadErr
	}
	m.loaded = false
	m.path = ""
	return nil
}

func (m *mockBackend) IsLoaded() bool {
	return m.loaded
}

func (m *mockBackend) ModelPath() string {
	return m.path
}

func (m *mockBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	m.chatCalls.Add(1)
	if m.chatFunc != nil {
		m.chatFunc(msgs)
	}
	if m.chatDelay > 0 {
		time.Sleep(m.chatDelay)
	}
	if m.chatErr != nil {
		return m.chatErr
	}
	onDelta("Hello ")
	onDelta("world!")
	return nil
}

// newMockBackendFactory returns a factory that returns mockBackends.
func newMockBackendFactory(backends ...*mockBackend) func(*config.Config) ModelBackend {
	idx := 0
	return func(cfg *config.Config) ModelBackend {
		if idx < len(backends) {
			b := backends[idx]
			idx++
			return b
		}
		return &mockBackend{}
	}
}

func newTestRegister(t *testing.T, backends ...*mockBackend) *ModelRegister {
	t.Helper()
	cfg := config.Defaults()
	factory := newMockBackendFactory(backends...)
	return NewModelRegister(cfg, factory)
}

// --- ModelRegistry tests ---

// TestModelRegister_LoadUnload verifies the full load/unload lifecycle: loads a model
// by name, checks Status() returns the correct ModelInfo, unloads by name, and
// confirms Status() returns nil afterward.
func TestModelRegister_LoadUnload(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegister(t, backend)

	if err := reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load: %v", err)
	}

	model := reg.Status()
	if model == nil {
		t.Fatal("expected model info, got nil")
	}
	if model.Name != "test" {
		t.Errorf("Name = %q, want test", model.Name)
	}

	if err := reg.Unload("test"); err != nil {
		t.Fatalf("Unload: %v", err)
	}

	if reg.Status() != nil {
		t.Error("expected nil status after unload")
	}
}

// TestModelRegister_Load_WhenAlreadyLoaded_Errors verifies that attempting to load
// a second model while one is already loaded returns an error, enforcing the
// single-model constraint.
func TestModelRegister_Load_WhenAlreadyLoaded_Errors(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegister(t, b1)

	reg.Load(context.Background(), "test", "/model1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load(context.Background(), "test2", "/model2.gguf", []int{1}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected error when a model is already loaded")
	}
}

// TestModelRegister_LoadError verifies that when the backend returns a load error,
// the registry remains unloaded (Status() returns nil), and a fresh registry
// can still load successfully afterward.
func TestModelRegister_LoadError(t *testing.T) {
	backend := &mockBackend{loadErr: errors.New("load failed")}
	reg := newTestRegister(t, backend)

	err := reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Error("expected load error")
	}

	if reg.Status() != nil {
		t.Error("no model should be loaded after error")
	}

	// Should be able to load again after failure
	backend2 := &mockBackend{}
	reg2 := newTestRegister(t, backend2)
	if err := reg2.Load(context.Background(), "test2", "/model2.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("should be able to load after prior failure: %v", err)
	}
}

// TestModelRegister_UnloadWhenEmpty verifies that calling Unload on an empty
// registry (no model loaded) returns an error.
func TestModelRegister_UnloadWhenEmpty(t *testing.T) {
	reg := newTestRegister(t)

	err := reg.Unload("")
	if err == nil {
		t.Error("expected error when unloading with no model")
	}
}

// TestModelRegister_UnloadEmptyName verifies that calling Unload with an empty name
// successfully unloads the currently loaded model.
func TestModelRegister_UnloadEmptyName(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	// Unload without name should unload the loaded model
	if err := reg.Unload(""); err != nil {
		t.Fatalf("Unload: %v", err)
	}

	if reg.Status() != nil {
		t.Error("expected nil status after unload")
	}
}

// TestModelRegister_UnloadWrongName verifies that calling Unload with a name that
// doesn't match the loaded model returns an error.
func TestModelRegister_UnloadWrongName(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Unload("nonexistent")
	if err == nil {
		t.Error("expected error for wrong model name")
	}
}

// TestModelRegister_UnloadNotLoaded verifies that calling Unload with a specific
// name when no model is loaded returns an error.
func TestModelRegister_UnloadNotLoaded(t *testing.T) {
	reg := newTestRegister(t)

	err := reg.Unload("nonexistent")
	if err == nil {
		t.Error("expected error when nothing is loaded")
	}
}

// TestModelRegister_Chat verifies that Chat delegates to the backend's RunChat,
// collecting streamed tokens ("Hello ", "world!") via the onToken callback.
func TestModelRegister_Chat(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var tokens []string
	err := reg.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(token string) { tokens = append(tokens, token) },
		0, false,
		false)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(tokens) != 2 || tokens[0] != "Hello " || tokens[1] != "world!" {
		t.Errorf("tokens = %v", tokens)
	}
}

// TestModelRegister_Chat_NoModelLoaded verifies that Chat returns an error when
// no model is loaded.
func TestModelRegister_Chat_NoModelLoaded(t *testing.T) {
	reg := newTestRegister(t)

	err := reg.Chat(context.Background(),
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
		false)
	if err == nil {
		t.Error("expected error when no model loaded")
	}
}

// TestModelRegister_Chat_ChatError verifies that Chat propagates errors returned
// by the backend's RunChat method.
func TestModelRegister_Chat_ChatError(t *testing.T) {
	backend := &mockBackend{chatErr: errors.New("chat failed")}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(),
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
		false)
	if err == nil || !strings.Contains(err.Error(), "chat failed") {
		t.Errorf("expected chat error, got: %v", err)
	}
}

// TestModelRegister_ConcurrentChat verifies that 10 concurrent Chat calls all
// complete successfully and the backend receives exactly 10 RunChat calls,
// confirming thread safety under the RWMutex.
func TestModelRegister_ConcurrentChat(t *testing.T) {
	backend := &mockBackend{chatDelay: 10 * time.Millisecond}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reg.Chat(context.Background(),
				[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
				protocol.InferenceOpts{},
				func(string) {},
				0, false,
				false)
		}()
	}
	wg.Wait()

	if got := backend.chatCalls.Load(); got != 10 {
		t.Errorf("chatCalls = %d, want 10", got)
	}
}

// TestModelRegister_Status_Empty verifies Status() returns nil when no model
// is loaded.
func TestModelRegister_Status_Empty(t *testing.T) {
	reg := newTestRegister(t)

	if reg.Status() != nil {
		t.Error("expected nil status when no model loaded")
	}
}

// TestModelRegister_Status_WithModel verifies Status() returns correct ModelInfo
// (name and model path) after loading a model.
func TestModelRegister_Status_WithModel(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	model := reg.Status()
	if model == nil {
		t.Fatal("expected model info, got nil")
	}
	if model.Name != "test" {
		t.Errorf("Name = %q, want test", model.Name)
	}
	if model.ModelPath != "/model.gguf" {
		t.Errorf("ModelPath = %q, want /model.gguf", model.ModelPath)
	}
}

// TestModelRegister_IsLoaded verifies the IsLoaded() getter returns false initially,
// true after Load, and false again after Unload.
func TestModelRegister_IsLoaded(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegister(t, backend)

	if reg.IsLoaded() {
		t.Error("should not be loaded initially")
	}

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	if !reg.IsLoaded() {
		t.Error("should be loaded after Load")
	}

	reg.Unload("")

	if reg.IsLoaded() {
		t.Error("should not be loaded after Unload")
	}
}

// TestModelRegister_Shutdown verifies Shutdown unloads the backend, clears Status()
// to nil, and sets IsLoaded() to false.
func TestModelRegister_Shutdown(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	reg.Shutdown()

	if reg.Status() != nil {
		t.Error("expected nil status after shutdown")
	}
	if reg.IsLoaded() {
		t.Error("should not be loaded after shutdown")
	}
	if backend.loaded {
		t.Error("backend should be unloaded after shutdown")
	}
}

// TestModelRegister_UnloadError verifies that Unload propagates errors returned
// by the backend's UnloadModel method.
func TestModelRegister_UnloadError(t *testing.T) {
	backend := &mockBackend{unloadErr: errors.New("unload failed")}
	reg := newTestRegister(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Unload("test")
	if err == nil {
		t.Error("expected unload error")
	}
}

// TestModelRegister_Load_ContextCancelled verifies that cancelling the context
// during a slow Load causes it to return an error, and the registry remains
// in an unloaded state afterward.
func TestModelRegister_Load_ContextCancelled(t *testing.T) {
	backend := &mockBackend{loadDelay: 5 * time.Second}
	reg := newTestRegister(t, backend)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	err := reg.Load(ctx, "test", "/model.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	if !errors.Is(ctx.Err(), context.Canceled) {
		t.Errorf("context should be cancelled, got: %v", ctx.Err())
	}

	if reg.Status() != nil {
		t.Error("expected nil status after cancelled load")
	}
}

// TestModelRegister_Load_FailureAllowsReload verifies that after a failed Load
// (backend error), the registry is not stuck in a "loaded" state and a
// subsequent Load on the same registry succeeds.
func TestModelRegister_Load_FailureAllowsReload(t *testing.T) {
	backend := &mockBackend{loadErr: fmt.Errorf("simulated failure")}
	backend2 := &mockBackend{}
	reg := newTestRegister(t, backend, backend2)

	err := reg.Load(context.Background(), "test", "/model.gguf", []int{0, 1}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected load error")
	}

	if reg.Status() != nil {
		t.Error("expected nil status after failed load")
	}

	// Should be able to load again after failure
	if err := reg.Load(context.Background(), "test2", "/model2.gguf", []int{0, 1}, 0, LoadOpts{}); err != nil {
		t.Fatalf("should be able to load after failed load: %v", err)
	}
}

// TestModelRegister_LoadAfterUnload verifies the full cycle of load -> unload ->
// load with a different model, confirming the second model is correctly
// reflected in Status().
func TestModelRegister_LoadAfterUnload(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegister(t, b1, b2)

	if err := reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load m1: %v", err)
	}
	if err := reg.Unload("m1"); err != nil {
		t.Fatalf("Unload m1: %v", err)
	}
	if err := reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load m2 after unload: %v", err)
	}

	model := reg.Status()
	if model == nil || model.Name != "m2" {
		t.Errorf("expected m2 to be loaded, got %v", model)
	}
}

// --- Continuous context tests ---

// helper: loads a model on a register and returns the backend and a chatFunc capture slice.
func loadWithCapture(t *testing.T) (*ModelRegister, *mockBackend, *[][]protocol.ChatMessage) {
	t.Helper()
	var captured [][]protocol.ChatMessage
	backend := &mockBackend{}
	backend.chatFunc = func(msgs []protocol.ChatMessage) {
		// Deep-copy so later mutations don't affect our snapshot.
		cp := make([]protocol.ChatMessage, len(msgs))
		copy(cp, msgs)
		captured = append(captured, cp)
	}
	reg := newTestRegister(t, backend)
	if err := reg.Load(context.Background(), "test", "/m.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load: %v", err)
	}
	return reg, backend, &captured
}

func chatMsg(role, content string) protocol.ChatMessage {
	return protocol.ChatMessage{Role: role, Content: content}
}

// assertRoles is a test helper that checks the role sequence of a message slice.
func assertRoles(t *testing.T, label string, msgs []protocol.ChatMessage, wantRoles ...string) {
	t.Helper()
	if len(msgs) != len(wantRoles) {
		t.Fatalf("%s: got %d messages, want %d\n  got:  %v", label, len(msgs), len(wantRoles), msgsToRoles(msgs))
	}
	for i, want := range wantRoles {
		if msgs[i].Role != want {
			t.Errorf("%s: msgs[%d].Role = %q, want %q\n  got:  %v", label, i, msgs[i].Role, want, msgsToRoles(msgs))
			return
		}
	}
}

func msgsToRoles(msgs []protocol.ChatMessage) []string {
	roles := make([]string, len(msgs))
	for i, m := range msgs {
		roles[i] = m.Role
	}
	return roles
}

// TestChat_HistoryAccumulation verifies that successive Chat calls on the same
// shellPID accumulate into a single conversation history, with the system
// message appearing only at the front and user/assistant turns appended.
func TestChat_HistoryAccumulation(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 42

	// Turn 1: system + user → backend sees [system, user]
	err := reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "Be helpful"), chatMsg("user", "hello")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}

	assertRoles(t, "turn 1 backend", (*captured)[0], "system", "user")

	// Turn 2: same system + new user → backend should see full history
	err = reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "Be helpful"), chatMsg("user", "how are you")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	if err != nil {
		t.Fatalf("turn 2: %v", err)
	}

	turn2Msgs := (*captured)[1]
	assertRoles(t, "turn 2 backend", turn2Msgs, "system", "user", "assistant", "user")

	if turn2Msgs[0].Content != "Be helpful" {
		t.Errorf("system msg content = %q, want 'Be helpful'", turn2Msgs[0].Content)
	}
	if turn2Msgs[1].Content != "hello" {
		t.Errorf("first user msg = %q, want 'hello'", turn2Msgs[1].Content)
	}
	if turn2Msgs[2].Content != "Hello world!" {
		t.Errorf("assistant msg = %q, want 'Hello world!'", turn2Msgs[2].Content)
	}
	if turn2Msgs[3].Content != "how are you" {
		t.Errorf("second user msg = %q, want 'how are you'", turn2Msgs[3].Content)
	}
}

// TestChat_SystemMsgsNotDuplicated verifies that system messages from the
// request are NOT re-injected into an existing history, which would break
// the user/assistant alternation required by chat templates.
func TestChat_SystemMsgsNotDuplicated(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 1

	// Turn 1
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "q1")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	// Turn 2
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "q2")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	// Turn 3
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "q3")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)

	turn3 := (*captured)[2]
	// Should be: system, user, assistant, user, assistant, user
	// NOT: system, user, assistant, system, user, assistant, system, user
	assertRoles(t, "turn 3", turn3, "system", "user", "assistant", "user", "assistant", "user")

	// Count system messages — should be exactly 1.
	systemCount := 0
	for _, m := range turn3 {
		if m.Role == "system" {
			systemCount++
		}
	}
	if systemCount != 1 {
		t.Errorf("expected 1 system message, got %d in: %v", systemCount, msgsToRoles(turn3))
	}
}

// TestChat_ClearContextResetsHistory verifies that clearContext=true wipes the
// conversation history for that shellPID and re-applies the system prompt,
// starting a fresh conversation.
func TestChat_ClearContextResetsHistory(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 10

	// Build up some history.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "q1")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "q2")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)

	// Now clear context.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "new sys"), chatMsg("user", "fresh start")},
		protocol.InferenceOpts{}, func(string) {}, pid, true,
		false)

	turn3 := (*captured)[2]
	// Should be a fresh conversation: just [system, user], no prior history.
	assertRoles(t, "after clear", turn3, "system", "user")
	if turn3[0].Content != "new sys" {
		t.Errorf("system content = %q, want 'new sys'", turn3[0].Content)
	}
	if turn3[1].Content != "fresh start" {
		t.Errorf("user content = %q, want 'fresh start'", turn3[1].Content)
	}
}

// TestChat_ClearContextThenContinue verifies that after a clearContext call,
// subsequent calls accumulate history normally from the new starting point.
func TestChat_ClearContextThenContinue(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 20

	// Build history then clear.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "old"), chatMsg("user", "old q")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "new"), chatMsg("user", "reset")},
		protocol.InferenceOpts{}, func(string) {}, pid, true,
		false)

	// Continue after clear.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "new"), chatMsg("user", "follow up")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)

	turn3 := (*captured)[2]
	assertRoles(t, "post-clear continuation", turn3, "system", "user", "assistant", "user")
	if turn3[0].Content != "new" {
		t.Errorf("system = %q, want 'new' (not 'old')", turn3[0].Content)
	}
	if turn3[1].Content != "reset" {
		t.Errorf("first user = %q, want 'reset'", turn3[1].Content)
	}
	if turn3[3].Content != "follow up" {
		t.Errorf("second user = %q, want 'follow up'", turn3[3].Content)
	}
}

// TestChat_ShellPIDIsolation verifies that different shellPIDs maintain
// completely independent conversation histories.
func TestChat_ShellPIDIsolation(t *testing.T) {
	reg, _, captured := loadWithCapture(t)

	// Shell A: two turns.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "A-sys"), chatMsg("user", "A1")},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
		false)
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "A-sys"), chatMsg("user", "A2")},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
		false)

	// Shell B: one turn — should NOT see shell A's history.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "B-sys"), chatMsg("user", "B1")},
		protocol.InferenceOpts{}, func(string) {}, 200, false,
		false)

	shellBMsgs := (*captured)[2]
	assertRoles(t, "shell B", shellBMsgs, "system", "user")
	if shellBMsgs[0].Content != "B-sys" {
		t.Errorf("shell B system = %q, want 'B-sys'", shellBMsgs[0].Content)
	}
	if shellBMsgs[1].Content != "B1" {
		t.Errorf("shell B user = %q, want 'B1'", shellBMsgs[1].Content)
	}

	// Verify shell A still has its full history.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "A-sys"), chatMsg("user", "A3")},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
		false)

	shellA3 := (*captured)[3]
	assertRoles(t, "shell A turn 3", shellA3, "system", "user", "assistant", "user", "assistant", "user")
}

// TestChat_ClearContextOnlyAffectsTargetShell verifies that clearContext on
// one shellPID does not disturb another shell's history.
func TestChat_ClearContextOnlyAffectsTargetShell(t *testing.T) {
	reg, _, captured := loadWithCapture(t)

	// Build history on both shells.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "A1")},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
		false)
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "B1")},
		protocol.InferenceOpts{}, func(string) {}, 200, false,
		false)

	// Clear shell 100 only.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "A-fresh")},
		protocol.InferenceOpts{}, func(string) {}, 100, true,
		false)

	shellAClear := (*captured)[2]
	assertRoles(t, "shell A after clear", shellAClear, "system", "user")

	// Shell 200 should still have its history.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "B2")},
		protocol.InferenceOpts{}, func(string) {}, 200, false,
		false)

	shellB2 := (*captured)[3]
	assertRoles(t, "shell B unaffected", shellB2, "system", "user", "assistant", "user")
}

// TestChat_NoSystemMessages verifies that Chat works correctly when no system
// prompt is provided — the conversation should just be user/assistant turns.
func TestChat_NoSystemMessages(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 5

	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("user", "q1")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("user", "q2")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)

	turn1 := (*captured)[0]
	assertRoles(t, "turn 1 no sys", turn1, "user")

	turn2 := (*captured)[1]
	assertRoles(t, "turn 2 no sys", turn2, "user", "assistant", "user")
}

// TestChat_AssistantResponseRecorded verifies the assistant's streamed response
// is captured and stored in history for subsequent turns.
func TestChat_AssistantResponseRecorded(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 7

	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("user", "hi")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)

	// Second call — the backend should see the previous assistant response.
	reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("user", "thanks")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)

	turn2 := (*captured)[1]
	if turn2[1].Role != "assistant" || turn2[1].Content != "Hello world!" {
		t.Errorf("expected assistant 'Hello world!' in history, got role=%q content=%q",
			turn2[1].Role, turn2[1].Content)
	}
}

// TestChat_BackendErrorDoesNotAppendAssistant verifies that when RunChat fails,
// the assistant response is NOT appended to history, but the user message IS
// present so it can be retried.
func TestChat_BackendErrorDoesNotAppendAssistant(t *testing.T) {
	backend := &mockBackend{chatErr: errors.New("inference failed")}
	reg := newTestRegister(t, backend)
	reg.Load(context.Background(), "test", "/m.gguf", []int{0}, 0, LoadOpts{})
	pid := 9

	// First call fails.
	err := reg.Chat(context.Background(),
		[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", "q1")},
		protocol.InferenceOpts{}, func(string) {}, pid, false,
		false)
	if err == nil {
		t.Fatal("expected error")
	}

	// Inspect history directly — should have [system, user] but no assistant.
	reg.historyMu.Lock()
	hist := reg.history[pid]
	reg.historyMu.Unlock()

	assertRoles(t, "after error", hist, "system", "user")
}

// --- PTY clear-context tests ---

// TestChat_PtyClearContext_RemovesTerminalContext verifies that when isPty=true
// and clearContext=true, the last system message (terminal context) is popped
// before the PTY instruction is appended, ensuring stale terminal context does
// not carry into a fresh conversation.
func TestChat_PtyClearContext_RemovesTerminalContext(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 50

	sysPrompt := chatMsg("system", "You are helpful")
	termCtx := chatMsg("system", "Recent terminal output (for context):\n```\n$ ls\nfoo bar\n```")

	// Turn 1: PTY session without clear — terminal context should be preserved.
	err := reg.Chat(context.Background(),
		[]protocol.ChatMessage{sysPrompt, termCtx, chatMsg("user", "what files do I have?")},
		protocol.InferenceOpts{}, func(string) {}, pid, false, true)
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}

	turn1 := (*captured)[0]
	// Expect: [system(prompt), system(termCtx), system(pty_instruction), user]
	assertRoles(t, "turn 1 (no clear)", turn1, "system", "system", "system", "user")
	if turn1[1].Content != termCtx.Content {
		t.Errorf("turn 1: terminal context should be preserved, got %q", turn1[1].Content)
	}
	if !strings.Contains(turn1[2].Content, "jarvis commands") {
		t.Errorf("turn 1: expected PTY instruction, got %q", turn1[2].Content)
	}

	// Turn 2: clearContext=true with isPty=true — terminal context should be removed.
	newTermCtx := chatMsg("system", "Recent terminal output (for context):\n```\n$ pwd\n/home/user\n```")
	err = reg.Chat(context.Background(),
		[]protocol.ChatMessage{sysPrompt, newTermCtx, chatMsg("user", "start fresh")},
		protocol.InferenceOpts{}, func(string) {}, pid, true, true)
	if err != nil {
		t.Fatalf("turn 2 (clear): %v", err)
	}

	turn2 := (*captured)[1]
	// After pop: terminal context removed, PTY instruction appended.
	// Expect: [system(prompt), system(pty_instruction), user]
	assertRoles(t, "turn 2 (clear)", turn2, "system", "system", "user")

	if turn2[0].Content != "You are helpful" {
		t.Errorf("turn 2: first system should be prompt, got %q", turn2[0].Content)
	}
	if strings.Contains(turn2[1].Content, "terminal output") {
		t.Errorf("turn 2: terminal context should have been removed, got %q", turn2[1].Content)
	}
	if !strings.Contains(turn2[1].Content, "jarvis commands") {
		t.Errorf("turn 2: expected PTY instruction, got %q", turn2[1].Content)
	}
}

// TestChat_PtyClearContext_SingleSystemMessage verifies the edge case where
// the only system message is the terminal context (no system prompt).
// Popping it should leave just the PTY instruction.
func TestChat_PtyClearContext_SingleSystemMessage(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 51

	termCtx := chatMsg("system", "Recent terminal output (for context):\n```\n$ echo hi\nhi\n```")

	err := reg.Chat(context.Background(),
		[]protocol.ChatMessage{termCtx, chatMsg("user", "clear please")},
		protocol.InferenceOpts{}, func(string) {}, pid, true, true)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	msgs := (*captured)[0]
	// After pop: terminal context removed, only PTY instruction remains.
	// Expect: [system(pty_instruction), user]
	assertRoles(t, "single sys clear", msgs, "system", "user")

	if strings.Contains(msgs[0].Content, "terminal output") {
		t.Errorf("terminal context should have been removed, got %q", msgs[0].Content)
	}
	if !strings.Contains(msgs[0].Content, "jarvis commands") {
		t.Errorf("expected PTY instruction, got %q", msgs[0].Content)
	}
}

// TestChat_PtyWithoutClear_PreservesTerminalContext verifies that when
// isPty=true but clearContext=false, the terminal context system message
// is preserved in what reaches the backend.
func TestChat_PtyWithoutClear_PreservesTerminalContext(t *testing.T) {
	reg, _, captured := loadWithCapture(t)
	pid := 52

	sysPrompt := chatMsg("system", "You are helpful")
	termCtx := chatMsg("system", "Recent terminal output (for context):\n```\n$ whoami\ndevon\n```")

	err := reg.Chat(context.Background(),
		[]protocol.ChatMessage{sysPrompt, termCtx, chatMsg("user", "who am I?")},
		protocol.InferenceOpts{}, func(string) {}, pid, false, true)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	msgs := (*captured)[0]
	// No clear — terminal context should be kept.
	// Expect: [system(prompt), system(termCtx), system(pty_instruction), user]
	assertRoles(t, "pty no clear", msgs, "system", "system", "system", "user")

	if !strings.Contains(msgs[1].Content, "whoami") {
		t.Errorf("terminal context should be preserved, got %q", msgs[1].Content)
	}
	if !strings.Contains(msgs[2].Content, "jarvis commands") {
		t.Errorf("expected PTY instruction as third system msg, got %q", msgs[2].Content)
	}
}

// TestChat_ConcurrentDifferentShells verifies that concurrent Chat calls on
// different shellPIDs do not interfere with each other's histories.
func TestChat_ConcurrentDifferentShells(t *testing.T) {
	backend := &mockBackend{chatDelay: 5 * time.Millisecond}
	reg := newTestRegister(t, backend)
	reg.Load(context.Background(), "test", "/m.gguf", []int{0}, 0, LoadOpts{})

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(pid int) {
			defer wg.Done()
			reg.Chat(context.Background(),
				[]protocol.ChatMessage{chatMsg("system", "sys"), chatMsg("user", fmt.Sprintf("q from %d", pid))},
				protocol.InferenceOpts{}, func(string) {}, pid, false,
				false)
		}(i)
	}
	wg.Wait()

	// Each shell should have its own independent history of 3 messages.
	reg.historyMu.Lock()
	defer reg.historyMu.Unlock()
	for i := 0; i < 10; i++ {
		hist := reg.history[i]
		assertRoles(t, fmt.Sprintf("shell %d", i), hist, "system", "user", "assistant")
	}
}
