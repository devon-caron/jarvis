package daemon

import (
	"context"
	"errors"
	"fmt"
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

func newTestRegistry(t *testing.T, backends ...*mockBackend) *ModelRegistry {
	t.Helper()
	cfg := config.Defaults()
	factory := newMockBackendFactory(backends...)
	return NewModelRegistry(cfg, factory)
}

// --- ModelRegistry tests ---

// TestRegistry_LoadUnload verifies the full load/unload lifecycle: loads a model
// by name, checks Status() returns the correct ModelInfo, unloads by name, and
// confirms Status() returns nil afterward.
func TestRegistry_LoadUnload(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

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

// TestRegistry_Load_WhenAlreadyLoaded_Errors verifies that attempting to load
// a second model while one is already loaded returns an error, enforcing the
// single-model constraint.
func TestRegistry_Load_WhenAlreadyLoaded_Errors(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	reg.Load(context.Background(), "test", "/model1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load(context.Background(), "test2", "/model2.gguf", []int{1}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected error when a model is already loaded")
	}
}

// TestRegistry_LoadError verifies that when the backend returns a load error,
// the registry remains unloaded (Status() returns nil), and a fresh registry
// can still load successfully afterward.
func TestRegistry_LoadError(t *testing.T) {
	backend := &mockBackend{loadErr: errors.New("load failed")}
	reg := newTestRegistry(t, backend)

	err := reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Error("expected load error")
	}

	if reg.Status() != nil {
		t.Error("no model should be loaded after error")
	}

	// Should be able to load again after failure
	backend2 := &mockBackend{}
	reg2 := newTestRegistry(t, backend2)
	if err := reg2.Load(context.Background(), "test2", "/model2.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("should be able to load after prior failure: %v", err)
	}
}

// TestRegistry_UnloadWhenEmpty verifies that calling Unload on an empty
// registry (no model loaded) returns an error.
func TestRegistry_UnloadWhenEmpty(t *testing.T) {
	reg := newTestRegistry(t)

	err := reg.Unload("")
	if err == nil {
		t.Error("expected error when unloading with no model")
	}
}

// TestRegistry_UnloadEmptyName verifies that calling Unload with an empty name
// successfully unloads the currently loaded model.
func TestRegistry_UnloadEmptyName(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	// Unload without name should unload the loaded model
	if err := reg.Unload(""); err != nil {
		t.Fatalf("Unload: %v", err)
	}

	if reg.Status() != nil {
		t.Error("expected nil status after unload")
	}
}

// TestRegistry_UnloadWrongName verifies that calling Unload with a name that
// doesn't match the loaded model returns an error.
func TestRegistry_UnloadWrongName(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Unload("nonexistent")
	if err == nil {
		t.Error("expected error for wrong model name")
	}
}

// TestRegistry_UnloadNotLoaded verifies that calling Unload with a specific
// name when no model is loaded returns an error.
func TestRegistry_UnloadNotLoaded(t *testing.T) {
	reg := newTestRegistry(t)

	err := reg.Unload("nonexistent")
	if err == nil {
		t.Error("expected error when nothing is loaded")
	}
}

// TestRegistry_Chat verifies that Chat delegates to the backend's RunChat,
// collecting streamed tokens ("Hello ", "world!") via the onToken callback.
func TestRegistry_Chat(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var tokens []string
	err := reg.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(token string) { tokens = append(tokens, token) },
		0, false,
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(tokens) != 2 || tokens[0] != "Hello " || tokens[1] != "world!" {
		t.Errorf("tokens = %v", tokens)
	}
}

// TestRegistry_Chat_NoModelLoaded verifies that Chat returns an error when
// no model is loaded.
func TestRegistry_Chat_NoModelLoaded(t *testing.T) {
	reg := newTestRegistry(t)

	err := reg.Chat(context.Background(),
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
	)
	if err == nil {
		t.Error("expected error when no model loaded")
	}
}

// TestRegistry_Chat_ChatError verifies that Chat propagates errors returned
// by the backend's RunChat method.
func TestRegistry_Chat_ChatError(t *testing.T) {
	backend := &mockBackend{chatErr: errors.New("chat failed")}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(),
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
	)
	if err == nil || err.Error() != "chat failed" {
		t.Errorf("expected chat error, got: %v", err)
	}
}

// TestRegistry_ConcurrentChat verifies that 10 concurrent Chat calls all
// complete successfully and the backend receives exactly 10 RunChat calls,
// confirming thread safety under the RWMutex.
func TestRegistry_ConcurrentChat(t *testing.T) {
	backend := &mockBackend{chatDelay: 10 * time.Millisecond}
	reg := newTestRegistry(t, backend)

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
			)
		}()
	}
	wg.Wait()

	if got := backend.chatCalls.Load(); got != 10 {
		t.Errorf("chatCalls = %d, want 10", got)
	}
}

// TestRegistry_Status_Empty verifies Status() returns nil when no model
// is loaded.
func TestRegistry_Status_Empty(t *testing.T) {
	reg := newTestRegistry(t)

	if reg.Status() != nil {
		t.Error("expected nil status when no model loaded")
	}
}

// TestRegistry_Status_WithModel verifies Status() returns correct ModelInfo
// (name and model path) after loading a model.
func TestRegistry_Status_WithModel(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

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

// TestRegistry_IsLoaded verifies the IsLoaded() getter returns false initially,
// true after Load, and false again after Unload.
func TestRegistry_IsLoaded(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

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

// TestRegistry_Shutdown verifies Shutdown unloads the backend, clears Status()
// to nil, and sets IsLoaded() to false.
func TestRegistry_Shutdown(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

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

// TestRegistry_UnloadError verifies that Unload propagates errors returned
// by the backend's UnloadModel method.
func TestRegistry_UnloadError(t *testing.T) {
	backend := &mockBackend{unloadErr: errors.New("unload failed")}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Unload("test")
	if err == nil {
		t.Error("expected unload error")
	}
}

// TestRegistry_Load_ContextCancelled verifies that cancelling the context
// during a slow Load causes it to return an error, and the registry remains
// in an unloaded state afterward.
func TestRegistry_Load_ContextCancelled(t *testing.T) {
	backend := &mockBackend{loadDelay: 5 * time.Second}
	reg := newTestRegistry(t, backend)

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

// TestRegistry_Load_FailureAllowsReload verifies that after a failed Load
// (backend error), the registry is not stuck in a "loaded" state and a
// subsequent Load on the same registry succeeds.
func TestRegistry_Load_FailureAllowsReload(t *testing.T) {
	backend := &mockBackend{loadErr: fmt.Errorf("simulated failure")}
	backend2 := &mockBackend{}
	reg := newTestRegistry(t, backend, backend2)

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

// TestRegistry_LoadAfterUnload verifies the full cycle of load -> unload ->
// load with a different model, confirming the second model is correctly
// reflected in Status().
func TestRegistry_LoadAfterUnload(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

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
