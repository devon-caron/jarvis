package daemon

import (
	"context"
	"errors"
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
	unloadErr  error
	chatErr    error
	chatDelay  time.Duration
	chatCalls  atomic.Int32
}

func (m *mockBackend) LoadModel(path string, gpus []int, opts LoadOpts) error {
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

func (m *mockBackend) GetStatus() (*protocol.ModelStatus, error) {
	return &protocol.ModelStatus{
		ModelPath: m.path,
		GPULayers: 80,
		GPUs: []protocol.GPUInfo{
			{DeviceID: 0, DeviceName: "Test GPU", FreeMemoryMB: 20000, TotalMemoryMB: 24000},
		},
	}, nil
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

// --- ModelSlot tests ---

func TestModelSlot_Chat_ResetsTimer(t *testing.T) {
	backend := &mockBackend{}
	backend.LoadModel("/model.gguf", []int{0}, LoadOpts{})

	expired := make(chan string, 1)
	slot := newModelSlot("test", backend, []int{0}, 100*time.Millisecond, func(name string) {
		expired <- name
	})

	// Chat should reset timer
	time.Sleep(60 * time.Millisecond)
	err := slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(string) {},
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	// Should not expire yet (timer was reset)
	select {
	case <-expired:
		t.Fatal("should not have expired yet")
	case <-time.After(60 * time.Millisecond):
		// Good — not expired
	}

	// Should eventually expire
	select {
	case name := <-expired:
		if name != "test" {
			t.Errorf("expired name = %q, want test", name)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected timer to fire")
	}
}

func TestModelSlot_Unload_StopsTimer(t *testing.T) {
	backend := &mockBackend{}
	backend.LoadModel("/model.gguf", []int{0}, LoadOpts{})

	expired := make(chan string, 1)
	slot := newModelSlot("test", backend, []int{0}, 50*time.Millisecond, func(name string) {
		expired <- name
	})

	slot.Unload()

	// Timer should not fire after unload
	select {
	case <-expired:
		t.Fatal("timer should not fire after unload")
	case <-time.After(100 * time.Millisecond):
		// Good
	}
}

func TestModelSlot_Chat_NoTimer(t *testing.T) {
	backend := &mockBackend{}
	backend.LoadModel("/model.gguf", []int{0}, LoadOpts{})

	// timeout=0 means no timer
	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	var tokens []string
	err := slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(token string) { tokens = append(tokens, token) },
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(tokens) != 2 {
		t.Errorf("got %d tokens, want 2", len(tokens))
	}
}

func TestModelSlot_Chat_NotLoaded(t *testing.T) {
	backend := &mockBackend{} // not loaded
	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	err := slot.Chat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {})
	if !errors.Is(err, ErrNoModel) {
		t.Errorf("expected ErrNoModel, got: %v", err)
	}
}

func TestModelSlot_Status(t *testing.T) {
	backend := &mockBackend{}
	backend.LoadModel("/model.gguf", []int{0}, LoadOpts{})

	slot := newModelSlot("mymodel", backend, []int{0, 1}, 30*time.Minute, nil)
	info := slot.Status()

	if info.Name != "mymodel" {
		t.Errorf("Name = %q, want mymodel", info.Name)
	}
	if info.ModelPath != "/model.gguf" {
		t.Errorf("ModelPath = %q, want /model.gguf", info.ModelPath)
	}
	if len(info.GPUs) != 2 || info.GPUs[0] != 0 || info.GPUs[1] != 1 {
		t.Errorf("GPUs = %v, want [0, 1]", info.GPUs)
	}
	if info.Timeout != "30m0s" {
		t.Errorf("Timeout = %q, want 30m0s", info.Timeout)
	}
}

// --- ModelRegistry tests ---

func TestRegistry_LoadUnload(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	if err := reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load: %v", err)
	}

	models := reg.Status()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].Name != "test" {
		t.Errorf("Name = %q, want test", models[0].Name)
	}

	if err := reg.Unload("test"); err != nil {
		t.Fatalf("Unload: %v", err)
	}

	models = reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestRegistry_Load_DuplicateName_Errors(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	reg.Load("test", "/model1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load("test", "/model2.gguf", []int{1}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected error when loading under an already-loaded name")
	}
}

func TestRegistry_LoadSameGPU_Conflict(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	reg.Load("test", "/model1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load("test", "/model1.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected GPU conflict error when reloading same model on same GPU")
	}
}

func TestRegistry_GPUConflict(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	reg.Load("model1", "/m1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load("model2", "/m2.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected GPU conflict error")
	}
}

func TestRegistry_MultiGPU_NoConflict(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	if err := reg.Load("model1", "/m1.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load model1: %v", err)
	}
	if err := reg.Load("model2", "/m2.gguf", []int{1}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load model2: %v", err)
	}

	models := reg.Status()
	if len(models) != 2 {
		t.Errorf("expected 2 models, got %d", len(models))
	}
}

func TestRegistry_LoadError(t *testing.T) {
	backend := &mockBackend{loadErr: errors.New("load failed")}
	reg := newTestRegistry(t, backend)

	err := reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Error("expected load error")
	}

	models := reg.Status()
	if len(models) != 0 {
		t.Error("no models should be loaded after error")
	}
}

func TestRegistry_UnloadWhenEmpty(t *testing.T) {
	reg := newTestRegistry(t)

	err := reg.Unload("")
	if err == nil {
		t.Error("expected error when unloading with no models")
	}
}

func TestRegistry_UnloadEmptyName_SingleModel(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{})

	// Unload without name when only one model is loaded
	if err := reg.Unload(""); err != nil {
		t.Fatalf("Unload: %v", err)
	}

	models := reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestRegistry_UnloadEmptyName_MultipleModels(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	err := reg.Unload("")
	if err == nil {
		t.Error("expected error when multiple models loaded and no name given")
	}
}

func TestRegistry_UnloadNotFound(t *testing.T) {
	reg := newTestRegistry(t)

	err := reg.Unload("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

func TestRegistry_Chat_NoModel(t *testing.T) {
	reg := newTestRegistry(t)

	err := reg.Chat(context.Background(), "", -1, nil, protocol.InferenceOpts{}, func(string) {})
	if !errors.Is(err, ErrNoModel) {
		t.Errorf("expected ErrNoModel, got: %v", err)
	}
}

func TestRegistry_Chat_SingleModel_AutoRoute(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var tokens []string
	err := reg.Chat(context.Background(), "", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(token string) { tokens = append(tokens, token) },
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(tokens) != 2 || tokens[0] != "Hello " || tokens[1] != "world!" {
		t.Errorf("tokens = %v", tokens)
	}
}

func TestRegistry_Chat_ExplicitName(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "m2", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(string) {},
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if b2.chatCalls.Load() != 1 {
		t.Errorf("expected m2 to be called, got %d calls", b2.chatCalls.Load())
	}
}

func TestRegistry_Chat_ExplicitName_NotFound(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "nonexistent", -1,
		nil, protocol.InferenceOpts{}, func(string) {},
	)
	if err == nil {
		t.Error("expected error for nonexistent model name")
	}
}

func TestRegistry_Chat_DefaultGPU_Routing(t *testing.T) {
	cfg := config.Defaults()
	cfg.DefaultGPU = 1

	b1 := &mockBackend{}
	b2 := &mockBackend{}
	idx := 0
	backends := []*mockBackend{b1, b2}
	factory := func(c *config.Config) ModelBackend {
		b := backends[idx]
		idx++
		return b
	}
	reg := NewModelRegistry(cfg, factory)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	// Chat without name should route to default GPU (1) = m2
	err := reg.Chat(context.Background(), "", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(string) {},
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if b2.chatCalls.Load() != 1 {
		t.Errorf("expected m2 to be called (default GPU), got %d calls", b2.chatCalls.Load())
	}
	if b1.chatCalls.Load() != 0 {
		t.Errorf("m1 should not be called, got %d calls", b1.chatCalls.Load())
	}
}

func TestRegistry_Chat_DefaultGPU_NoModelOnDefault(t *testing.T) {
	cfg := config.Defaults()
	cfg.DefaultGPU = 2

	b1 := &mockBackend{}
	b2 := &mockBackend{}
	idx := 0
	backends := []*mockBackend{b1, b2}
	factory := func(c *config.Config) ModelBackend {
		b := backends[idx]
		idx++
		return b
	}
	reg := NewModelRegistry(cfg, factory)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	// No model on GPU 2 — should error
	err := reg.Chat(context.Background(), "", -1,
		nil, protocol.InferenceOpts{}, func(string) {},
	)
	if err == nil {
		t.Error("expected error when no model on default GPU")
	}
}

func TestRegistry_Chat_ChatError(t *testing.T) {
	backend := &mockBackend{chatErr: errors.New("chat failed")}
	reg := newTestRegistry(t, backend)

	reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "", -1,
		nil, protocol.InferenceOpts{}, func(string) {},
	)
	if err == nil || err.Error() != "chat failed" {
		t.Errorf("expected chat error, got: %v", err)
	}
}

func TestRegistry_ConcurrentChat(t *testing.T) {
	backend := &mockBackend{chatDelay: 10 * time.Millisecond}
	reg := newTestRegistry(t, backend)

	reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reg.Chat(context.Background(), "", -1,
				[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
				protocol.InferenceOpts{},
				func(string) {},
			)
		}()
	}
	wg.Wait()

	if got := backend.chatCalls.Load(); got != 10 {
		t.Errorf("chatCalls = %d, want 10", got)
	}
}

func TestRegistry_Status(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	// Empty registry
	models := reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}

	// With models
	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 30*time.Minute, LoadOpts{})

	models = reg.Status()
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
}

func TestRegistry_Shutdown(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	reg.Shutdown()

	models := reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models after shutdown, got %d", len(models))
	}
}

func TestRegistry_TimerExpiry(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load("test", "/model.gguf", []int{0}, 50*time.Millisecond, LoadOpts{})

	// Model should be loaded
	models := reg.Status()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}

	// Wait for timer expiry
	time.Sleep(150 * time.Millisecond)

	models = reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models after timeout, got %d", len(models))
	}
}

func TestRegistry_MultiModelIndependentUnload(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load("m1", "/m1.gguf", []int{0}, 50*time.Millisecond, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{}) // no timeout

	// Wait for m1 to expire
	time.Sleep(150 * time.Millisecond)

	models := reg.Status()
	if len(models) != 1 {
		t.Fatalf("expected 1 model after m1 timeout, got %d", len(models))
	}
	if models[0].Name != "m2" {
		t.Errorf("remaining model should be m2, got %q", models[0].Name)
	}
}

func TestRegistry_RemoveExpired_NotFound(t *testing.T) {
	reg := newTestRegistry(t)

	// Calling removeExpired for a name that doesn't exist should not panic
	reg.removeExpired("nonexistent")

	models := reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestRegistry_UnloadError(t *testing.T) {
	backend := &mockBackend{unloadErr: errors.New("unload failed")}
	reg := newTestRegistry(t, backend)

	reg.Load("test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Unload("test")
	if err == nil {
		t.Error("expected unload error")
	}
}

func TestRegistry_GPUConflict_MultiGPU(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	// Load on GPUs 0 and 1
	reg.Load("m1", "/m1.gguf", []int{0, 1}, 0, LoadOpts{})

	// Try to load on GPU 1 — should conflict
	err := reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})
	if err == nil {
		t.Error("expected GPU conflict for overlapping GPU")
	}
}

func TestRegistry_Chat_GPURouting(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	// Route explicitly to GPU 1 → should hit m2
	err := reg.Chat(context.Background(), "", 1,
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(string) {},
	)
	if err != nil {
		t.Fatalf("Chat by GPU: %v", err)
	}
	if b2.chatCalls.Load() != 1 {
		t.Errorf("expected m2 (GPU 1) to be called, got %d calls", b2.chatCalls.Load())
	}
	if b1.chatCalls.Load() != 0 {
		t.Errorf("m1 should not be called, got %d calls", b1.chatCalls.Load())
	}
}

func TestRegistry_UnloadByGPU(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load("m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	if err := reg.UnloadByGPU(0); err != nil {
		t.Fatalf("UnloadByGPU(0): %v", err)
	}

	models := reg.Status()
	if len(models) != 1 || models[0].Name != "m2" {
		t.Errorf("expected only m2 remaining, got %v", models)
	}
}

func TestRegistry_UnloadByGPU_NoModel(t *testing.T) {
	reg := newTestRegistry(t)

	err := reg.UnloadByGPU(0)
	if err == nil {
		t.Error("expected error when no model on GPU")
	}
}

func TestRegistry_Chat_GPURouting_NoModel(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load("m1", "/m1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "", 1,
		nil, protocol.InferenceOpts{}, func(string) {},
	)
	if err == nil {
		t.Error("expected error when no model on requested GPU")
	}
}
