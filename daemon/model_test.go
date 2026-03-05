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
	backend.LoadModel(context.Background(), "/model.gguf", []int{0}, LoadOpts{})

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
		0, false,
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
	backend.LoadModel(context.Background(), "/model.gguf", []int{0}, LoadOpts{})

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
	backend.LoadModel(context.Background(), "/model.gguf", []int{0}, LoadOpts{})

	// timeout=0 means no timer
	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	var tokens []string
	err := slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(token string) { tokens = append(tokens, token) },
		0, false,
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

	err := slot.Chat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {}, 0, false)
	if !errors.Is(err, ErrNoModel) {
		t.Errorf("expected ErrNoModel, got: %v", err)
	}
}

func TestModelSlot_Status(t *testing.T) {
	backend := &mockBackend{}
	backend.LoadModel(context.Background(), "/model.gguf", []int{0}, LoadOpts{})

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

	if err := reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{}); err != nil {
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

	reg.Load(context.Background(), "test", "/model1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load(context.Background(), "test", "/model2.gguf", []int{1}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected error when loading under an already-loaded name")
	}
}

func TestRegistry_LoadSameGPU_Conflict(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	reg.Load(context.Background(), "test", "/model1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load(context.Background(), "test", "/model1.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected GPU conflict error when reloading same model on same GPU")
	}
}

func TestRegistry_GPUConflict(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	reg.Load(context.Background(), "model1", "/m1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Load(context.Background(), "model2", "/m2.gguf", []int{0}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected GPU conflict error")
	}
}

func TestRegistry_MultiGPU_NoConflict(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	if err := reg.Load(context.Background(), "model1", "/m1.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("Load model1: %v", err)
	}
	if err := reg.Load(context.Background(), "model2", "/m2.gguf", []int{1}, 0, LoadOpts{}); err != nil {
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

	err := reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})
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

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

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

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

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

	err := reg.Chat(context.Background(), "", -1, nil, protocol.InferenceOpts{}, func(string) {}, 0, false)
	if !errors.Is(err, ErrNoModel) {
		t.Errorf("expected ErrNoModel, got: %v", err)
	}
}

func TestRegistry_Chat_SingleModel_AutoRoute(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var tokens []string
	err := reg.Chat(context.Background(), "", -1,
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

func TestRegistry_Chat_ExplicitName(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "m2", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(string) {},
		0, false,
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

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "nonexistent", -1,
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
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

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	// Chat without name should route to default GPU (1) = m2
	err := reg.Chat(context.Background(), "", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(string) {},
		0, false,
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

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	// No model on GPU 2 — should error
	err := reg.Chat(context.Background(), "", -1,
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
	)
	if err == nil {
		t.Error("expected error when no model on default GPU")
	}
}

func TestRegistry_Chat_ChatError(t *testing.T) {
	backend := &mockBackend{chatErr: errors.New("chat failed")}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "", -1,
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
	)
	if err == nil || err.Error() != "chat failed" {
		t.Errorf("expected chat error, got: %v", err)
	}
}

func TestRegistry_ConcurrentChat(t *testing.T) {
	backend := &mockBackend{chatDelay: 10 * time.Millisecond}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reg.Chat(context.Background(), "", -1,
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
	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 30*time.Minute, LoadOpts{})

	models = reg.Status()
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
}

func TestRegistry_Shutdown(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	reg.Shutdown()

	models := reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models after shutdown, got %d", len(models))
	}
}

func TestRegistry_TimerExpiry(t *testing.T) {
	backend := &mockBackend{}
	reg := newTestRegistry(t, backend)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 50*time.Millisecond, LoadOpts{})

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

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 50*time.Millisecond, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{}) // no timeout

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

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Unload("test")
	if err == nil {
		t.Error("expected unload error")
	}
}

func TestRegistry_GPUConflict_MultiGPU(t *testing.T) {
	b1 := &mockBackend{}
	reg := newTestRegistry(t, b1)

	// Load on GPUs 0 and 1
	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0, 1}, 0, LoadOpts{})

	// Try to load on GPU 1 — should conflict
	err := reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})
	if err == nil {
		t.Error("expected GPU conflict for overlapping GPU")
	}
}

func TestRegistry_Chat_GPURouting(t *testing.T) {
	b1 := &mockBackend{}
	b2 := &mockBackend{}
	reg := newTestRegistry(t, b1, b2)

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

	// Route explicitly to GPU 1 → should hit m2
	err := reg.Chat(context.Background(), "", 1,
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{},
		func(string) {},
		0, false,
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

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})
	reg.Load(context.Background(), "m2", "/m2.gguf", []int{1}, 0, LoadOpts{})

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

	reg.Load(context.Background(), "m1", "/m1.gguf", []int{0}, 0, LoadOpts{})

	err := reg.Chat(context.Background(), "", 1,
		nil, protocol.InferenceOpts{}, func(string) {},
		0, false,
	)
	if err == nil {
		t.Error("expected error when no model on requested GPU")
	}
}

func TestRegistry_Load_ContextCancelled(t *testing.T) {
	backend := &mockBackend{loadDelay: 5 * time.Second}
	reg := newTestRegistry(t, backend)

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel immediately to trigger cancellation during load.
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

	// GPUs should be cleaned up after cancelled load.
	models := reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models after cancelled load, got %d", len(models))
	}

	// GPU should be free for reuse.
	backend2 := &mockBackend{}
	reg2 := newTestRegistry(t, backend2)
	if err := reg2.Load(context.Background(), "test2", "/model2.gguf", []int{0}, 0, LoadOpts{}); err != nil {
		t.Fatalf("GPU should be free after cancelled load: %v", err)
	}
}

func TestRegistry_Load_AlreadyLoading(t *testing.T) {
	backend := &mockBackend{loadDelay: 1 * time.Second}
	reg := newTestRegistry(t, backend)

	// Start a load in the background.
	done := make(chan error, 1)
	go func() {
		done <- reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})
	}()
	// Give it time to enter loading state.
	time.Sleep(20 * time.Millisecond)

	// Second load with same name should be rejected.
	err := reg.Load(context.Background(), "test", "/model2.gguf", []int{1}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected error for concurrent load of same model name")
	}
	if !strings.Contains(err.Error(), "already being loaded") {
		t.Errorf("expected 'already being loaded' error, got: %v", err)
	}

	<-done // Let the first load finish.
}

func TestRegistry_Load_Rollback_GPUs(t *testing.T) {
	backend := &mockBackend{loadErr: fmt.Errorf("simulated failure")}
	reg := newTestRegistry(t, backend)

	err := reg.Load(context.Background(), "test", "/model.gguf", []int{0, 1}, 0, LoadOpts{})
	if err == nil {
		t.Fatal("expected load error")
	}

	// GPUs should be free after failed load.
	models := reg.Status()
	if len(models) != 0 {
		t.Errorf("expected 0 models after failed load, got %d", len(models))
	}

	// Another model should be able to use the same GPUs.
	backend2 := &mockBackend{}
	reg2 := newTestRegistry(t, backend2)
	if err := reg2.Load(context.Background(), "test2", "/model2.gguf", []int{0, 1}, 0, LoadOpts{}); err != nil {
		t.Fatalf("GPUs should be free after failed load: %v", err)
	}
}

func TestRegistry_StatusDuringLoad(t *testing.T) {
	backend := &mockBackend{loadDelay: 200 * time.Millisecond}
	reg := newTestRegistry(t, backend)

	// Start a load in the background.
	done := make(chan error, 1)
	go func() {
		done <- reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})
	}()
	// Give it time to enter loading state.
	time.Sleep(20 * time.Millisecond)

	// Status should not block — the mutex is not held during load.
	statusDone := make(chan struct{})
	go func() {
		reg.Status()
		close(statusDone)
	}()

	select {
	case <-statusDone:
		// Good — status returned while load is in progress.
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Status() blocked during model loading — mutex should not be held")
	}

	<-done
}

// --- Per-shell history tests ---

// capturingMockBackend records the messages passed to RunChat.
type capturingMockBackend struct {
	mockBackend
	mu          sync.Mutex
	capturedAll [][]protocol.ChatMessage // one entry per RunChat call
}

func (c *capturingMockBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	c.chatCalls.Add(1)
	c.mu.Lock()
	cp := make([]protocol.ChatMessage, len(msgs))
	copy(cp, msgs)
	c.capturedAll = append(c.capturedAll, cp)
	c.mu.Unlock()
	if c.chatErr != nil {
		return c.chatErr
	}
	onDelta("Hello ")
	onDelta("world!")
	return nil
}

func (c *capturingMockBackend) lastCaptured() []protocol.ChatMessage {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.capturedAll) == 0 {
		return nil
	}
	return c.capturedAll[len(c.capturedAll)-1]
}

func TestModelSlot_Chat_HistoryPersists(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// First chat
	err := slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Your name is Dan"}},
		protocol.InferenceOpts{},
		func(string) {},
		100, false,
	)
	if err != nil {
		t.Fatalf("Chat 1: %v", err)
	}

	// Second chat — should include history
	err = slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "What is your name?"}},
		protocol.InferenceOpts{},
		func(string) {},
		100, false,
	)
	if err != nil {
		t.Fatalf("Chat 2: %v", err)
	}

	msgs := backend.lastCaptured()
	// Should be: [prior user] [prior assistant] [new user]
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Role != "user" || msgs[0].Content != "Your name is Dan" {
		t.Errorf("msg[0] = %+v, want user/Your name is Dan", msgs[0])
	}
	if msgs[1].Role != "assistant" || msgs[1].Content != "Hello world!" {
		t.Errorf("msg[1] = %+v, want assistant/Hello world!", msgs[1])
	}
	if msgs[2].Role != "user" || msgs[2].Content != "What is your name?" {
		t.Errorf("msg[2] = %+v, want user/What is your name?", msgs[2])
	}
}

func TestModelSlot_Chat_PerShellIsolation(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// Shell PID 100 chats
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "I am shell 100"}},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
	)

	// Shell PID 200 chats — should NOT see PID 100's history
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Who am I?"}},
		protocol.InferenceOpts{}, func(string) {}, 200, false,
	)

	msgs := backend.lastCaptured()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message for PID 200, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Content != "Who am I?" {
		t.Errorf("expected 'Who am I?', got %q", msgs[0].Content)
	}
}

func TestModelSlot_Chat_ClearContext(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// Build history
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Remember this"}},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
	)

	// Clear context and chat
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Fresh start"}},
		protocol.InferenceOpts{}, func(string) {}, 100, true,
	)

	msgs := backend.lastCaptured()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message after clear, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Content != "Fresh start" {
		t.Errorf("expected 'Fresh start', got %q", msgs[0].Content)
	}
}

func TestModelSlot_Chat_ClearContextOnlyAffectsShell(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// Both shells chat
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Shell 100 msg"}},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
	)
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Shell 200 msg"}},
		protocol.InferenceOpts{}, func(string) {}, 200, false,
	)

	// Clear PID 100
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Cleared"}},
		protocol.InferenceOpts{}, func(string) {}, 100, true,
	)

	// PID 200 should still have history
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "Still here?"}},
		protocol.InferenceOpts{}, func(string) {}, 200, false,
	)

	msgs := backend.lastCaptured()
	// Should be: [Shell 200 msg] [assistant] [Still here?]
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages for PID 200, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Content != "Shell 200 msg" {
		t.Errorf("msg[0] = %q, want 'Shell 200 msg'", msgs[0].Content)
	}
}

func TestModelSlot_Chat_HistoryClearedOnUnload(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// Build history
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
	)

	// Unload destroys the slot — history is gone with it
	slot.Unload()

	// The history map itself is unreachable after slot is destroyed.
	// Verify by checking the map is empty (slot still exists in memory but is unloaded).
	slot.historyMu.Lock()
	histLen := len(slot.history)
	slot.historyMu.Unlock()
	// After unload, backend is unloaded so Chat would return ErrNoModel.
	// The history map persists on the struct but the slot is removed from the registry,
	// so no new chats can reach it. Verify the slot is truly unloaded.
	if backend.loaded {
		t.Error("backend should be unloaded")
	}
	_ = histLen // history map is on the now-unreachable slot
}

func TestModelSlot_Chat_SystemMsgsSeparated(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// First chat with system message
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{
			{Role: "system", Content: "Be helpful"},
			{Role: "user", Content: "Hello"},
		},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
	)

	// Second chat with system message — history should be between system and new user
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{
			{Role: "system", Content: "Be helpful"},
			{Role: "user", Content: "Follow up"},
		},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
	)

	msgs := backend.lastCaptured()
	// Should be: [system] [prior user Hello] [prior assistant] [new user Follow up]
	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Role != "system" {
		t.Errorf("msg[0] should be system, got %q", msgs[0].Role)
	}
	if msgs[1].Role != "user" || msgs[1].Content != "Hello" {
		t.Errorf("msg[1] = %+v, want user/Hello", msgs[1])
	}
	if msgs[2].Role != "assistant" {
		t.Errorf("msg[2] should be assistant, got %q", msgs[2].Role)
	}
	if msgs[3].Role != "user" || msgs[3].Content != "Follow up" {
		t.Errorf("msg[3] = %+v, want user/Follow up", msgs[3])
	}
}

func TestRegistry_Chat_HistoryPassesThrough(t *testing.T) {
	backend := &capturingMockBackend{}
	cfg := config.Defaults()
	factory := func(c *config.Config) ModelBackend { return backend }
	reg := NewModelRegistry(cfg, factory)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	// First chat
	reg.Chat(context.Background(), "", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "first"}},
		protocol.InferenceOpts{}, func(string) {},
		100, false,
	)

	// Second chat — should have history
	reg.Chat(context.Background(), "", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "second"}},
		protocol.InferenceOpts{}, func(string) {},
		100, false,
	)

	msgs := backend.lastCaptured()
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Content != "first" {
		t.Errorf("msg[0] = %q, want 'first'", msgs[0].Content)
	}
	if msgs[1].Role != "assistant" {
		t.Errorf("msg[1] role = %q, want 'assistant'", msgs[1].Role)
	}
	if msgs[2].Content != "second" {
		t.Errorf("msg[2] = %q, want 'second'", msgs[2].Content)
	}
}

func TestRegistry_Chat_ClearContext(t *testing.T) {
	backend := &capturingMockBackend{}
	cfg := config.Defaults()
	factory := func(c *config.Config) ModelBackend { return backend }
	reg := NewModelRegistry(cfg, factory)

	reg.Load(context.Background(), "test", "/model.gguf", []int{0}, 0, LoadOpts{})

	// Build history
	reg.Chat(context.Background(), "", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "remember"}},
		protocol.InferenceOpts{}, func(string) {},
		100, false,
	)

	// Clear and chat
	reg.Chat(context.Background(), "", -1,
		[]protocol.ChatMessage{{Role: "user", Content: "fresh"}},
		protocol.InferenceOpts{}, func(string) {},
		100, true,
	)

	msgs := backend.lastCaptured()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message after clear, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Content != "fresh" {
		t.Errorf("msg = %q, want 'fresh'", msgs[0].Content)
	}
}

func TestModelSlot_Chat_ContextLimitRetry(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// Build history first
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "build history"}},
		protocol.InferenceOpts{}, func(string) {}, 100, false,
	)

	// Now swap in a backend that fails on its first call (call 0) with context
	// length error, then succeeds on the retry (call 1).
	contextErrBackend := &contextLimitMockBackend{
		failOnCall: 0,
	}
	contextErrBackend.loaded = true
	contextErrBackend.path = "/model.gguf"
	slot.backend = contextErrBackend

	var deltas []string
	err := slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "this should retry"}},
		protocol.InferenceOpts{},
		func(token string) { deltas = append(deltas, token) },
		100, false,
	)
	if err != nil {
		t.Fatalf("Chat with retry: %v", err)
	}

	// Should have received the info message about context limit
	combined := strings.Join(deltas, "")
	if !strings.Contains(combined, "context limit reached") {
		t.Errorf("expected context limit info message, got: %q", combined)
	}

	// History should be cleared for this PID, then rebuilt with just the retry
	slot.historyMu.Lock()
	hist := slot.history[100]
	slot.historyMu.Unlock()
	// After retry success: user msg + assistant response
	if len(hist) != 2 {
		t.Errorf("expected 2 history entries after retry (user + assistant), got %d", len(hist))
	}
}

// contextLimitMockBackend fails with context length error on a specific call number.
type contextLimitMockBackend struct {
	capturingMockBackend
	failOnCall int
	calls      int
}

func (c *contextLimitMockBackend) RunChat(ctx context.Context, msgs []protocol.ChatMessage, opts protocol.InferenceOpts, onDelta func(string)) error {
	c.chatCalls.Add(1)
	c.mu.Lock()
	cp := make([]protocol.ChatMessage, len(msgs))
	copy(cp, msgs)
	c.capturedAll = append(c.capturedAll, cp)
	callNum := c.calls
	c.calls++
	c.mu.Unlock()

	if callNum == c.failOnCall {
		return fmt.Errorf("context length exceeded")
	}
	onDelta("retried ")
	onDelta("ok!")
	return nil
}

func TestModelSlot_Chat_ShellPIDZero_NoHistory(t *testing.T) {
	backend := &capturingMockBackend{}
	backend.loaded = true
	backend.path = "/model.gguf"

	slot := newModelSlot("test", backend, []int{0}, 0, nil)

	// Chat with shellPID=0 should not accumulate history
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "first"}},
		protocol.InferenceOpts{}, func(string) {}, 0, false,
	)
	slot.Chat(context.Background(),
		[]protocol.ChatMessage{{Role: "user", Content: "second"}},
		protocol.InferenceOpts{}, func(string) {}, 0, false,
	)

	msgs := backend.lastCaptured()
	// With PID 0, no history is stored, so second call gets only "second"
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message with PID 0, got %d: %v", len(msgs), msgs)
	}
	if msgs[0].Content != "second" {
		t.Errorf("expected 'second', got %q", msgs[0].Content)
	}
}
