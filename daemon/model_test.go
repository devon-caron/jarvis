package daemon

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/devon-caron/jarvis/protocol"
)

// mockBackend implements ModelBackend for testing.
type mockBackend struct {
	loaded    bool
	path      string
	loadErr   error
	unloadErr error
	chatErr   error
	chatDelay time.Duration
	chatCalls atomic.Int32
}

func (m *mockBackend) LoadModel(path string, gpuLayers int) error {
	if m.loadErr != nil {
		return m.loadErr
	}
	m.loaded = true
	m.path = path
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

func TestModelManager_LoadUnload(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)

	if mgr.IsLoaded() {
		t.Error("should not be loaded initially")
	}

	if err := mgr.Load("/model.gguf", -1); err != nil {
		t.Fatalf("Load: %v", err)
	}

	if !mgr.IsLoaded() {
		t.Error("should be loaded after Load")
	}
	if mgr.ModelPath() != "/model.gguf" {
		t.Errorf("ModelPath = %q, want /model.gguf", mgr.ModelPath())
	}

	if err := mgr.Unload(); err != nil {
		t.Fatalf("Unload: %v", err)
	}

	if mgr.IsLoaded() {
		t.Error("should not be loaded after Unload")
	}
}

func TestModelManager_LoadReplacesExisting(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)

	mgr.Load("/model1.gguf", -1)
	mgr.Load("/model2.gguf", -1)

	if mgr.ModelPath() != "/model2.gguf" {
		t.Errorf("ModelPath = %q, want /model2.gguf", mgr.ModelPath())
	}
}

func TestModelManager_LoadError(t *testing.T) {
	backend := &mockBackend{loadErr: errors.New("load failed")}
	mgr := NewModelManager(backend)

	err := mgr.Load("/model.gguf", -1)
	if err == nil {
		t.Error("expected load error")
	}
	if mgr.IsLoaded() {
		t.Error("should not be loaded after error")
	}
}

func TestModelManager_UnloadWhenNotLoaded(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)

	err := mgr.Unload()
	if err == nil {
		t.Error("expected error when unloading with no model")
	}
}

func TestModelManager_Chat(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)

	// Chat without model
	err := mgr.Chat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {})
	if !errors.Is(err, ErrNoModel) {
		t.Errorf("Chat without model should return ErrNoModel, got: %v", err)
	}

	// Load and chat
	mgr.Load("/model.gguf", -1)

	var tokens []string
	err = mgr.Chat(context.Background(),
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
	if tokens[0] != "Hello " || tokens[1] != "world!" {
		t.Errorf("tokens = %v", tokens)
	}
}

func TestModelManager_ChatError(t *testing.T) {
	backend := &mockBackend{chatErr: errors.New("chat failed")}
	mgr := NewModelManager(backend)
	mgr.Load("/model.gguf", -1)

	err := mgr.Chat(context.Background(), nil, protocol.InferenceOpts{}, func(string) {})
	if err == nil || err.Error() != "chat failed" {
		t.Errorf("expected chat error, got: %v", err)
	}
}

func TestModelManager_Status(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)

	// Status without model
	loaded, status := mgr.Status()
	if loaded {
		t.Error("should not be loaded")
	}
	if status != nil {
		t.Error("status should be nil when not loaded")
	}

	// Status with model
	mgr.Load("/model.gguf", -1)
	loaded, status = mgr.Status()
	if !loaded {
		t.Error("should be loaded")
	}
	if status == nil {
		t.Fatal("status should not be nil")
	}
	if status.GPULayers != 80 {
		t.Errorf("GPULayers = %d, want 80", status.GPULayers)
	}
}

func TestModelManager_Shutdown(t *testing.T) {
	backend := &mockBackend{}
	mgr := NewModelManager(backend)

	// Shutdown with no model is fine
	if err := mgr.Shutdown(); err != nil {
		t.Fatalf("Shutdown (no model): %v", err)
	}

	// Shutdown with loaded model
	mgr.Load("/model.gguf", -1)
	if err := mgr.Shutdown(); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if mgr.IsLoaded() {
		t.Error("should not be loaded after shutdown")
	}
}

func TestModelManager_ConcurrentChat(t *testing.T) {
	backend := &mockBackend{chatDelay: 10 * time.Millisecond}
	mgr := NewModelManager(backend)
	mgr.Load("/model.gguf", -1)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mgr.Chat(context.Background(),
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

func TestModelManager_LoadBlocksChat(t *testing.T) {
	backend := &mockBackend{chatDelay: 50 * time.Millisecond}
	mgr := NewModelManager(backend)
	mgr.Load("/model1.gguf", -1)

	// Start a chat that takes time
	chatDone := make(chan struct{})
	go func() {
		mgr.Chat(context.Background(),
			[]protocol.ChatMessage{{Role: "user", Content: "hi"}},
			protocol.InferenceOpts{},
			func(string) {},
		)
		close(chatDone)
	}()

	// Give chat time to start
	time.Sleep(10 * time.Millisecond)

	// Load should block until chat finishes, then replace model
	loadDone := make(chan struct{})
	go func() {
		mgr.Load("/model2.gguf", -1)
		close(loadDone)
	}()

	// Chat should finish first
	<-chatDone
	<-loadDone

	if mgr.ModelPath() != "/model2.gguf" {
		t.Errorf("ModelPath = %q, want /model2.gguf", mgr.ModelPath())
	}
}
