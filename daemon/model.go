package daemon

import (
	"sync"
	"time"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/protocol"
)

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

func (m *ModelRegistry) Shutdown() {
	// TODO: Unimplementeds
	panic("unimplemented")
}
