package ptyshell

import (
	"strings"
	"sync"
	"time"
)

// ContextEntry represents a single entry in the LLM context buffer.
type ContextEntry struct {
	Type      string    // "command", "output", "system"
	Content   string
	Timestamp time.Time
}

// ContextBuffer accumulates sanitized terminal I/O for LLM consumption.
// All writes are thread-safe. Older entries are trimmed when maxSize is exceeded.
type ContextBuffer struct {
	mu      sync.Mutex
	entries []ContextEntry
	maxSize int
}

// NewContextBuffer creates a ContextBuffer with the given max entry count.
func NewContextBuffer(maxSize int) *ContextBuffer {
	return &ContextBuffer{
		maxSize: maxSize,
		entries: make([]ContextEntry, 0, 256),
	}
}

// AppendCommand adds a sanitized command entry to the context buffer.
func (cb *ContextBuffer) AppendCommand(sanitizedCmd string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.entries = append(cb.entries, ContextEntry{
		Type:      "command",
		Content:   sanitizedCmd,
		Timestamp: time.Now(),
	})
	cb.trim()
}

// AppendOutput adds sanitized output to the context buffer.
// Consecutive output entries are merged to reduce entry count.
func (cb *ContextBuffer) AppendOutput(sanitizedOutput string) {
	if sanitizedOutput == "" {
		return
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Merge with last entry if it's also output
	if len(cb.entries) > 0 && cb.entries[len(cb.entries)-1].Type == "output" {
		cb.entries[len(cb.entries)-1].Content += sanitizedOutput
		return
	}

	cb.entries = append(cb.entries, ContextEntry{
		Type:      "output",
		Content:   sanitizedOutput,
		Timestamp: time.Now(),
	})
	cb.trim()
}

// AppendSystem adds a system message entry to the context buffer.
func (cb *ContextBuffer) AppendSystem(msg string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.entries = append(cb.entries, ContextEntry{
		Type:      "system",
		Content:   msg,
		Timestamp: time.Now(),
	})
	cb.trim()
}

// Format returns the context buffer as a string suitable for LLM consumption.
func (cb *ContextBuffer) Format() string {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	var sb strings.Builder
	sb.WriteString("Terminal session context:\n")
	for _, e := range cb.entries {
		switch e.Type {
		case "command":
			sb.WriteString("$ " + e.Content + "\n")
		case "output":
			sb.WriteString(e.Content)
			if !strings.HasSuffix(e.Content, "\n") {
				sb.WriteString("\n")
			}
		case "system":
			sb.WriteString("[" + e.Content + "]\n")
		}
	}
	return sb.String()
}

// Len returns the number of entries in the buffer.
func (cb *ContextBuffer) Len() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return len(cb.entries)
}

// Entries returns a copy of all entries.
func (cb *ContextBuffer) Entries() []ContextEntry {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cp := make([]ContextEntry, len(cb.entries))
	copy(cp, cb.entries)
	return cp
}

// trim drops oldest entries when maxSize is exceeded.
func (cb *ContextBuffer) trim() {
	if cb.maxSize > 0 && len(cb.entries) > cb.maxSize {
		cb.entries = cb.entries[len(cb.entries)-cb.maxSize:]
	}
}
