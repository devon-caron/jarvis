package pty

import (
	"bytes"
	"io"
	"strings"
	"sync"
)

const redactionMask = "*************"

// Redactor stores secret values and replaces them with a mask in byte streams.
type Redactor struct {
	mu      sync.RWMutex
	secrets [][]byte
}

// NewRedactor creates an empty Redactor.
func NewRedactor() *Redactor {
	return &Redactor{}
}

// AddSecret registers a secret value for redaction. Empty values are ignored.
func (r *Redactor) AddSecret(secret string) {
	if secret == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.secrets = append(r.secrets, []byte(secret))
}

// Redact replaces all known secret values in data with the redaction mask.
func (r *Redactor) Redact(data []byte) []byte {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := data
	for _, secret := range r.secrets {
		result = bytes.ReplaceAll(result, secret, []byte(redactionMask))
	}
	return result
}

// RedactString replaces all known secret values in a string.
func (r *Redactor) RedactString(s string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, secret := range r.secrets {
		s = strings.ReplaceAll(s, string(secret), redactionMask)
	}
	return s
}

// RedactingWriter wraps an io.Writer and redacts known secrets before writing
// to the underlying writer. It also writes unredacted data to a passthrough
// writer (typically stdout) so the user sees the original output.
type RedactingWriter struct {
	redactor    *Redactor
	redacted    io.Writer // receives redacted output (ring buffer)
	passthrough io.Writer // receives unredacted output (stdout)

	mu        sync.Mutex
	redactAll bool // when true, replace all output with mask
}

// NewRedactingWriter creates a writer that redacts secrets before writing to
// the redacted writer, while passing unredacted data to the passthrough writer.
func NewRedactingWriter(redactor *Redactor, redacted, passthrough io.Writer) *RedactingWriter {
	return &RedactingWriter{
		redactor:    redactor,
		redacted:    redacted,
		passthrough: passthrough,
	}
}

// SetRedactAll enables or disables full output redaction (for line-start #+ mode).
func (w *RedactingWriter) SetRedactAll(on bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.redactAll = on
}

// Write writes unredacted data to passthrough, and redacted data to the
// redacted writer.
func (w *RedactingWriter) Write(p []byte) (int, error) {
	n := len(p)

	// Always write unredacted to passthrough (user sees original)
	if w.passthrough != nil {
		w.passthrough.Write(p)
	}

	w.mu.Lock()
	redactAll := w.redactAll
	w.mu.Unlock()

	if redactAll {
		// Replace entire output with mask
		w.redacted.Write([]byte(redactionMask))
		return n, nil
	}

	// Redact known secrets and write to ring buffer
	redacted := w.redactor.Redact(p)
	w.redacted.Write(redacted)

	return n, nil
}
