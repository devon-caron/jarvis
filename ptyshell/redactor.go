package ptyshell

import (
	"strings"
	"sync"
)

// uiUsurpers is the default set of programs that take over the terminal UI
// and should have their I/O fully redacted from LLM context.
var uiUsurpers = map[string]bool{
	"vim": true, "nvim": true, "nano": true, "vi": true,
	"emacs": true, "htop": true, "top": true, "less": true,
	"more": true, "man": true, "tmux": true, "screen": true,
	"mc": true, "ranger": true, "nnn": true,
}

// Redactor manages the sanitization pipeline for terminal I/O.
// It is shared between SanitizeInput and SanitizeOutput so that secrets
// parsed from input can be scrubbed from echoed output.
type Redactor struct {
	mu              sync.Mutex
	activeSecrets   []string // literal strings to scrub from echoed output
	fullRedact      bool     // #: mode or UI usurper active
	altScreenActive bool     // \x1b[?1049h detected in output
}

// NewRedactor creates a Redactor with default settings.
func NewRedactor() *Redactor {
	return &Redactor{}
}

// SanitizeInput parses redaction markers from user input and returns
// the actual bash command to execute and the redacted string for LLM context.
// All input must flow through this function without exception.
//
// Redaction modes:
//   - #:command  — full I/O redaction: command runs but both command and output are censored
//   - #+content+ — selective: content between markers is redacted in context
//   - #+tail     — tail: everything after #+ to end of input is redacted in context
//   - (none)     — passthrough: input is returned unchanged for both bash and context
func (r *Redactor) SanitizeInput(input string) (bashCmd, contextStr string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	trimmed := strings.TrimSpace(input)

	// Mode 3: Full I/O redaction (#:command)
	if strings.HasPrefix(trimmed, "#:") {
		cmd := strings.TrimPrefix(trimmed, "#:")
		r.fullRedact = true
		r.activeSecrets = nil
		return cmd, "[FULLY REDACTED COMMAND]", nil
	}

	// UI usurper detection: auto-promote to full redaction
	cmdName := extractCommandName(trimmed)
	if uiUsurpers[cmdName] {
		r.fullRedact = true
		r.activeSecrets = nil
		return trimmed, "[REDACTED: " + cmdName + " session]", nil
	}

	// Parse #+ markers (selective and tail redaction)
	bashCmd, contextStr, secrets := parseRedactionMarkers(trimmed)

	// Track secrets for output scrubbing
	r.activeSecrets = secrets

	return bashCmd, contextStr, nil
}

// SanitizeOutput processes raw PTY output and returns the display string
// (always the raw output) and the sanitized string for LLM context.
// All output must flow through this function without exception.
func (r *Redactor) SanitizeOutput(raw string) (displayStr, contextStr string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	displayStr = raw

	// Detect alternate screen mode enter/exit
	if strings.Contains(raw, "\x1b[?1049h") {
		r.altScreenActive = true
	}
	if strings.Contains(raw, "\x1b[?1049l") {
		r.altScreenActive = false
	}

	// Full redaction mode: suppress all context
	if r.fullRedact || r.altScreenActive {
		return displayStr, "", nil
	}

	// Scrub known secrets from output (catches echoed input)
	contextStr = raw
	for _, secret := range r.activeSecrets {
		contextStr = strings.ReplaceAll(contextStr, secret, "[REDACTED]")
	}

	return displayStr, contextStr, nil
}

// EndFullRedact is called when the prompt detector identifies that
// a fully-redacted command has completed (next prompt appeared).
// It resets fullRedact and clears active secrets.
func (r *Redactor) EndFullRedact() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fullRedact = false
	r.activeSecrets = nil
}

// IsFullRedact returns whether full redaction mode is currently active.
func (r *Redactor) IsFullRedact() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.fullRedact
}

// SanitizePrompt strips redaction markers from a jarvis CLI prompt argument.
// This is called by cmd/root.go when JARVIS_PTY=1 so that secrets in
// `jarvis "Hello #+secret+"` never reach the daemon unsanitized.
// It returns the sanitized prompt for LLM context (secrets replaced with [REDACTED]).
func SanitizePrompt(prompt string) string {
	_, contextStr, _ := parseRedactionMarkers(prompt)
	return contextStr
}

// extractCommandName returns the base command name from an input line,
// stripping leading env assignments (FOO=bar) and sudo.
func extractCommandName(input string) string {
	fields := strings.Fields(input)
	for _, f := range fields {
		// Skip env var assignments (KEY=VALUE)
		if strings.Contains(f, "=") && !strings.HasPrefix(f, "-") {
			continue
		}
		// Skip sudo
		if f == "sudo" {
			continue
		}
		// Extract basename (e.g., /usr/bin/vim -> vim)
		if idx := strings.LastIndex(f, "/"); idx >= 0 {
			f = f[idx+1:]
		}
		return f
	}
	return ""
}

// parseRedactionMarkers scans input for #+ markers and returns the bash command,
// the context string with redacted content, and a list of secret literals.
func parseRedactionMarkers(input string) (bashCmd, contextStr string, secrets []string) {
	var bashParts, ctxParts []string
	i := 0

	for i < len(input) {
		idx := strings.Index(input[i:], "#+")
		if idx == -1 {
			// No more markers; rest is plain text
			tail := input[i:]
			bashParts = append(bashParts, tail)
			ctxParts = append(ctxParts, tail)
			break
		}

		// Add text before marker
		before := input[i : i+idx]
		bashParts = append(bashParts, before)
		ctxParts = append(ctxParts, before)
		i += idx + 2 // skip #+

		// Check for closing +
		closeIdx := strings.Index(input[i:], "+")
		if closeIdx == -1 {
			// Tail mode: everything after #+ is redacted
			secret := input[i:]
			bashParts = append(bashParts, secret)
			ctxParts = append(ctxParts, "[REDACTED]")
			secrets = append(secrets, secret)
			i = len(input)
			break
		}

		// Selective mode: content between #+ and +
		secret := input[i : i+closeIdx]
		bashParts = append(bashParts, secret)
		ctxParts = append(ctxParts, "[REDACTED]")
		secrets = append(secrets, secret)
		i += closeIdx + 1 // skip closing +
	}

	return strings.Join(bashParts, ""), strings.Join(ctxParts, ""), secrets
}
