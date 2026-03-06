package ptyshell

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
)

// PromptDetector tracks command boundaries by injecting a unique marker
// into the shell prompt and watching for it in PTY output.
//
// Flow:
//  1. The child shell's PROMPT_COMMAND (bash) or precmd (zsh) emits the marker.
//  2. FeedInput() captures keystrokes to reconstruct the command line.
//  3. FeedOutput() watches for the marker to detect command completion.
//  4. When a command completes, OnCommandDone is called with the reconstructed command.
type PromptDetector struct {
	mu            sync.Mutex
	marker        string           // unique string injected into prompt
	inputBuf      strings.Builder  // accumulates typed input between prompts
	outputBuf     strings.Builder  // accumulates output for marker scanning
	OnCommandDone func(cmd string) // callback when a command finishes
	seenFirstPrompt bool           // ignore the initial prompt on shell startup
	hasCommand    bool             // whether we have a command buffered
}

// NewPromptDetector creates a PromptDetector with a random unique marker.
func NewPromptDetector(onCommandDone func(cmd string)) *PromptDetector {
	return &PromptDetector{
		marker:        generateMarker(),
		OnCommandDone: onCommandDone,
	}
}

// Marker returns the unique prompt marker string.
func (pd *PromptDetector) Marker() string {
	return pd.marker
}

// ShellEnv returns environment variables to inject into the child shell
// for prompt marker emission.
//
// For bash: PROMPT_COMMAND prints the marker after each command.
// For zsh: precmd function prints the marker before each prompt.
// The marker is emitted via printf to avoid newline issues.
func (pd *PromptDetector) ShellEnv(shell string) []string {
	marker := pd.Marker()

	// Use a non-printable wrapper so the marker doesn't show in the terminal.
	// We use \x01 and \x02 (readline ignore markers) to wrap the marker.
	printCmd := fmt.Sprintf(`printf '\x01%s\x02'`, marker)

	switch {
	case strings.HasSuffix(shell, "zsh"):
		// For zsh, set precmd via environment
		return []string{
			fmt.Sprintf("JARVIS_PROMPT_MARKER=%s", marker),
		}
	default:
		// For bash and other shells, use PROMPT_COMMAND
		return []string{
			fmt.Sprintf("PROMPT_COMMAND=%s", printCmd),
			fmt.Sprintf("JARVIS_PROMPT_MARKER=%s", marker),
		}
	}
}

// ZshRCSnippet returns a zsh snippet to source for precmd integration.
// This should be injected via ZDOTDIR or sourced from the shell env.
func (pd *PromptDetector) ZshRCSnippet() string {
	return fmt.Sprintf(`precmd() { printf '\01%s\02'; }`, pd.marker)
}

// FeedInput records raw keystrokes from the user for command reconstruction.
func (pd *PromptDetector) FeedInput(chunk []byte) {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	for _, b := range chunk {
		switch b {
		case '\r', '\n':
			// Enter pressed — mark that we have a pending command
			pd.hasCommand = true
		case 127, '\b':
			// Backspace — remove last character from buffer
			s := pd.inputBuf.String()
			if len(s) > 0 {
				pd.inputBuf.Reset()
				pd.inputBuf.WriteString(s[:len(s)-1])
			}
		case 3:
			// Ctrl-C — discard current input
			pd.inputBuf.Reset()
			pd.hasCommand = false
		default:
			if b >= 32 { // printable characters only
				pd.inputBuf.WriteByte(b)
			}
		}
	}
}

// FeedOutput scans PTY output for the prompt marker to detect command completion.
func (pd *PromptDetector) FeedOutput(chunk []byte) {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	pd.outputBuf.Write(chunk)

	// Scan for marker wrapped in \x01...\x02
	wrappedMarker := "\x01" + pd.marker + "\x02"
	output := pd.outputBuf.String()

	for {
		idx := strings.Index(output, wrappedMarker)
		if idx == -1 {
			// Keep only the tail that might contain a partial marker
			if len(output) > len(wrappedMarker) {
				keepFrom := len(output) - len(wrappedMarker)
				pd.outputBuf.Reset()
				pd.outputBuf.WriteString(output[keepFrom:])
			}
			return
		}

		// Found the marker
		if !pd.seenFirstPrompt {
			// Skip the initial prompt on shell startup
			pd.seenFirstPrompt = true
		} else if pd.hasCommand {
			// A command has completed
			cmd := strings.TrimSpace(pd.inputBuf.String())
			if cmd != "" && pd.OnCommandDone != nil {
				pd.OnCommandDone(cmd)
			}
		}

		// Reset for next command
		pd.inputBuf.Reset()
		pd.hasCommand = false

		// Continue scanning the rest of the output
		output = output[idx+len(wrappedMarker):]
		pd.outputBuf.Reset()
		pd.outputBuf.WriteString(output)
	}
}

// generateMarker creates a random hex string for use as a prompt marker.
func generateMarker() string {
	b := make([]byte, 16)
	rand.Read(b)
	return "JARVIS_" + hex.EncodeToString(b)
}
