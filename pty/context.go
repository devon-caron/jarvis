package pty

import (
	"errors"
	"os"
	"regexp"
	"strings"
)

const maxContextSize = 16 * 1024 // 16KB

// ANSI escape sequence pattern: ESC[ ... final byte, plus OSC sequences
var ansiRe = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b\[[0-9;]*m|\x1b\(B`)

// WriteContext writes data to the context file, creating or truncating it.
func WriteContext(path string, data []byte) error {
	return os.WriteFile(path, data, 0600)
}

// ReadContext reads the context file. Returns empty string if file doesn't exist.
func ReadContext(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	return string(data), nil
}

// StripANSI removes ANSI escape sequences from a string.
func StripANSI(s string) string {
	return ansiRe.ReplaceAllString(s, "")
}

// SanitizeContext strips ANSI escapes, then truncates
// to keep the most recent maxContextSize bytes.
func SanitizeContext(s string) string {
	s = StripANSI(s)
	// Clean up common terminal artifacts
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")

	if len(s) > maxContextSize {
		s = s[len(s)-maxContextSize:]
		// Trim to first complete line
		if idx := strings.Index(s, "\n"); idx >= 0 {
			s = s[idx+1:]
		}
	}
	return s
}
