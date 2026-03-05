package pty

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteReadContext_RoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ctx")
	data := "$ ls\nfile1 file2\n$ echo hello\nhello\n"
	if err := WriteContext(path, []byte(data)); err != nil {
		t.Fatal(err)
	}
	got, err := ReadContext(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != data {
		t.Errorf("got %q, want %q", got, data)
	}
}

func TestReadContext_NotExist(t *testing.T) {
	got, err := ReadContext(filepath.Join(t.TempDir(), "nope"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

func TestStripANSI(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"color", "\x1b[31mred\x1b[0m", "red"},
		{"cursor", "\x1b[2Jcleared", "cleared"},
		{"bold", "\x1b[1mbold\x1b[0m text", "bold text"},
		{"multiple", "\x1b[32m\x1b[1mgreen bold\x1b[0m", "green bold"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StripANSI(tt.input)
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestStripANSI_PreservesPlainText(t *testing.T) {
	input := "just plain text with symbols #$%^&*()"
	got := StripANSI(input)
	if got != input {
		t.Errorf("got %q, want %q", got, input)
	}
}

func TestStripRedactionMarkers(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"selective", "cmd #+leaked+ arg", "cmd " + redactionMask + " arg"},
		{"tail", "cmd #+leaked\nnext", "cmd " + redactionMask + "\nnext"},
		{"both", "#+a+ and #+b\n", redactionMask + " and " + redactionMask + "\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StripRedactionMarkers(tt.input)
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSanitizeContext_CombinedStripping(t *testing.T) {
	input := "\x1b[31m$ cmd #+secret+\x1b[0m\noutput\n"
	got := SanitizeContext(input)
	if strings.Contains(got, "\x1b") {
		t.Error("ANSI escapes not stripped")
	}
	if strings.Contains(got, "#+") {
		t.Error("redaction markers not stripped")
	}
	if !strings.Contains(got, "output") {
		t.Error("expected 'output' to be preserved")
	}
}

func TestSanitizeContext_KeepsTail(t *testing.T) {
	// Build a string larger than maxContextSize
	line := "line of text here\n"
	var b strings.Builder
	for b.Len() < maxContextSize+1000 {
		b.WriteString(line)
	}
	full := b.String()
	got := SanitizeContext(full)
	if len(got) > maxContextSize {
		t.Errorf("len = %d, want <= %d", len(got), maxContextSize)
	}
	// Should end with the same content as the original (tail preserved)
	if !strings.HasSuffix(full, got[len(got)-100:]) {
		t.Error("tail not preserved after truncation")
	}
}
