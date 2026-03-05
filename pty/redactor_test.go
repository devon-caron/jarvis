package pty

import (
	"bytes"
	"testing"
)

func TestRedactor_SingleSecret(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("password123")
	got := string(r.Redact([]byte("my password is password123 ok")))
	want := "my password is " + redactionMask + " ok"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRedactor_MultipleSecrets(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("secret1")
	r.AddSecret("secret2")
	got := string(r.Redact([]byte("secret1 and secret2")))
	want := redactionMask + " and " + redactionMask
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRedactor_SecretAtBoundaries(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("abc")

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"start", "abc def", redactionMask + " def"},
		{"end", "def abc", "def " + redactionMask},
		{"middle", "x abc y", "x " + redactionMask + " y"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := string(r.Redact([]byte(tt.input)))
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestRedactor_PartialMatchAcrossWrites(t *testing.T) {
	// The simple redactor operates per-write, so partial matches across
	// separate Write calls won't be caught. This tests that within a
	// single write the match works.
	r := NewRedactor()
	r.AddSecret("abcdef")
	got := string(r.Redact([]byte("xxabcdefyy")))
	want := "xx" + redactionMask + "yy"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRedactor_OverlappingSecrets(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("abc")
	r.AddSecret("bcd")
	// "abcd" contains overlapping matches; bytes.ReplaceAll processes sequentially
	got := string(r.Redact([]byte("abcd")))
	// After replacing "abc" -> mask, "bcd" pattern is broken
	want := redactionMask + "d"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRedactor_EmptySecret(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("")
	// Empty secret should be ignored
	got := string(r.Redact([]byte("hello")))
	if got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
}

func TestRedactor_SpecialCharsInSecret(t *testing.T) {
	r := NewRedactor()
	r.AddSecret(`p@$$w0rd!#%`)
	got := string(r.Redact([]byte(`auth: p@$$w0rd!#%`)))
	want := "auth: " + redactionMask
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestRedactor_NoMatch(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("xyz")
	input := "no match here"
	got := string(r.Redact([]byte(input)))
	if got != input {
		t.Errorf("got %q, want %q", got, input)
	}
}

func TestRedactingWriter_WritesRedacted(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("secret")
	var redactedBuf bytes.Buffer
	w := NewRedactingWriter(r, &redactedBuf, nil)
	w.Write([]byte("my secret value"))
	got := redactedBuf.String()
	want := "my " + redactionMask + " value"
	if got != want {
		t.Errorf("redacted: got %q, want %q", got, want)
	}
}

func TestRedactingWriter_StdoutUnredacted(t *testing.T) {
	r := NewRedactor()
	r.AddSecret("secret")
	var redactedBuf, stdoutBuf bytes.Buffer
	w := NewRedactingWriter(r, &redactedBuf, &stdoutBuf)
	w.Write([]byte("my secret value"))

	// stdout should see original
	if got := stdoutBuf.String(); got != "my secret value" {
		t.Errorf("stdout: got %q, want original", got)
	}

	// redacted should see mask
	if got := redactedBuf.String(); got != "my "+redactionMask+" value" {
		t.Errorf("redacted: got %q, want masked", got)
	}
}
