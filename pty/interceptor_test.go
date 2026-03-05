package pty

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

// newTestInterceptor creates an interceptor with mock funcs for testing.
func newTestInterceptor(env map[string]string) (*Interceptor, *bytes.Buffer, *Redactor) {
	var ptmxBuf bytes.Buffer
	redactor := NewRedactor()
	rw := NewRedactingWriter(redactor, &bytes.Buffer{}, nil)

	ic := NewInterceptor(&ptmxBuf, redactor, rw)
	ic.PromptFunc = func(prompt string) (string, error) {
		// Extract the variable name from "Enter <name>: "
		name := strings.TrimPrefix(prompt, "Enter ")
		name = strings.TrimSuffix(name, ": ")
		if val, ok := env[name]; ok {
			return val, nil
		}
		return "prompted_value", nil
	}
	ic.EvalFunc = func(expr string) (string, error) {
		// Simple mock: resolve ${VAR} and $VAR from env map
		cleaned := strings.TrimPrefix(expr, "${")
		cleaned = strings.TrimSuffix(cleaned, "}")
		cleaned = strings.TrimPrefix(cleaned, "$")
		cleaned = strings.TrimPrefix(cleaned, "(echo ")
		cleaned = strings.TrimSuffix(cleaned, ")")
		if val, ok := env[cleaned]; ok {
			return val, nil
		}
		// For $(echo hi) style
		if strings.HasPrefix(expr, "$(") && strings.HasSuffix(expr, ")") {
			inner := expr[2 : len(expr)-1]
			if strings.HasPrefix(inner, "echo ") {
				return strings.TrimPrefix(inner, "echo "), nil
			}
		}
		return expr, nil
	}

	return ic, &ptmxBuf, redactor
}

func TestInterceptor_NormalInput_PassThrough(t *testing.T) {
	ic, ptmxBuf, _ := newTestInterceptor(nil)
	ic.Write([]byte("ls -la\r"))
	got := ptmxBuf.String()
	if got != "ls -la\r" {
		t.Errorf("got %q, want %q", got, "ls -la\r")
	}
}

func TestInterceptor_HashNotFollowedByPlus(t *testing.T) {
	ic, ptmxBuf, _ := newTestInterceptor(nil)
	ic.Write([]byte("#x\r"))
	got := ptmxBuf.String()
	if got != "#x\r" {
		t.Errorf("got %q, want %q", got, "#x\r")
	}
}

func TestInterceptor_HashPlusAtEOF(t *testing.T) {
	ic, ptmxBuf, _ := newTestInterceptor(nil)
	// Write #+ without Enter - should just buffer
	ic.Write([]byte("cmd #+"))
	// Only the regular chars should be in ptmx (no Enter processed)
	got := ptmxBuf.String()
	if got != "cmd #+" {
		t.Errorf("got %q, want %q", got, "cmd #+")
	}
}

func TestInterceptor_SelectivePrompt(t *testing.T) {
	env := map[string]string{"password": "s3cret"}
	ic, ptmxBuf, redactor := newTestInterceptor(env)

	ic.Write([]byte("cmd --flag=#+password+\r"))
	got := ptmxBuf.String()

	// Should contain Ctrl-U (0x15) to clear line, then reconstructed command
	if !strings.Contains(got, "\x15") {
		t.Error("expected Ctrl-U to clear line")
	}
	if !strings.Contains(got, "cmd --flag=s3cret\r") {
		t.Errorf("expected reconstructed command with resolved value, got %q", got)
	}
	// Secret should be registered
	redacted := string(redactor.Redact([]byte("s3cret")))
	if redacted != redactionMask {
		t.Errorf("secret not registered: %q", redacted)
	}
}

func TestInterceptor_SelectiveBashExpr(t *testing.T) {
	env := map[string]string{"VAR": "resolved_value"}
	ic, ptmxBuf, _ := newTestInterceptor(env)

	ic.Write([]byte("cmd --key=#+${VAR}+\r"))
	got := ptmxBuf.String()

	if !strings.Contains(got, "cmd --key=resolved_value\r") {
		t.Errorf("expected resolved bash expr, got %q", got)
	}
}

func TestInterceptor_SelectiveCommand(t *testing.T) {
	env := map[string]string{"hi": "hi"}
	ic, ptmxBuf, _ := newTestInterceptor(env)

	ic.Write([]byte("cmd --key=#+$(echo hi)+\r"))
	got := ptmxBuf.String()

	if !strings.Contains(got, "cmd --key=hi\r") {
		t.Errorf("expected command result, got %q", got)
	}
}

func TestInterceptor_TailRedaction_MidLine(t *testing.T) {
	env := map[string]string{"MY_SECRET": "top_secret"}
	ic, ptmxBuf, _ := newTestInterceptor(env)

	ic.Write([]byte("command #+--secret=${MY_SECRET}\r"))
	got := ptmxBuf.String()

	// Should contain Ctrl-U then clean command
	if !strings.Contains(got, "\x15") {
		t.Error("expected Ctrl-U")
	}
	if !strings.Contains(got, "command --secret=${MY_SECRET}\r") {
		t.Errorf("expected clean command, got %q", got)
	}
}

func TestInterceptor_TailRedaction_FullLine(t *testing.T) {
	ic, ptmxBuf, _ := newTestInterceptor(nil)

	ic.Write([]byte("#+full secret command\r"))
	got := ptmxBuf.String()

	if !strings.Contains(got, "\x15") {
		t.Error("expected Ctrl-U")
	}
	// The clean command (without #+) should be sent
	if !strings.Contains(got, "full secret command\r") {
		t.Errorf("expected clean command, got %q", got)
	}
}

func TestInterceptor_TailRedaction_SetsOutputRedactFlag(t *testing.T) {
	var ptmxBuf bytes.Buffer
	redactor := NewRedactor()
	var redactedBuf bytes.Buffer
	rw := NewRedactingWriter(redactor, &redactedBuf, nil)
	ic := NewInterceptor(&ptmxBuf, redactor, rw)
	ic.EvalFunc = func(expr string) (string, error) { return "", nil }

	// Line-start #+ should set redactAll
	ic.Write([]byte("#+hidden command\r"))

	// Now write some "output" through the RedactingWriter
	rw.Write([]byte("sensitive output"))
	got := redactedBuf.String()
	if got != redactionMask {
		t.Errorf("expected all output redacted, got %q", got)
	}
}

func TestInterceptor_Continuation_UnmatchedPlus(t *testing.T) {
	ic, ptmxBuf, _ := newTestInterceptor(map[string]string{"password": "pw123"})

	// Type a line with unclosed #+...
	ic.Write([]byte("cmd #+password"))
	// The characters are forwarded normally
	if !strings.Contains(ptmxBuf.String(), "cmd #+password") {
		t.Errorf("expected chars forwarded, got %q", ptmxBuf.String())
	}

	// Press Enter - should enter continuation mode
	ptmxBuf.Reset()
	ic.Write([]byte("\r"))
	// Should send Ctrl-U and show #> prompt (via fmt.Print)
	if !strings.Contains(ptmxBuf.String(), "\x15") {
		t.Error("expected Ctrl-U on continuation")
	}
}

func TestInterceptor_MultilineContent(t *testing.T) {
	env := map[string]string{"password": "pw123"}
	ic, ptmxBuf, _ := newTestInterceptor(env)

	// First line: unclosed #+password (no closing +)
	// But this looks like tail redaction... We need #+password with intention to close
	// For continuation, we need #+content+ where content spans multiple lines
	// This means: "cmd #+pass" on one line, "word+" on the next
	ic.Write([]byte("cmd #+pass\r"))
	// In continuation mode now
	ptmxBuf.Reset()

	ic.Write([]byte("word+\r"))
	got := ptmxBuf.String()
	// Should have resolved "pass\nword" as the content
	// Since it contains no $, it should prompt
	if !strings.Contains(got, "\r") {
		t.Errorf("expected command sent, got %q", got)
	}
}

func TestInterceptor_CancelWithCtrlC(t *testing.T) {
	ic, ptmxBuf, _ := newTestInterceptor(nil)

	// Start typing a #+ pattern
	ic.Write([]byte("cmd #+sec"))

	// Ctrl-C should reset state
	ptmxBuf.Reset()
	ic.Write([]byte{3}) // Ctrl-C

	// Verify state is reset by typing a normal command
	ptmxBuf.Reset()
	ic.Write([]byte("ls\r"))
	got := ptmxBuf.String()
	if got != "ls\r" {
		t.Errorf("expected normal passthrough after Ctrl-C, got %q", got)
	}
}

func TestInterceptor_MultipleMarkersOneLine(t *testing.T) {
	env := map[string]string{"a": "val_a", "b": "val_b"}
	ic, ptmxBuf, _ := newTestInterceptor(env)

	ic.Write([]byte("cmd #+a+ #+b+\r"))
	got := ptmxBuf.String()

	if !strings.Contains(got, "cmd val_a val_b\r") {
		t.Errorf("expected both resolved, got %q", got)
	}
}

func TestInterceptor_EmptyContent(t *testing.T) {
	ic, ptmxBuf, _ := newTestInterceptor(nil)

	ic.Write([]byte("cmd #++\r"))
	got := ptmxBuf.String()

	// Empty content should be a no-op for the marker
	if !strings.Contains(got, "cmd \r") {
		t.Errorf("expected empty content removed, got %q", got)
	}
}

func TestInterceptor_SelectiveValueAddedToRedactor(t *testing.T) {
	env := map[string]string{"token": "abc123"}
	ic, _, redactor := newTestInterceptor(env)

	ic.Write([]byte("curl -H #+token+\r"))

	// Verify secret was registered
	got := string(redactor.Redact([]byte("abc123")))
	if got != redactionMask {
		t.Errorf("expected secret registered, got %q", got)
	}
}

func TestInterceptor_TailRedaction_BashExprEvaluated(t *testing.T) {
	env := map[string]string{"API_KEY": "key123"}
	ic, _, redactor := newTestInterceptor(env)
	// Override EvalFunc to handle the specific expression
	ic.EvalFunc = func(expr string) (string, error) {
		if expr == "${API_KEY}" {
			return "key123", nil
		}
		cleaned := strings.TrimPrefix(expr, "$")
		if val, ok := env[cleaned]; ok {
			return val, nil
		}
		return "", fmt.Errorf("unknown: %s", expr)
	}

	ic.Write([]byte("cmd #+--key=${API_KEY}\r"))

	// The resolved value should be redacted
	got := string(redactor.Redact([]byte("key123")))
	if got != redactionMask {
		t.Errorf("expected resolved value redacted, got %q", got)
	}
}
