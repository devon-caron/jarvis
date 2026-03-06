package ptyshell

import (
	"testing"
)

func TestSanitizeInput_PlainPassthrough(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("echo hello world")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "echo hello world" {
		t.Errorf("bash = %q, want %q", bash, "echo hello world")
	}
	if ctx != "echo hello world" {
		t.Errorf("ctx = %q, want %q", ctx, "echo hello world")
	}
}

func TestSanitizeInput_SelectiveRedaction(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("echo #+secret+ world")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "echo secret world" {
		t.Errorf("bash = %q, want %q", bash, "echo secret world")
	}
	if ctx != "echo [REDACTED] world" {
		t.Errorf("ctx = %q, want %q", ctx, "echo [REDACTED] world")
	}
}

func TestSanitizeInput_MultipleSelectiveRedactions(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("cmd #+aaa+ and #+bbb+")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "cmd aaa and bbb" {
		t.Errorf("bash = %q, want %q", bash, "cmd aaa and bbb")
	}
	if ctx != "cmd [REDACTED] and [REDACTED]" {
		t.Errorf("ctx = %q, want %q", ctx, "cmd [REDACTED] and [REDACTED]")
	}
}

func TestSanitizeInput_TailRedaction(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("curl -H #+Bearer tok123")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "curl -H Bearer tok123" {
		t.Errorf("bash = %q, want %q", bash, "curl -H Bearer tok123")
	}
	if ctx != "curl -H [REDACTED]" {
		t.Errorf("ctx = %q, want %q", ctx, "curl -H [REDACTED]")
	}
}

func TestSanitizeInput_FullIORedaction(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("#:cat /etc/passwd")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "cat /etc/passwd" {
		t.Errorf("bash = %q, want %q", bash, "cat /etc/passwd")
	}
	if ctx != "[FULLY REDACTED COMMAND]" {
		t.Errorf("ctx = %q, want %q", ctx, "[FULLY REDACTED COMMAND]")
	}
	if !r.IsFullRedact() {
		t.Error("fullRedact should be true after #: command")
	}
}

func TestSanitizeInput_UIUsurper_Vim(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("vim file.txt")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "vim file.txt" {
		t.Errorf("bash = %q, want %q", bash, "vim file.txt")
	}
	if ctx != "[REDACTED: vim session]" {
		t.Errorf("ctx = %q, want %q", ctx, "[REDACTED: vim session]")
	}
	if !r.IsFullRedact() {
		t.Error("fullRedact should be true for vim")
	}
}

func TestSanitizeInput_UIUsurper_SudoVim(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("sudo vim /etc/hosts")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "sudo vim /etc/hosts" {
		t.Errorf("bash = %q, want %q", bash, "sudo vim /etc/hosts")
	}
	if ctx != "[REDACTED: vim session]" {
		t.Errorf("ctx = %q, want %q", ctx, "[REDACTED: vim session]")
	}
}

func TestSanitizeInput_UIUsurper_FullPath(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("/usr/bin/nano file.txt")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "/usr/bin/nano file.txt" {
		t.Errorf("bash = %q, want %q", bash, "/usr/bin/nano file.txt")
	}
	if ctx != "[REDACTED: nano session]" {
		t.Errorf("ctx = %q, want %q", ctx, "[REDACTED: nano session]")
	}
}

func TestSanitizeInput_UIUsurper_EnvVar(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("TERM=xterm htop")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "TERM=xterm htop" {
		t.Errorf("bash = %q, want %q", bash, "TERM=xterm htop")
	}
	if ctx != "[REDACTED: htop session]" {
		t.Errorf("ctx = %q, want %q", ctx, "[REDACTED: htop session]")
	}
}

func TestSanitizeInput_NotUIUsurper(t *testing.T) {
	r := NewRedactor()
	bash, ctx, err := r.SanitizeInput("ls -la")
	if err != nil {
		t.Fatal(err)
	}
	if bash != "ls -la" {
		t.Errorf("bash = %q, want %q", bash, "ls -la")
	}
	if ctx != "ls -la" {
		t.Errorf("ctx = %q, want %q", ctx, "ls -la")
	}
	if r.IsFullRedact() {
		t.Error("fullRedact should be false for ls")
	}
}

func TestSanitizeOutput_PlainPassthrough(t *testing.T) {
	r := NewRedactor()
	display, ctx, err := r.SanitizeOutput("hello world\n")
	if err != nil {
		t.Fatal(err)
	}
	if display != "hello world\n" {
		t.Errorf("display = %q, want %q", display, "hello world\n")
	}
	if ctx != "hello world\n" {
		t.Errorf("ctx = %q, want %q", ctx, "hello world\n")
	}
}

func TestSanitizeOutput_FullRedactMode(t *testing.T) {
	r := NewRedactor()
	// Trigger full redact via a #: command
	r.SanitizeInput("#:secret-command")

	display, ctx, err := r.SanitizeOutput("sensitive output data\n")
	if err != nil {
		t.Fatal(err)
	}
	if display != "sensitive output data\n" {
		t.Errorf("display = %q, want raw output", display)
	}
	if ctx != "" {
		t.Errorf("ctx = %q, want empty string during full redact", ctx)
	}
}

func TestSanitizeOutput_SecretScrubbing(t *testing.T) {
	r := NewRedactor()
	// First, process input with selective redaction to set active secrets
	r.SanitizeInput("echo #+mysecret+ done")

	display, ctx, err := r.SanitizeOutput("mysecret done\n")
	if err != nil {
		t.Fatal(err)
	}
	if display != "mysecret done\n" {
		t.Errorf("display = %q, want raw output", display)
	}
	if ctx != "[REDACTED] done\n" {
		t.Errorf("ctx = %q, want %q", ctx, "[REDACTED] done\n")
	}
}

func TestSanitizeOutput_AltScreenEnter(t *testing.T) {
	r := NewRedactor()

	display, ctx, err := r.SanitizeOutput("stuff\x1b[?1049hmore stuff")
	if err != nil {
		t.Fatal(err)
	}
	if display != "stuff\x1b[?1049hmore stuff" {
		t.Errorf("display should be raw output")
	}
	if ctx != "" {
		t.Errorf("ctx = %q, want empty during alt screen", ctx)
	}
}

func TestSanitizeOutput_AltScreenLeave(t *testing.T) {
	r := NewRedactor()

	// Enter alt screen
	r.SanitizeOutput("\x1b[?1049h")

	// Should be suppressed
	_, ctx, _ := r.SanitizeOutput("tui garbage")
	if ctx != "" {
		t.Errorf("ctx = %q, want empty during alt screen", ctx)
	}

	// Leave alt screen
	r.SanitizeOutput("\x1b[?1049l")

	// Should work normally again
	_, ctx, _ = r.SanitizeOutput("normal output")
	if ctx != "normal output" {
		t.Errorf("ctx = %q, want %q after leaving alt screen", ctx, "normal output")
	}
}

func TestEndFullRedact(t *testing.T) {
	r := NewRedactor()
	r.SanitizeInput("#:secret")
	if !r.IsFullRedact() {
		t.Error("should be in full redact mode")
	}

	r.EndFullRedact()
	if r.IsFullRedact() {
		t.Error("should not be in full redact mode after EndFullRedact")
	}

	// Output should pass through normally
	_, ctx, _ := r.SanitizeOutput("back to normal\n")
	if ctx != "back to normal\n" {
		t.Errorf("ctx = %q, want %q", ctx, "back to normal\n")
	}
}

func TestExtractCommandName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"vim file.txt", "vim"},
		{"sudo vim file.txt", "vim"},
		{"/usr/bin/nano file.txt", "nano"},
		{"TERM=xterm htop", "htop"},
		{"FOO=bar BAZ=qux less file", "less"},
		{"ls -la", "ls"},
		{"", ""},
	}
	for _, tt := range tests {
		got := extractCommandName(tt.input)
		if got != tt.want {
			t.Errorf("extractCommandName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestParseRedactionMarkers_NoMarkers(t *testing.T) {
	bash, ctx, secrets := parseRedactionMarkers("echo hello")
	if bash != "echo hello" {
		t.Errorf("bash = %q", bash)
	}
	if ctx != "echo hello" {
		t.Errorf("ctx = %q", ctx)
	}
	if len(secrets) != 0 {
		t.Errorf("secrets = %v, want empty", secrets)
	}
}

func TestParseRedactionMarkers_Selective(t *testing.T) {
	bash, ctx, secrets := parseRedactionMarkers("a #+X+ b")
	if bash != "a X b" {
		t.Errorf("bash = %q", bash)
	}
	if ctx != "a [REDACTED] b" {
		t.Errorf("ctx = %q", ctx)
	}
	if len(secrets) != 1 || secrets[0] != "X" {
		t.Errorf("secrets = %v", secrets)
	}
}

func TestParseRedactionMarkers_Tail(t *testing.T) {
	bash, ctx, secrets := parseRedactionMarkers("prefix #+tail content")
	if bash != "prefix tail content" {
		t.Errorf("bash = %q", bash)
	}
	if ctx != "prefix [REDACTED]" {
		t.Errorf("ctx = %q", ctx)
	}
	if len(secrets) != 1 || secrets[0] != "tail content" {
		t.Errorf("secrets = %v", secrets)
	}
}

func TestParseRedactionMarkers_Mixed(t *testing.T) {
	bash, ctx, secrets := parseRedactionMarkers("cmd #+a+ middle #+tail end")
	if bash != "cmd a middle tail end" {
		t.Errorf("bash = %q", bash)
	}
	if ctx != "cmd [REDACTED] middle [REDACTED]" {
		t.Errorf("ctx = %q", ctx)
	}
	if len(secrets) != 2 {
		t.Errorf("secrets = %v, want 2 entries", secrets)
	}
}

func TestSanitizeInput_SecretsResetBetweenCommands(t *testing.T) {
	r := NewRedactor()

	// First command with a secret
	r.SanitizeInput("echo #+secretA+")
	_, ctx, _ := r.SanitizeOutput("secretA\n")
	if ctx != "[REDACTED]\n" {
		t.Errorf("first output: ctx = %q, want %q", ctx, "[REDACTED]\n")
	}

	// Second command without a secret — old secret should not carry over
	r.SanitizeInput("echo hello")
	_, ctx, _ = r.SanitizeOutput("secretA\n")
	if ctx != "secretA\n" {
		t.Errorf("second output: ctx = %q, want %q (old secret should be cleared)", ctx, "secretA\n")
	}
}
