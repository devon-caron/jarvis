package ptyshell

import (
	"strings"
	"testing"
)

func TestPromptDetector_MarkerGeneration(t *testing.T) {
	pd := NewPromptDetector(nil)
	marker := pd.Marker()
	if !strings.HasPrefix(marker, "JARVIS_") {
		t.Errorf("marker = %q, want JARVIS_ prefix", marker)
	}
	if len(marker) < 20 {
		t.Errorf("marker too short: %q", marker)
	}
}

func TestPromptDetector_MarkerUniqueness(t *testing.T) {
	pd1 := NewPromptDetector(nil)
	pd2 := NewPromptDetector(nil)
	if pd1.Marker() == pd2.Marker() {
		t.Error("two detectors should have different markers")
	}
}

func TestPromptDetector_SkipsFirstPrompt(t *testing.T) {
	called := false
	pd := NewPromptDetector(func(cmd string) {
		called = true
	})

	// First prompt marker (shell startup) — should be ignored
	marker := "\x01" + pd.Marker() + "\x02"
	pd.FeedOutput([]byte(marker))

	if called {
		t.Error("OnCommandDone should not be called on first prompt")
	}
}

func TestPromptDetector_DetectsCommand(t *testing.T) {
	var captured string
	pd := NewPromptDetector(func(cmd string) {
		captured = cmd
	})

	marker := "\x01" + pd.Marker() + "\x02"

	// Initial prompt (skipped)
	pd.FeedOutput([]byte(marker))

	// User types "ls -la" and presses enter
	pd.FeedInput([]byte("ls -la"))
	pd.FeedInput([]byte{'\r'})

	// Command output followed by next prompt
	pd.FeedOutput([]byte("file1.txt\nfile2.txt\n" + marker))

	if captured != "ls -la" {
		t.Errorf("captured = %q, want %q", captured, "ls -la")
	}
}

func TestPromptDetector_MultipleCommands(t *testing.T) {
	var commands []string
	pd := NewPromptDetector(func(cmd string) {
		commands = append(commands, cmd)
	})

	marker := "\x01" + pd.Marker() + "\x02"

	// Initial prompt
	pd.FeedOutput([]byte(marker))

	// First command
	pd.FeedInput([]byte("echo hello"))
	pd.FeedInput([]byte{'\r'})
	pd.FeedOutput([]byte("hello\n" + marker))

	// Second command
	pd.FeedInput([]byte("pwd"))
	pd.FeedInput([]byte{'\r'})
	pd.FeedOutput([]byte("/home/user\n" + marker))

	if len(commands) != 2 {
		t.Fatalf("got %d commands, want 2", len(commands))
	}
	if commands[0] != "echo hello" {
		t.Errorf("commands[0] = %q", commands[0])
	}
	if commands[1] != "pwd" {
		t.Errorf("commands[1] = %q", commands[1])
	}
}

func TestPromptDetector_BackspaceHandling(t *testing.T) {
	var captured string
	pd := NewPromptDetector(func(cmd string) {
		captured = cmd
	})

	marker := "\x01" + pd.Marker() + "\x02"
	pd.FeedOutput([]byte(marker))

	// Type "lsx", backspace, then "-la"
	pd.FeedInput([]byte("lsx"))
	pd.FeedInput([]byte{127}) // backspace
	pd.FeedInput([]byte(" -la"))
	pd.FeedInput([]byte{'\r'})
	pd.FeedOutput([]byte("output\n" + marker))

	if captured != "ls -la" {
		t.Errorf("captured = %q, want %q", captured, "ls -la")
	}
}

func TestPromptDetector_CtrlCDiscardsInput(t *testing.T) {
	called := false
	pd := NewPromptDetector(func(cmd string) {
		called = true
	})

	marker := "\x01" + pd.Marker() + "\x02"
	pd.FeedOutput([]byte(marker))

	// Type something then Ctrl-C
	pd.FeedInput([]byte("partial command"))
	pd.FeedInput([]byte{3}) // Ctrl-C

	// Next prompt appears
	pd.FeedOutput([]byte(marker))

	if called {
		t.Error("OnCommandDone should not be called after Ctrl-C")
	}
}

func TestPromptDetector_EmptyCommandIgnored(t *testing.T) {
	called := false
	pd := NewPromptDetector(func(cmd string) {
		called = true
	})

	marker := "\x01" + pd.Marker() + "\x02"
	pd.FeedOutput([]byte(marker))

	// Just press enter (empty command)
	pd.FeedInput([]byte{'\r'})
	pd.FeedOutput([]byte(marker))

	if called {
		t.Error("OnCommandDone should not be called for empty command")
	}
}

func TestPromptDetector_SplitMarker(t *testing.T) {
	var captured string
	pd := NewPromptDetector(func(cmd string) {
		captured = cmd
	})

	marker := "\x01" + pd.Marker() + "\x02"
	pd.FeedOutput([]byte(marker))

	pd.FeedInput([]byte("echo test"))
	pd.FeedInput([]byte{'\r'})

	// Marker arrives in two chunks
	half := len(marker) / 2
	pd.FeedOutput([]byte("test\n" + marker[:half]))
	pd.FeedOutput([]byte(marker[half:]))

	if captured != "echo test" {
		t.Errorf("captured = %q, want %q", captured, "echo test")
	}
}

func TestPromptDetector_ShellEnvBash(t *testing.T) {
	pd := NewPromptDetector(nil)
	env := pd.ShellEnv("/bin/bash")

	foundPC := false
	foundMarker := false
	for _, e := range env {
		if strings.HasPrefix(e, "PROMPT_COMMAND=") {
			foundPC = true
			if !strings.Contains(e, pd.Marker()) {
				t.Errorf("PROMPT_COMMAND doesn't contain marker: %s", e)
			}
		}
		if strings.HasPrefix(e, "JARVIS_PROMPT_MARKER=") {
			foundMarker = true
		}
	}
	if !foundPC {
		t.Error("missing PROMPT_COMMAND for bash")
	}
	if !foundMarker {
		t.Error("missing JARVIS_PROMPT_MARKER")
	}
}

func TestPromptDetector_ShellEnvZsh(t *testing.T) {
	pd := NewPromptDetector(nil)
	env := pd.ShellEnv("/bin/zsh")

	foundMarker := false
	for _, e := range env {
		if strings.HasPrefix(e, "PROMPT_COMMAND=") {
			t.Error("zsh should not have PROMPT_COMMAND")
		}
		if strings.HasPrefix(e, "JARVIS_PROMPT_MARKER=") {
			foundMarker = true
		}
	}
	if !foundMarker {
		t.Error("missing JARVIS_PROMPT_MARKER for zsh")
	}
}
