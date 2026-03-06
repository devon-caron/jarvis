package ptyshell

import (
	"strings"
	"testing"
)

func TestContextBuffer_AppendCommand(t *testing.T) {
	cb := NewContextBuffer(100)
	cb.AppendCommand("ls -la")
	if cb.Len() != 1 {
		t.Errorf("Len() = %d, want 1", cb.Len())
	}
	entries := cb.Entries()
	if entries[0].Type != "command" || entries[0].Content != "ls -la" {
		t.Errorf("entry = %+v", entries[0])
	}
}

func TestContextBuffer_AppendOutput_Merge(t *testing.T) {
	cb := NewContextBuffer(100)
	cb.AppendOutput("line1\n")
	cb.AppendOutput("line2\n")
	if cb.Len() != 1 {
		t.Errorf("consecutive outputs should merge: Len() = %d, want 1", cb.Len())
	}
	entries := cb.Entries()
	if entries[0].Content != "line1\nline2\n" {
		t.Errorf("merged content = %q", entries[0].Content)
	}
}

func TestContextBuffer_AppendOutput_NoMergeAfterCommand(t *testing.T) {
	cb := NewContextBuffer(100)
	cb.AppendOutput("out1\n")
	cb.AppendCommand("cmd")
	cb.AppendOutput("out2\n")
	if cb.Len() != 3 {
		t.Errorf("Len() = %d, want 3", cb.Len())
	}
}

func TestContextBuffer_AppendOutput_EmptyString(t *testing.T) {
	cb := NewContextBuffer(100)
	cb.AppendOutput("")
	if cb.Len() != 0 {
		t.Errorf("empty output should not be appended: Len() = %d", cb.Len())
	}
}

func TestContextBuffer_AppendSystem(t *testing.T) {
	cb := NewContextBuffer(100)
	cb.AppendSystem("session started")
	entries := cb.Entries()
	if len(entries) != 1 || entries[0].Type != "system" {
		t.Errorf("entry = %+v", entries)
	}
}

func TestContextBuffer_Trim(t *testing.T) {
	cb := NewContextBuffer(3)
	cb.AppendCommand("cmd1")
	cb.AppendCommand("cmd2")
	cb.AppendCommand("cmd3")
	cb.AppendCommand("cmd4")
	if cb.Len() != 3 {
		t.Errorf("Len() = %d, want 3 after trim", cb.Len())
	}
	entries := cb.Entries()
	if entries[0].Content != "cmd2" {
		t.Errorf("oldest entry should be cmd2, got %q", entries[0].Content)
	}
}

func TestContextBuffer_Format(t *testing.T) {
	cb := NewContextBuffer(100)
	cb.AppendCommand("ls -la")
	cb.AppendOutput("file1.txt\nfile2.txt\n")
	cb.AppendSystem("session started")
	cb.AppendCommand("echo hello")
	cb.AppendOutput("hello")

	formatted := cb.Format()

	if !strings.Contains(formatted, "Terminal session context:") {
		t.Error("missing header")
	}
	if !strings.Contains(formatted, "$ ls -la") {
		t.Error("missing command")
	}
	if !strings.Contains(formatted, "file1.txt\nfile2.txt") {
		t.Error("missing output")
	}
	if !strings.Contains(formatted, "[session started]") {
		t.Error("missing system message")
	}
	if !strings.Contains(formatted, "$ echo hello") {
		t.Error("missing second command")
	}
}

func TestContextBuffer_Entries_ReturnsCopy(t *testing.T) {
	cb := NewContextBuffer(100)
	cb.AppendCommand("cmd1")
	entries := cb.Entries()
	entries[0].Content = "modified"

	// Original should be unchanged
	original := cb.Entries()
	if original[0].Content != "cmd1" {
		t.Error("Entries() should return a copy")
	}
}
