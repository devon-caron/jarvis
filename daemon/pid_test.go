package daemon

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestWriteAndReadPID(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")

	err := WritePID(path)
	if err != nil {
		t.Fatalf("WritePID: %v", err)
	}

	pid, err := ReadPID(path)
	if err != nil {
		t.Fatalf("ReadPID: %v", err)
	}

	if pid != os.Getpid() {
		t.Errorf("ReadPID = %d, want %d", pid, os.Getpid())
	}
}

func TestReadPID_MissingFile(t *testing.T) {
	_, err := ReadPID("/nonexistent/path/test.pid")
	if err == nil {
		t.Error("expected error for missing PID file")
	}
}

func TestReadPID_InvalidContents(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")
	os.WriteFile(path, []byte("not-a-number"), 0644)

	_, err := ReadPID(path)
	if err == nil {
		t.Error("expected error for invalid PID contents")
	}
}

func TestReadPID_WithWhitespace(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")
	os.WriteFile(path, []byte("  12345  \n"), 0644)

	pid, err := ReadPID(path)
	if err != nil {
		t.Fatalf("ReadPID: %v", err)
	}
	if pid != 12345 {
		t.Errorf("ReadPID = %d, want 12345", pid)
	}
}

func TestRemovePID(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")
	os.WriteFile(path, []byte("12345"), 0644)

	err := RemovePID(path)
	if err != nil {
		t.Fatalf("RemovePID: %v", err)
	}

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("PID file should be removed")
	}
}

func TestIsRunning_NoFile(t *testing.T) {
	if IsRunning("/nonexistent/path/test.pid") {
		t.Error("IsRunning should return false for missing PID file")
	}
}

func TestIsRunning_CurrentProcess(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")
	os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())), 0644)

	if !IsRunning(path) {
		t.Error("IsRunning should return true for current process")
	}
}

func TestIsRunning_DeadProcess(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")
	// Use a very high PID that almost certainly doesn't exist
	os.WriteFile(path, []byte("4999999"), 0644)

	if IsRunning(path) {
		t.Error("IsRunning should return false for dead process")
	}
}

func TestIsProcessAlive_Self(t *testing.T) {
	if !isProcessAlive(os.Getpid()) {
		t.Error("isProcessAlive should return true for current process")
	}
}

func TestIsProcessAlive_Dead(t *testing.T) {
	if isProcessAlive(4999999) {
		t.Error("isProcessAlive should return false for non-existent PID")
	}
}
