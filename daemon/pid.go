package daemon

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// WritePID writes the current process PID to the given path.
func WritePID(path string) error {
	return os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())), 0644)
}

// ReadPID reads a PID from the given file path.
func ReadPID(path string) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return 0, fmt.Errorf("invalid PID file contents: %w", err)
	}
	return pid, nil
}

// RemovePID removes the PID file at the given path.
func RemovePID(path string) error {
	return os.Remove(path)
}

// IsRunning checks if a process with the PID in the given file is still alive.
// Returns false if the PID file doesn't exist or the process is dead.
func IsRunning(path string) bool {
	pid, err := ReadPID(path)
	if err != nil {
		return false
	}
	return isProcessAlive(pid)
}

// isProcessAlive checks if a process with the given PID exists.
func isProcessAlive(pid int) bool {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	// Signal 0 doesn't send a signal but checks if the process exists.
	err = proc.Signal(syscall.Signal(0))
	return err == nil
}
