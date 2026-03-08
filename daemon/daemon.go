package daemon

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/internal"
)

// Creates PID, Log, etc. files and starts daemon process.
func Run() error {
	// load config
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("error loading config: %v", err)
	}

	// initialize logger
	if err := os.MkdirAll(internal.LogDir(), 0755); err != nil {
		return fmt.Errorf("error creating log directories: %v", err)
	}
	logFile, err := os.OpenFile(internal.LogPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("error opening log file: %v", err)
	}
	log.SetOutput(logFile)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// write PID file
	pidPath := internal.PIDPath()
	if err := WritePID(pidPath); err != nil {
		return fmt.Errorf("error writing to PID path: %v", err)
	}
	defer RemovePID(pidPath)

	log.Printf("jarvis daemon process starting (pid=%d)", os.Getpid())

	registry := NewModelRegistry(cfg, NewServerBackend)
	defer registry.Shutdown()

	return nil
}

func WritePID(path string) error {
	return os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())), 0644)
}

func RemovePID(path string) error {
	return os.Remove(path)
}

// Returns the full string for shorthand splitmode flags
func NormalizeSplitMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "none":
		return "", nil
	case "l", "layer":
		return "layer", nil
	case "r", "row":
		return "row", nil
	case "g", "graph":
		return "graph", nil
	default:
		return "", fmt.Errorf("invalid split mode %q: must be l(ayer), r(ow), or g(raph)", mode)
	}
}

// checks running daemon status
func IsRunning(pidPath string) bool {
	pid, err := ReadPID(pidPath)
	if err != nil {
		return false
	}

	return doesProcessExist(pid)
}

// retrieves the daemon's PID
func ReadPID(pidPath string) (int, error) {
	data, err := os.ReadFile(pidPath)
	if err != nil {
		return -1, fmt.Errorf("error reading pid file: %v", err)
	}
	pid, err := strconv.Atoi(string(data))
	if err != nil {
		return -1, fmt.Errorf("pid file data error: %v", err)
	}

	return pid, nil
}

// helper that checks for daemon process existence
func doesProcessExist(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}

	err = process.Signal(syscall.Signal(0))
	return err == nil
}
