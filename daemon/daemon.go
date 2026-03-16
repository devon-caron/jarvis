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
	"github.com/sirupsen/logrus"
)

var logger *logrus.Logger

// Creates PID, Log, etc. files and starts daemon process.
func Run(loggers ...*logrus.Logger) error {
	// load config
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("error loading config: %v", err)
	}

	if len(loggers) > 0 {
		logger = loggers[0]
		// Use the provided logger
		logrus.SetOutput(logger.Out)
		logrus.SetLevel(logger.Level)

		logger.Info("Initialized with provided logger")
	}

	// initialize logger
	if err := os.MkdirAll(internal.LogDir(), 0755); err != nil {
		return fmt.Errorf("error creating log directories: %v", err)
	}
	logger.Info("Initialized log directory: ", internal.LogDir())
	log.Println("Initialized log directory: ", internal.LogDir())
	logFile, err := os.OpenFile(internal.LogPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("error opening log file: %v", err)
	}
	logger.Info("Initialized log file: ", internal.LogPath())
	log.Println("Initialized log file: ", internal.LogPath())
	log.SetOutput(logFile)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// write PID file
	pidPath := internal.PIDPath()
	if err := WritePID(pidPath); err != nil {
		return fmt.Errorf("error writing to PID path: %v", err)
	}
	logger.Info("Initialized PID file: ", pidPath)
	log.Println("Initialized PID file: ", pidPath)
	defer RemovePID(pidPath)

	// Print two sets of logs for debug purposes
	log.Printf("jarvis daemon process starting (pid=%d)", os.Getpid())
	logger.Infof("jarvis daemon process starting (pid=%d)", os.Getpid())

	registry := NewModelRegistry(cfg, NewServerBackend)
	defer registry.Shutdown()

	// Create handler and server
	stopCh := make(chan struct{}, 1)
	handler := NewHandler(registry, cfg, stopCh)
	server := NewServer(internal.SocketPath(), handler)
	log.Println("Server created: ", server)

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
