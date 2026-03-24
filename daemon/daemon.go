package daemon

import (
	"fmt"
	"log"
	"os"
	"os/signal"
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
	log.Println("Initialized log directory: ", internal.LogDir())

	logFile, err := os.OpenFile(internal.LogPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("error opening log file: %v", err)
	}
	log.Println("Initialized log file: ", internal.LogPath())
	log.SetOutput(logFile)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// write PID file
	pidPath := internal.PIDPath()
	if err := WritePID(pidPath); err != nil {
		return fmt.Errorf("error writing to PID path: %v", err)
	}
	log.Println("Initialized PID file: ", pidPath)
	defer RemovePID(pidPath)

	log.Printf("jarvis daemon process starting (pid=%d)", os.Getpid())

	mr := NewModelRegister(cfg, NewServerBackend)
	defer mr.Shutdown()

	// Create handler and server
	stopCh := make(chan struct{}, 1)
	handler := NewHandler(mr, cfg, stopCh)
	server := NewServer(internal.SocketPath(), handler)
	log.Println("Server created: ", server)

	if err := server.Listen(); err != nil {
		return err
	}
	defer server.Close()

	log.Println("Server listening on: ", internal.SocketPath())

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Serve in a goroutine
	errCh := make(chan error, 1)
	go func() {
		errCh <- server.Serve()
	}()

	// Wait for stop signal or serve error
	select {
	case <-stopCh:
		log.Printf("stop requested, shutting down")
	case sig := <-sigCh:
		log.Printf("received signal %v, shutting down", sig)
	case err := <-errCh:
		if err != nil {
			log.Printf("server error: %v", err)
			return err
		}
	}

	server.Close()
	mr.Shutdown()
	log.Printf("daemon stopped")
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
	default:
		return "", fmt.Errorf("invalid split mode %q: must be l(ayer) or r(ow)", mode)
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
