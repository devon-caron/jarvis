package internal

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
)

// SocketPath returns the path to the daemon's Unix socket.
// Uses $XDG_RUNTIME_DIR/jarvis.sock if available, otherwise /tmp/jarvis-$UID.sock.
func SocketPath() string {
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, "jarvis.sock")
	}
	return fmt.Sprintf("/tmp/jarvis-%d.sock", os.Getuid())
}

// PIDPath returns the path to the daemon's PID file.
func PIDPath() string {
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, "jarvis.pid")
	}
	return fmt.Sprintf("/tmp/jarvis-%d.pid", os.Getuid())
}

// ConfigDir returns the configuration directory path.
func ConfigDir() string {
	if dir := os.Getenv("XDG_CONFIG_HOME"); dir != "" {
		return filepath.Join(dir, "jarvis")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "jarvis")
}

// ConfigPath returns the path to the config file.
func ConfigPath() string {
	return filepath.Join(ConfigDir(), "config.yaml")
}

// LogDir returns the log directory path.
func LogDir() string {
	if dir := os.Getenv("XDG_DATA_HOME"); dir != "" {
		return filepath.Join(dir, "jarvis")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "jarvis")
}

// LogPath returns the path to the daemon log file.
func LogPath() string {
	return filepath.Join(LogDir(), "daemon.log")
}

// WorkerSocketPath returns the Unix socket path for a named worker subprocess.
func WorkerSocketPath(name string) string {
	safe := regexp.MustCompile(`[^a-zA-Z0-9_-]`).ReplaceAllString(name, "_")
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, "jarvis-worker-"+safe+".sock")
	}
	return fmt.Sprintf("/tmp/jarvis-worker-%s-%d.sock", safe, os.Getuid())
}
