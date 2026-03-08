package internal

import (
	"fmt"
	"os"
	"path/filepath"
)

// ConfigDir returns the configuration directory path.
// By default, the program uses XDG defaults if available and ~/.config if not.
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

// PIDPath returns the path to the daemon's PID file.
// By default, the program uses XDG defaults if available and the /tmp/ directory if not.
func PIDPath() string {
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, "jarvis.pid")
	}
	return fmt.Sprintf("/tmp/jarvis-%d.pid", os.Getuid())
}

// SocketPath returns the path to the daemon's Unix socket.
// By default, the program uses XDG defaults if available and the /tmp/ directory if not.
func SocketPath() string {
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, "jarvis.sock")
	}
	return fmt.Sprintf("/tmp/jarvis-%d.sock", os.Getuid())
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
