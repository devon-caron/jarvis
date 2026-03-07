package internal

import (
	"fmt"
	"os"
	"path/filepath"
)

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

// PIDPath returns the path to the daemon's PID file.
func PIDPath() string {
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, "jarvis.pid")
	}
	return fmt.Sprintf("/tmp/jarvis-%d.pid", os.Getuid())
}
