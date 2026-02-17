package internal

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestSocketPath_XDGSet(t *testing.T) {
	t.Setenv("XDG_RUNTIME_DIR", "/run/user/1000")
	got := SocketPath()
	want := "/run/user/1000/jarvis.sock"
	if got != want {
		t.Errorf("SocketPath() = %q, want %q", got, want)
	}
}

func TestSocketPath_Fallback(t *testing.T) {
	t.Setenv("XDG_RUNTIME_DIR", "")
	got := SocketPath()
	want := fmt.Sprintf("/tmp/jarvis-%d.sock", os.Getuid())
	if got != want {
		t.Errorf("SocketPath() = %q, want %q", got, want)
	}
}

func TestPIDPath_XDGSet(t *testing.T) {
	t.Setenv("XDG_RUNTIME_DIR", "/run/user/1000")
	got := PIDPath()
	want := "/run/user/1000/jarvis.pid"
	if got != want {
		t.Errorf("PIDPath() = %q, want %q", got, want)
	}
}

func TestPIDPath_Fallback(t *testing.T) {
	t.Setenv("XDG_RUNTIME_DIR", "")
	got := PIDPath()
	want := fmt.Sprintf("/tmp/jarvis-%d.pid", os.Getuid())
	if got != want {
		t.Errorf("PIDPath() = %q, want %q", got, want)
	}
}

func TestConfigDir_XDGSet(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "/home/test/.myconfig")
	got := ConfigDir()
	want := "/home/test/.myconfig/jarvis"
	if got != want {
		t.Errorf("ConfigDir() = %q, want %q", got, want)
	}
}

func TestConfigDir_Fallback(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "")
	got := ConfigDir()
	home, _ := os.UserHomeDir()
	want := filepath.Join(home, ".config", "jarvis")
	if got != want {
		t.Errorf("ConfigDir() = %q, want %q", got, want)
	}
}

func TestConfigPath(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "/tmp/testconfig")
	got := ConfigPath()
	want := "/tmp/testconfig/jarvis/config.yaml"
	if got != want {
		t.Errorf("ConfigPath() = %q, want %q", got, want)
	}
}

func TestLogDir_XDGSet(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", "/home/test/.mydata")
	got := LogDir()
	want := "/home/test/.mydata/jarvis"
	if got != want {
		t.Errorf("LogDir() = %q, want %q", got, want)
	}
}

func TestLogDir_Fallback(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", "")
	got := LogDir()
	home, _ := os.UserHomeDir()
	want := filepath.Join(home, ".local", "share", "jarvis")
	if got != want {
		t.Errorf("LogDir() = %q, want %q", got, want)
	}
}

func TestLogPath(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", "/tmp/testdata")
	got := LogPath()
	want := "/tmp/testdata/jarvis/daemon.log"
	if got != want {
		t.Errorf("LogPath() = %q, want %q", got, want)
	}
}
