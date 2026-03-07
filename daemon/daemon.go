package daemon

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
)

func Run() error {
	return nil
}

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

func IsRunning(pidPath string) bool {
	pid, err := ReadPID(pidPath)
	if err != nil {
		return false
	}

	return doesProcessExist(pid)
}

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

func doesProcessExist(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}

	err = process.Signal(syscall.Signal(0))
	return err == nil
}
