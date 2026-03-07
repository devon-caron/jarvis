package daemon

import "fmt"

func Run() error {
	return nil
}

func NormalizeSplitMode(mode string) (string, error) {
	if mode == "g" || mode == "graph" {
		return "graph", nil
	}
	if mode == "l" || mode == "layer" {
		return "layer", nil
	}
	if mode == "r" || mode == "row" {
		return "row", nil
	}
	return "", fmt.Errorf("invalid split mode: %s", mode)
}

func IsRunning(pidPath string) bool {
	// TODO: Implement PID file checking
	return false
}

func ReadPID(pidPath string) (int, error) {
	// TODO: Implement PID file reading
	return -1, fmt.Errorf("not implemented")
}
