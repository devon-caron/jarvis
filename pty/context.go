package pty

import "os"

// WriteContext writes data to the context file, creating or truncating it.
func WriteContext(path string, data []byte) error {
	return os.WriteFile(path, data, 0600)
}
