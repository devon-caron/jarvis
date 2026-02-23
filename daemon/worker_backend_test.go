package daemon

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestWorkerErrorMsg(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"empty", "", ""},
		{"whitespace only", "   \n  ", ""},
		{"cobra prefix stripped", "Error: failed to load model: no such file", "failed to load model: no such file"},
		{"no prefix unchanged", "some error occurred", "some error occurred"},
		{"multiline takes last", "line one\nError: last error\n", "last error"},
		{"multiline with trailing blank", "line one\nError: last error\n\n  \n", "last error"},
		{"nested wrapping returns last line", "Error: outer\nError: inner detail", "inner detail"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := workerErrorMsg(tc.input)
			if got != tc.want {
				t.Errorf("workerErrorMsg(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// TestWorkerBackend_LoadModel_BadPath verifies that a subprocess exit produces
// a descriptive error rather than the opaque "worker exited before signaling ready".
//
// Uses the standard Go "TestMain as subprocess" pattern:
// the test re-invokes itself as the fake worker via GO_TEST_WORKER=1.
func TestWorkerBackend_LoadModel_BadPath(t *testing.T) {
	if os.Getenv("GO_TEST_WORKER") == "1" {
		// This branch runs in the child process.
		// Simulate a worker that fails to load: write error to stderr and exit non-zero.
		fmt.Fprintln(os.Stderr, "Error: failed to load model: no such file or directory")
		os.Exit(1)
	}

	// Spawn ourselves with the sentinel env var, capture stderr.
	rPipe, wPipe, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestWorkerBackend_LoadModel_BadPath")
	cmd.Env = append(os.Environ(), "GO_TEST_WORKER=1")
	cmd.ExtraFiles = []*os.File{wPipe}

	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	wPipe.Close()

	buf := make([]byte, 1)
	rPipe.Read(buf) // will get EOF since child never writes
	rPipe.Close()

	cmd.Wait()

	got := workerErrorMsg(stderrBuf.String())
	if got == "" {
		t.Error("expected non-empty error message from worker stderr")
	}
	if !strings.Contains(got, "no such file or directory") {
		t.Errorf("expected error to mention the underlying cause, got: %q", got)
	}
}
