package pty

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"time"

	creackpty "github.com/creack/pty"
	"golang.org/x/term"

	"syscall"
)

// Session manages a PTY-wrapped bash shell with secret redaction and
// terminal context capture.
type Session struct {
	ContextPath string // path to context file for LLM
	RingSize    int    // ring buffer capacity (default: DefaultRingSize)
	FlushInterval time.Duration // how often to flush ring buffer to context file

	ring     *RingBuffer
	redactor *Redactor
	rw       *RedactingWriter
	ic       *Interceptor

	cmd  *exec.Cmd
	ptmx *os.File
}

// NewSession creates a new PTY session that will write context to contextPath.
func NewSession(contextPath string) *Session {
	return &Session{
		ContextPath:   contextPath,
		RingSize:      DefaultRingSize,
		FlushInterval: 1 * time.Second,
	}
}

// Run starts the bash shell in a PTY, wires up I/O with interception and
// redaction, and blocks until the shell exits. The caller's terminal is put
// into raw mode for the duration.
func (s *Session) Run() error {
	// Ensure bash
	bashPath, err := exec.LookPath("bash")
	if err != nil {
		return fmt.Errorf("bash not found: %w", err)
	}

	s.cmd = exec.Command(bashPath)
	s.cmd.Env = append(os.Environ(), "JARVIS_PTY_CONTEXT="+s.ContextPath)

	// Start bash in PTY
	s.ptmx, err = creackpty.Start(s.cmd)
	if err != nil {
		return fmt.Errorf("failed to start PTY: %w", err)
	}
	defer s.ptmx.Close()

	// Set initial PTY size from terminal
	if ws, err := creackpty.GetsizeFull(os.Stdin); err == nil {
		creackpty.Setsize(s.ptmx, ws)
	}

	// Put terminal in raw mode
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return fmt.Errorf("failed to set raw mode: %w", err)
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	// Set up components
	ringSize := s.RingSize
	if ringSize == 0 {
		ringSize = DefaultRingSize
	}
	s.ring = NewRingBuffer(ringSize)
	s.redactor = NewRedactor()
	s.rw = NewRedactingWriter(s.redactor, s.ring, os.Stdout)
	s.ic = NewInterceptor(s.ptmx, s.redactor, s.rw)

	// Handle SIGWINCH for terminal resize
	sigwinch := make(chan os.Signal, 1)
	signal.Notify(sigwinch, syscall.SIGWINCH)
	defer signal.Stop(sigwinch)

	// Handle SIGTERM/SIGHUP for clean shutdown
	sigterm := make(chan os.Signal, 1)
	signal.Notify(sigterm, syscall.SIGTERM, syscall.SIGHUP)
	defer signal.Stop(sigterm)

	// Context flush goroutine
	done := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(s.FlushInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				s.flushContext()
			case <-done:
				s.flushContext() // final flush
				return
			}
		}
	}()

	// Signal handler goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-sigwinch:
				if ws, err := creackpty.GetsizeFull(os.Stdin); err == nil {
					creackpty.Setsize(s.ptmx, ws)
				}
			case sig := <-sigterm:
				// Send SIGHUP to child bash
				if s.cmd.Process != nil {
					s.cmd.Process.Signal(sig)
				}
				return
			case <-done:
				return
			}
		}
	}()

	// PTY output → stdout + ring buffer (via RedactingWriter)
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(s.rw, s.ptmx)
	}()

	// stdin → Interceptor → ptmx
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(s.ic, os.Stdin)
	}()

	// Wait for bash to exit
	err = s.cmd.Wait()
	close(done)
	wg.Wait()

	return err
}

// flushContext writes the ring buffer contents to the context file.
func (s *Session) flushContext() {
	data := s.ring.Bytes()
	if len(data) > 0 {
		WriteContext(s.ContextPath, data)
	}
}
