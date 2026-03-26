package pty

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"
)

type Session struct {
	ContextPath   string
	RingSize      int
	FlushInterval time.Duration

	ring *RingBuffer

	cmd  *exec.Cmd
	ptmx *os.File // PTY master file descriptor (pseudo terminal master multiplexer)
}

func NewSession(contextPath string) *Session {
	return &Session{
		ContextPath:   contextPath,
		RingSize:      DefaultRingSize,
		FlushInterval: 1 * time.Second,
	}
}

func (s *Session) Run() error {
	bashPath, err := exec.LookPath("bash")
	if err != nil {
		return err
	}

	s.cmd = exec.Command(bashPath)
	s.cmd.Env = append(os.Environ(), "JARVIS_CONTEXT_PATH="+s.ContextPath)

	s.ptmx, err = pty.Start(s.cmd)
	if err != nil {
		return fmt.Errorf("failed to start PTY: %w", err)
	}
	defer s.ptmx.Close()

	if winSize, err := pty.GetsizeFull(s.ptmx); err == nil {
		pty.Setsize(s.ptmx, winSize)
	}

	// put terminal in raw mode
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return fmt.Errorf("failed to make terminal raw: %w", err)
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	// Set up components
	ringSize := s.RingSize
	if ringSize == 0 {
		ringSize = DefaultRingSize
	}
	s.ring = NewRingBuffer(ringSize)

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
				if ws, err := pty.GetsizeFull(os.Stdin); err == nil {
					pty.Setsize(s.ptmx, ws)
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

	// output: ptmx -> stdout -> ringbuffer
	wg.Add(1)
	go func() {
		defer wg.Done()
		w := io.MultiWriter(os.Stdout, s.ring)
		io.Copy(w, s.ptmx)
	}()

	// input: stdin -> ptmx
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(s.ptmx, os.Stdin)
	}()

	// cleanup
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
