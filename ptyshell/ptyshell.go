package ptyshell

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"

	"github.com/devon-caron/jarvis/internal"
)

// Shell wraps a child shell process in a PTY with sanitized I/O.
// All input flows through the Redactor's SanitizeInput and all output flows
// through SanitizeOutput, ensuring the LLM context buffer never contains
// unsanitized content.
type Shell struct {
	ptmx        *os.File
	childCmd    *exec.Cmd
	redactor    *Redactor
	ctxBuf      *ContextBuffer
	promptDet   *PromptDetector
	origState   *term.State
	shell       string // path to user's shell (e.g., /bin/bash)
	contextPath string // path to context file on disk

	flushMu    sync.Mutex
	lastFlush  time.Time
	flushDirty bool
}

// New creates a new PTY shell. It does not start the shell; call Run() for that.
func New() *Shell {
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh"
	}

	redactor := NewRedactor()
	ctxBuf := NewContextBuffer(1000) // keep last 1000 entries

	var promptDet *PromptDetector
	promptDet = NewPromptDetector(func(cmd string) {
		// When a command completes, process it through the sanitizer
		_, contextStr, err := redactor.SanitizeInput(cmd)
		if err == nil {
			ctxBuf.AppendCommand(contextStr)
		}
		// If the previous command was #: or UI usurper, end full redact
		// now that the prompt has returned.
		// Note: SanitizeInput above may have set fullRedact if this new
		// command is also a #: command, which is correct.
		// EndFullRedact only needs to be called for the *previous* command's
		// scope. We handle this by calling EndFullRedact *before* SanitizeInput
		// for the new command.
		// Actually, the flow is:
		//   1. User runs #:secret-cmd → SanitizeInput sets fullRedact=true
		//   2. Output is suppressed
		//   3. Prompt reappears → OnCommandDone fires
		//   4. We need to end the previous fullRedact, then process the new command
		// So we call EndFullRedact before SanitizeInput. Let's restructure.
	})

	// Restructure the callback: end previous redact, then process new command
	promptDet.OnCommandDone = func(cmd string) {
		redactor.EndFullRedact()
		_, contextStr, err := redactor.SanitizeInput(cmd)
		if err == nil {
			ctxBuf.AppendCommand(contextStr)
		}
	}

	contextPath := internal.ContextPath(os.Getpid())

	return &Shell{
		redactor:    redactor,
		ctxBuf:      ctxBuf,
		promptDet:   promptDet,
		shell:       shell,
		contextPath: contextPath,
	}
}

// Run starts the PTY shell and blocks until the child shell exits.
// It puts the terminal into raw mode, starts the child shell on a PTY,
// and runs the I/O pump goroutines.
func (s *Shell) Run() error {
	// Print startup banner before entering raw mode
	fmt.Println("┌───────────────────────────────────────────────┐")
	fmt.Println("│  jarvis PTY shell active                      │")
	fmt.Println("│  Use redaction markers in prompts:            │")
	fmt.Println("│    ^(.*)#+<secret>+(.*)$  selective redaction │")
	fmt.Println("│    ^(.*)#+<tail>$         tail redaction      │")
	fmt.Println("│    ^#:command+(.*)$       full I/O redaction  │")
	fmt.Println("└───────────────────────────────────────────────┘")

	// Build the child shell command
	cmd := exec.Command(s.shell)
	cmd.Env = append(os.Environ(), s.promptDet.ShellEnv(s.shell)...)

	// Set PTY-mode env vars for child processes
	cmd.Env = append(cmd.Env,
		"JARVIS_PTY=1",
		fmt.Sprintf("JARVIS_CONTEXT_FILE=%s", s.contextPath),
	)

	// Pass through JARVIS_PTY_SYSTEM_PROMPT if set in parent env
	if sp := os.Getenv("JARVIS_PTY_SYSTEM_PROMPT"); sp != "" {
		cmd.Env = append(cmd.Env, fmt.Sprintf("JARVIS_PTY_SYSTEM_PROMPT=%s", sp))
	}

	// For zsh, we need to inject precmd. We do this by appending to ZDOTDIR
	// or by setting an env var that .zshrc can pick up.
	if strings.HasSuffix(s.shell, "zsh") {
		// Set an env var with the precmd snippet that zsh can eval
		cmd.Env = append(cmd.Env,
			fmt.Sprintf("JARVIS_ZSH_PRECMD=%s", s.promptDet.ZshRCSnippet()))
	}

	// Start the child shell on a PTY
	ptmx, err := pty.Start(cmd)
	if err != nil {
		return fmt.Errorf("failed to start PTY: %w", err)
	}
	s.ptmx = ptmx
	s.childCmd = cmd
	defer ptmx.Close()

	// Clean up context file on exit
	defer os.Remove(s.contextPath)

	// Save terminal state and set raw mode
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return fmt.Errorf("failed to set raw mode: %w", err)
	}
	s.origState = oldState
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	// Set initial PTY size to match the real terminal
	s.resizePTY()

	// Handle SIGWINCH to propagate terminal resizes
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGWINCH)
	go func() {
		for range sigCh {
			s.resizePTY()
		}
	}()
	defer signal.Stop(sigCh)
	defer close(sigCh)

	// Start I/O pumps
	errCh := make(chan error, 2)
	go func() { errCh <- s.pumpInput() }()
	go func() { errCh <- s.pumpOutput() }()

	// Wait for the child shell to exit
	cmdErr := cmd.Wait()

	// Final flush of context buffer
	s.flushContextNow()

	// If the child exited, return its error (if any).
	// The I/O pumps will naturally stop when the PTY closes.
	if cmdErr != nil {
		// Exit status 0 means normal exit, anything else is an error
		if exitErr, ok := cmdErr.(*exec.ExitError); ok {
			if exitErr.ExitCode() == 0 {
				return nil
			}
		}
	}
	return cmdErr
}

// ContextBuffer returns the shell's context buffer for external access.
func (s *Shell) ContextBuffer() *ContextBuffer {
	return s.ctxBuf
}

// pumpInput reads from stdin and writes to the PTY master.
// All bytes are forwarded immediately for responsiveness.
// The prompt detector captures input for command reconstruction.
func (s *Shell) pumpInput() error {
	buf := make([]byte, 4096)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		chunk := buf[:n]

		// Forward to PTY immediately (preserves tab completion, arrow keys, etc.)
		if _, werr := s.ptmx.Write(chunk); werr != nil {
			return werr
		}

		// Buffer for command reconstruction
		s.promptDet.FeedInput(chunk)
	}
}

// pumpOutput reads from the PTY master and writes to stdout.
// All output passes through SanitizeOutput before reaching the context buffer.
func (s *Shell) pumpOutput() error {
	buf := make([]byte, 4096)
	for {
		n, err := s.ptmx.Read(buf)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		raw := string(buf[:n])

		// ALL output flows through SanitizeOutput — no exceptions
		displayStr, contextStr, serr := s.redactor.SanitizeOutput(raw)
		if serr != nil {
			// On sanitization error, still display output but skip context
			os.Stdout.Write([]byte(raw))
			continue
		}

		// Write display output to user's terminal
		os.Stdout.Write([]byte(displayStr))

		// Write sanitized output to context buffer
		if contextStr != "" {
			s.ctxBuf.AppendOutput(contextStr)
			s.flushContext()
		}

		// Feed to prompt detector for command boundary detection
		s.promptDet.FeedOutput(buf[:n])
	}
}

// flushContext writes the context buffer to disk, throttled to avoid excessive I/O.
func (s *Shell) flushContext() {
	s.flushMu.Lock()
	defer s.flushMu.Unlock()

	s.flushDirty = true
	if time.Since(s.lastFlush) < 500*time.Millisecond {
		return
	}
	s.doFlush()
}

// flushContextNow forces an immediate flush regardless of throttle.
func (s *Shell) flushContextNow() {
	s.flushMu.Lock()
	defer s.flushMu.Unlock()
	if s.flushDirty {
		s.doFlush()
	}
}

// doFlush writes context buffer contents to the context file. Must be called with flushMu held.
func (s *Shell) doFlush() {
	data := s.ctxBuf.Format()
	// Write atomically via temp file
	tmp := s.contextPath + ".tmp"
	if err := os.WriteFile(tmp, []byte(data), 0600); err != nil {
		return
	}
	os.Rename(tmp, s.contextPath)
	s.lastFlush = time.Now()
	s.flushDirty = false
}

// resizePTY propagates the current terminal size to the PTY.
func (s *Shell) resizePTY() {
	if s.ptmx == nil {
		return
	}
	ws, err := pty.GetsizeFull(os.Stdin)
	if err != nil {
		return
	}
	pty.Setsize(s.ptmx, ws)
}
