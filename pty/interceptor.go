package pty

import (
	"bytes"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"

	"golang.org/x/term"
)

// Interceptor processes stdin bytes, intercepting #+ redaction syntax before
// forwarding input to the PTY master. It buffers each line and on Enter (\r),
// scans for #+ patterns, resolves secrets, and sends cleaned commands to bash.
type Interceptor struct {
	mu       sync.Mutex
	ptmx     io.Writer        // PTY master fd
	redactor *Redactor        // for registering resolved secrets
	rw       *RedactingWriter // for setting redactAll flag

	lineBuf  bytes.Buffer // current line being typed
	contBuf  bytes.Buffer // continuation buffer for multiline #+content+
	inCont   bool         // in continuation mode (unclosed #+content+)

	// PromptFunc is called to read a hidden password from the terminal.
	// Defaults to term.ReadPassword on fd 0, but can be overridden for testing.
	PromptFunc func(prompt string) (string, error)

	// EvalFunc evaluates a bash expression and returns the result.
	// Defaults to exec.Command("bash", "-c", ...), overridable for testing.
	EvalFunc func(expr string) (string, error)
}

// NewInterceptor creates an Interceptor that writes to ptmx and registers
// secrets with the given redactor.
func NewInterceptor(ptmx io.Writer, redactor *Redactor, rw *RedactingWriter) *Interceptor {
	return &Interceptor{
		ptmx:     ptmx,
		redactor: redactor,
		rw:       rw,
		PromptFunc: func(prompt string) (string, error) {
			fmt.Print(prompt)
			pw, err := term.ReadPassword(0)
			fmt.Println()
			return string(pw), err
		},
		EvalFunc: func(expr string) (string, error) {
			out, err := exec.Command("bash", "-c", fmt.Sprintf("printf '%%s' %s", expr)).CombinedOutput()
			return string(out), err
		},
	}
}

// Write processes input bytes. Each byte is forwarded to ptmx AND buffered.
// On Enter (\r), the line buffer is scanned for #+ patterns.
func (ic *Interceptor) Write(p []byte) (int, error) {
	ic.mu.Lock()
	defer ic.mu.Unlock()

	for _, b := range p {
		switch b {
		case '\r': // Enter
			ic.handleEnter()
		case 3: // Ctrl-C
			ic.lineBuf.Reset()
			ic.contBuf.Reset()
			ic.inCont = false
			ic.ptmx.Write([]byte{b})
		case 127, '\b': // Backspace/Delete
			if ic.lineBuf.Len() > 0 {
				ic.lineBuf.Truncate(ic.lineBuf.Len() - 1)
			}
			ic.ptmx.Write([]byte{b})
		default:
			ic.lineBuf.WriteByte(b)
			if !ic.inCont {
				ic.ptmx.Write([]byte{b})
			}
		}
	}
	return len(p), nil
}

func (ic *Interceptor) handleEnter() {
	line := ic.lineBuf.String()
	ic.lineBuf.Reset()

	// Handle continuation mode
	if ic.inCont {
		ic.contBuf.WriteByte('\n')
		ic.contBuf.WriteString(line)
		full := ic.contBuf.String()
		if strings.Contains(line, "+") {
			// Closing + found in this line
			ic.inCont = false
			ic.contBuf.Reset()
			ic.processLine(full)
			return
		}
		// Still unclosed, show continuation prompt
		fmt.Print("#> ")
		return
	}

	// Check for #+ patterns
	if !strings.Contains(line, "#+") {
		// No markers, just forward the Enter
		ic.ptmx.Write([]byte{'\r'})
		return
	}

	ic.processLine(line)
}

func (ic *Interceptor) processLine(line string) {
	// Check for tail redaction: #+content (no closing +)
	// We need to distinguish from #+content+ (selective)
	if hasTailRedaction(line) {
		ic.handleTailRedaction(line)
		return
	}

	// Check for selective: #+content+
	if hasSelectiveRedaction(line) {
		// Check for unclosed: has #+...but no closing + on this line
		if hasUnclosedSelective(line) {
			ic.inCont = true
			ic.contBuf.WriteString(line)
			// Send Ctrl-U to clear the echoed line from bash
			ic.ptmx.Write([]byte{0x15})
			fmt.Print("#> ")
			return
		}
		ic.handleSelectiveRedaction(line)
		return
	}

	// Fallback: forward as-is
	ic.ptmx.Write([]byte{'\r'})
}

// hasTailRedaction checks if line has #+content without a closing +.
func hasTailRedaction(line string) bool {
	idx := strings.Index(line, "#+")
	if idx < 0 {
		return false
	}
	after := line[idx+2:]
	// If there's no + in the remainder, it's tail redaction
	return !strings.Contains(after, "+")
}

// hasSelectiveRedaction checks if line contains #+content+ (with closing +).
func hasSelectiveRedaction(line string) bool {
	idx := strings.Index(line, "#+")
	if idx < 0 {
		return false
	}
	after := line[idx+2:]
	return strings.Contains(after, "+")
}

// hasUnclosedSelective checks if there's a #+ that isn't closed by + on this line.
// This handles partial #+content where the closing + hasn't been typed yet.
func hasUnclosedSelective(line string) bool {
	rest := line
	for {
		idx := strings.Index(rest, "#+")
		if idx < 0 {
			return false
		}
		after := rest[idx+2:]
		closeIdx := strings.Index(after, "+")
		if closeIdx < 0 {
			return true // Found #+ with no closing +
		}
		rest = after[closeIdx+1:]
	}
}

func (ic *Interceptor) handleTailRedaction(line string) {
	idx := strings.Index(line, "#+")

	// Send Ctrl-U to clear the echoed line
	ic.ptmx.Write([]byte{0x15})

	tail := line[idx+2:]
	prefix := line[:idx]

	// If #+ is at line start, redact all output until next prompt
	if idx == 0 {
		if ic.rw != nil {
			ic.rw.SetRedactAll(true)
		}
	}

	// Best-effort evaluate bash expressions in the tail content
	ic.evaluateAndRedact(tail)

	// Reconstruct clean command (strip #+)
	clean := prefix + tail
	ic.ptmx.Write([]byte(clean))
	ic.ptmx.Write([]byte{'\r'})
}

func (ic *Interceptor) handleSelectiveRedaction(line string) {
	// Send Ctrl-U to clear the echoed line
	ic.ptmx.Write([]byte{0x15})

	var result strings.Builder
	rest := line

	for {
		idx := strings.Index(rest, "#+")
		if idx < 0 {
			result.WriteString(rest)
			break
		}
		result.WriteString(rest[:idx])
		after := rest[idx+2:]

		closeIdx := strings.Index(after, "+")
		if closeIdx < 0 {
			// No closing + (shouldn't happen if hasSelectiveRedaction passed)
			result.WriteString(rest[idx:])
			break
		}

		content := after[:closeIdx]
		rest = after[closeIdx+1:]

		// Resolve the content
		resolved := ic.resolveContent(content)
		if resolved != "" {
			ic.redactor.AddSecret(resolved)
			result.WriteString(resolved)
		}
	}

	// Forward the reconstructed command
	ic.ptmx.Write([]byte(result.String()))
	ic.ptmx.Write([]byte{'\r'})
}

// resolveContent resolves #+content+:
// - If content contains $ → evaluate as bash expression
// - If content is a plain word → prompt for hidden input
// - If content is empty → no-op
func (ic *Interceptor) resolveContent(content string) string {
	if content == "" {
		return ""
	}

	if strings.ContainsAny(content, "$") {
		val, err := ic.EvalFunc(content)
		if err != nil {
			return content
		}
		return val
	}

	// Plain word: prompt for password
	val, err := ic.PromptFunc(fmt.Sprintf("Enter %s: ", content))
	if err != nil {
		return ""
	}
	return val
}

// evaluateAndRedact tries to evaluate bash expressions in content and add
// resolved values to the redactor.
func (ic *Interceptor) evaluateAndRedact(content string) {
	// Find ${...} and $(...) patterns
	for _, prefix := range []string{"${", "$("} {
		rest := content
		for {
			idx := strings.Index(rest, prefix)
			if idx < 0 {
				break
			}
			closer := "}"
			if prefix == "$(" {
				closer = ")"
			}
			end := strings.Index(rest[idx:], closer)
			if end < 0 {
				break
			}
			expr := rest[idx : idx+end+1]
			val, err := ic.EvalFunc(expr)
			if err == nil && val != "" {
				ic.redactor.AddSecret(val)
			}
			rest = rest[idx+end+1:]
		}
	}

	// Also try $VAR patterns (simple variable references)
	for i := 0; i < len(content); i++ {
		if content[i] == '$' && i+1 < len(content) && content[i+1] != '{' && content[i+1] != '(' {
			// Find end of variable name
			j := i + 1
			for j < len(content) && (isAlphaNum(content[j]) || content[j] == '_') {
				j++
			}
			if j > i+1 {
				varExpr := content[i:j]
				val, err := ic.EvalFunc(varExpr)
				if err == nil && val != "" {
					ic.redactor.AddSecret(val)
				}
			}
		}
	}
}

func isAlphaNum(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}
