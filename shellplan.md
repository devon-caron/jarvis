# Plan: PTY Shell with Terminal Context Capture and Secret Redaction

## Context

Users want to ask jarvis about terminal failures ("why did that fail?") without copy-pasting output, and need secrets protected from LLM context. By wrapping bash in a PTY, jarvis captures all terminal I/O transparently while redacting sensitive values.

`jarvis start` = PTY shell + daemon (default). `jarvis start -b` = background daemon only. `jarvis stop` stops both.

```
Terminal  <-->  jarvis PTY wrapper  <-->  bash
                     |
                InputInterceptor (handles #+...+ and #+... syntax)
                     |
                RedactingWriter --> RingBuffer (64KB)
                     |
                context file --> jarvis "prompt" --> daemon --> LLM
```

## Secret Redaction Syntax: `#+`

Two forms, distinguished by presence of closing `+`:

### `#+content+` — Selective inline redaction
Redacts just the marked portion. Rest of command visible in context.
```bash
curl -H "Auth: Bearer #+${API_TOKEN}+" https://api.example.com
# context: curl -H "Auth: Bearer *************" https://api.example.com

mysql -u admin -p#+password+
# prompts "Enter password:" (hidden input)
# context: mysql -u admin -p*************
```

**Content detection**: plain word (alphanum/underscore) → prompt for value. Contains `$` → evaluate as bash expression.

### `#+content` — Tail redaction (no closing `+`)
Everything from `#+` to EOL is `*****` in context. Command still executes.
```bash
# Mid-line: partial tail redaction
command #+--secret=${MY_SECRET}
# context: command ***************************

# Line start: full command redaction
#+curl -H "Auth: $TOK" https://secret.internal/data
# context: ***************************
# (command + all output hidden from context)
```

### Continuation
Unclosed `+` at EOL in `#+content+` shows `#> ` and waits for closing `+`, mirroring bash's behavior with unclosed quotes.

### Leak safety
- `#+` mid-word → literal in bash (no execution, no expansion)
- `#+` at line start → `#` makes it a bash comment (ignored entirely)
- `+` is not a bash metacharacter
- Zero conflicts with brace expansion, redirects, command substitution, or any bash feature

## Input Interceptor Design

State machine on stdin byte stream. All input is forwarded to ptmx normally AND buffered in a line buffer. On Enter (`\r`):

1. Scan line buffer for `#+` patterns
2. If no `#+` found: do nothing (command already sent to bash)
3. If `#+content+` found (selective):
   a. Send Ctrl-U (`\x15`) to bash to clear readline buffer
   b. Evaluate content: plain word → prompt with `term.ReadPassword`; bash expr → `bash -c "printf '%s' <content>"` using session env
   c. Add resolved value to Redactor
   d. Reconstruct command with resolved value substituted for `#+content+`
   e. Forward reconstructed command + `\r` to ptmx
4. If `#+content` found (tail, no closing `+`):
   a. Send Ctrl-U to bash
   b. Strip `#+`, forward clean command to ptmx
   c. Evaluate any `${}` / `$()` in content (best effort, exported vars), add resolved values to Redactor
   d. For line-start `#+`: set "redact all output" flag on RedactingWriter until next bash prompt

The Ctrl-U approach means bash's readline gets a fresh command without `#+` markers. The original echo (containing `#+`) is in the ring buffer but gets sanitized out when reading the context.

## Redaction Layers

Three layers ensure no secrets reach the LLM:

1. **Value-based redaction** (RedactingWriter wrapping ring buffer): known secret values → `*************` in real-time as output flows through
2. **Output suppression** (for line-start `#+`): all command output redacted until next prompt
3. **Pattern sanitization** (when reading context for LLM): strip any leaked `#+...+` and `#+...\n` patterns, strip ANSI escapes, truncate to 16KB

## PTY Architecture

- `github.com/creack/pty` for PTY allocation, `golang.org/x/term` for raw mode
- Shell restricted to **bash** (secret evaluation requires bash)
- Output flow: `ptmx → io.MultiWriter(stdout, RedactingWriter(ringBuffer))`
- Input flow: `stdin → Interceptor → ptmx`
- Ring buffer (64KB) flushed to context file periodically (1s)
- Child bash env: `JARVIS_PTY_CONTEXT=/path/to/file`

## Daemon Lifecycle

- **`jarvis start`** (default): start daemon if not running, enter PTY. On shell exit, stop daemon if jarvis started it.
- **`jarvis start -b`**: current behavior — daemon only, background.
- **`jarvis stop`**: stop daemon via socket + SIGTERM PTY process via PID file.

## Signal Handling

| Signal | Behavior |
|--------|----------|
| `SIGWINCH` | Read terminal size, apply to PTY via `pty.Setsize()` |
| `SIGINT` | Not caught — passes through PTY to bash |
| `SIGTERM` / `SIGHUP` | Restore terminal, SIGHUP child bash, clean up |
| `SIGTSTP` | Passes through — bash handles job control |

## Changes

### 1. Dependencies
```
go get github.com/creack/pty@v1.1.24
go get golang.org/x/term@latest
```

### 2. New package `pty/` — 6 source files

| File | Purpose |
|------|---------|
| `pty/ringbuffer.go` | Thread-safe circular byte buffer (io.Writer, 64KB) |
| `pty/redactor.go` | Secret value storage + RedactingWriter (wraps ring buffer, replaces known secrets with `*****`, handles cross-boundary matches) |
| `pty/interceptor.go` | stdin state machine for `#+` syntax (Ctrl-U, prompt, evaluate, reconstruct) |
| `pty/session.go` | PTY lifecycle: spawn bash, raw mode, SIGWINCH, I/O wiring, cleanup |
| `pty/context.go` | Context file read/write, ANSI stripping, `#+` pattern stripping, sanitization |
| `pty/sanitize.go` | (or in context.go) `SanitizeContext()`: StripANSI + StripMarkers + truncate |

### 3. `internal/paths.go`
Add `PTYPIDPath() string`

### 4. `cmd/start.go`
- Add `-b`/`--background` flag
- Extract `startDaemonProcess()` helper
- New `runStartPTY()`: check terminal, start daemon, write PTY PID, create Session, context flush goroutine, `session.Run()`, cleanup

### 5. `cmd/stop.go`
After daemon stop: read PTY PID file, SIGTERM, remove PID file

### 6. `protocol/message.go`
Add `TerminalContext string` to `ChatRequest`

### 7. `cmd/root.go`
In `handleChat()`: read `JARVIS_PTY_CONTEXT` env, load + sanitize context, set on request

### 8. `daemon/handler.go`
In `handleChat()`: if `TerminalContext != ""`, prepend system message with terminal output

## Tests

### `pty/ringbuffer_test.go` (9 tests)
- BasicWrite, ExactCapacity, Overflow, MultipleWrites, WrapAround, ConcurrentAccess, LargeWrite, Reset, IOWriterInterface

### `pty/redactor_test.go` (10 tests)
- SingleSecret, MultipleSecrets, SecretAtBoundaries (start/end/middle), PartialMatchAcrossWrites, OverlappingSecrets, EmptySecret, SpecialCharsInSecret, NoMatch, RedactingWriter_WritesRedacted, RedactingWriter_StdoutUnredacted

### `pty/interceptor_test.go` (16 tests)
- NormalInput_PassThrough (no `#+`, bytes unchanged)
- HashNotFollowedByPlus (`#x` flushed as normal)
- HashPlusAtEOF (buffer held, no crash)
- SelectivePrompt (`cmd --flag=#+password+` triggers prompt)
- SelectiveBashExpr (`cmd --key=#+${VAR}+` evaluates, redacts)
- SelectiveCommand (`cmd --key=#+$(echo hi)+` evaluates command)
- TailRedaction_MidLine (`cmd #+--secret=val` redacts tail)
- TailRedaction_FullLine (`#+full command` at line start)
- TailRedaction_SetsOutputRedactFlag (line-start `#+` suppresses output)
- Continuation_UnmatchedPlus (shows `#> ` prompt)
- MultilineContent (content spans lines via continuation)
- CancelWithCtrlC (resets interceptor state)
- MultipleMarkersOneLine (`cmd #+a+ #+b+`)
- EmptyContent (`#++` treated as no-op)
- SelectiveValueAddedToRedactor (verify value registered)
- TailRedaction_BashExprEvaluated (`#+--key=${VAR}` resolves exported var)

### `pty/context_test.go` (7 tests)
- WriteReadContext_RoundTrip
- ReadContext_NotExist (returns "", nil)
- StripANSI (colors, cursor movement removed)
- StripANSI_PreservesPlainText
- StripRedactionMarkers (`#+leaked+` and `#+leaked\n` → `*****`)
- SanitizeContext_CombinedStripping
- SanitizeContext_KeepsTail (truncation keeps most recent)

### `daemon/handler_test.go` (2 tests)
- Chat_TerminalContext (system message prepended)
- Chat_TerminalContext_Empty (no extra message)

### `cmd/cmd_test.go` (2 tests)
- RunStart_BackgroundFlag (`-b` parsed)
- HandleChat_WithTerminalContext (env + file → request)

## Files Modified

| File | Change |
|------|---------|
| `go.mod` | Add `creack/pty`, `golang.org/x/term` |
| `pty/ringbuffer.go` | **New** |
| `pty/redactor.go` | **New** |
| `pty/interceptor.go` | **New** |
| `pty/session.go` | **New** |
| `pty/context.go` | **New** |
| `pty/*_test.go` | **New** — 42 tests |
| `internal/paths.go` | Add `PTYPIDPath()` |
| `cmd/start.go` | Add `-b` flag, PTY/background split |
| `cmd/stop.go` | Also stop PTY via PID file |
| `protocol/message.go` | Add `TerminalContext` |
| `cmd/root.go` | Read context, sanitize, attach |
| `daemon/handler.go` | Inject terminal context |
| `daemon/handler_test.go` | Terminal context tests |
| `cmd/cmd_test.go` | Background flag + context tests |

## Verification

1. `go build ./...` — compiles
2. `go test ./...` — all pass (42+ new tests)
3. `jarvis start` opens bash identical to normal
4. Failing command → `jarvis "why did that fail?"` explains error
5. `jarvis start -b` starts daemon only
6. `jarvis stop` stops daemon + PTY
7. `exit` in PTY exits cleanly, daemon stops if jarvis started it
8. Terminal resize works (vim, htop)
9. Ctrl-C interrupts normally
10. `mysql -p#+password+` prompts, value redacted from context
11. `cmd --key=#+${API_KEY}+` resolves, redacted from context
12. `command #+--secret=${MY_SECRET}` tail redacted from context
13. `#+curl -H "Auth: $T" https://...` full command + output hidden
14. Unclosed `+` shows `#> ` continuation
