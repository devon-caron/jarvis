# Testing

## Prerequisites

Tests that link against llama-go require the `libllama.so` shared library at runtime.
Set `LD_LIBRARY_PATH` to the directory containing it before running tests:

```bash
export LD_LIBRARY_PATH=/path/to/llama-go/build/bin
```

Without this, `daemon` and `cmd` tests will fail with:

```
error while loading shared libraries: libllama.so.0: cannot open shared object file
```

Packages that do not link against llama-go (`config`, `protocol`, `internal`, `search`, `client`) run without it.

## Running Tests

### All packages

```bash
LD_LIBRARY_PATH=/path/to/llama-go/build/bin go test ./...
```

### Single package

```bash
LD_LIBRARY_PATH=/path/to/llama-go/build/bin go test ./daemon
```

### Verbose output

```bash
LD_LIBRARY_PATH=/path/to/llama-go/build/bin go test -v ./...
```

### With coverage

```bash
LD_LIBRARY_PATH=/path/to/llama-go/build/bin go test -cover ./...
```

### Coverage report (HTML)

```bash
LD_LIBRARY_PATH=/path/to/llama-go/build/bin go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

## Test Architecture

### Mock backends

Tests use a `mockBackend` that implements the `ModelBackend` interface without requiring a real GPU or model file. This allows testing the full `ModelRegistry`, `ModelSlot`, and `Handler` logic in isolation.

The mock supports configurable:
- Load/unload/chat errors
- Chat delay (for concurrency tests)
- Call counting (via `atomic.Int32`)

### Mock daemon (cmd tests)

CLI tests in `cmd/cmd_test.go` start a mock Unix socket server that speaks the NDJSON protocol. Tests set `XDG_RUNTIME_DIR` to a temp directory so the client connects to the mock instead of a real daemon.

### Mock searcher (handler tests)

Handler tests use a `mockSearcher` to test web search augmentation paths (success, zero results, error) without hitting a real search API.

## What Each Package Tests

### `config/`
- Default values
- YAML loading (valid, invalid, partial, missing file)
- `DefaultGPU` and `DefaultTimeout` fields
- Model alias resolution
- `Save()` round-trip (write then reload)
- `AddModel()` including nil map initialization
- `Load()` from default XDG path
- `WriteDefault()` and idempotency (errors on second call)
- Search API key resolution (config vs env var)

### `protocol/`
- Request/response marshal/unmarshal round-trips for all types
- New fields: `ChatRequest.Model`, `LoadRequest.Name/GPUs/Timeout`, `UnloadRequest.Name`, `StatusPayload.Models`
- Invalid JSON and missing type field handling
- Response helper constructors

### `daemon/`

**ModelSlot:**
- Timer-based inactivity timeout (fires unload callback)
- Chat resets the inactivity timer
- Chat without timer (timeout=0)
- Chat on unloaded slot returns `ErrNoModel`
- Unload stops the timer
- Status reporting (name, path, GPUs, timeout)

**ModelRegistry:**
- Load/unload lifecycle
- Load replaces existing model with same name
- GPU conflict detection (two models cannot share a GPU)
- Multi-GPU conflict detection (overlapping GPU sets)
- Multiple models on different GPUs (no conflict)
- Load error does not leave partial state
- Unload with empty name: auto-resolves when one model, errors when multiple
- Unload of nonexistent model
- Unload propagates backend errors
- Chat auto-routing: single model (no name needed)
- Chat explicit name routing
- Chat default GPU routing (multiple models, no name)
- Chat error when no model on default GPU
- Chat error propagation from backend
- Concurrent chat (10 goroutines)
- Timer expiry removes model from registry
- Independent timeout: one model expires, other stays
- `removeExpired` on nonexistent name (no-op)
- Shutdown unloads all models

**Handler:**
- Chat: streaming deltas + done
- Chat: no model loaded returns error
- Chat: nil payload returns error
- Chat: system prompt (config and request override)
- Chat: web search (success, zero results, error)
- Chat: model name routing (`-m` flag)
- Load: basic, with alias resolution, with timeout, with invalid timeout, with default timeout from config, with GPU list, default name fallback
- Unload: by name, nil payload with single model, no model loaded
- Status: single model (backwards compat fields), multi-model, no model
- Stop: sends signal on StopCh
- Unknown request type

**Server:**
- Status request over socket
- Chat streaming over socket
- Invalid request handling
- Server close
- Multiple sequential connections

**LlamaBackend:**
- Constructor
- Initial state (not loaded, empty path)
- Unload when nil (no error)
- GetStatus when nil (error)
- LoadModel with bad path (single GPU, multi-GPU)
- RunChat when nil (error)

**PID:**
- Write, read, remove PID file
- Process alive check

### `client/`
- Connect to socket
- Connect failure
- StreamChat (deltas + done)
- StreamChat error response
- SendAndWaitOK (success and error)
- SendAndReadStatus (success and error)
- ReadResponse on closed connection

### `cmd/`
- Chat: basic, batch mode, web search flag, system prompt flag, temperature flag, max-tokens flag, model flag
- Load: basic, with `-g` GPUs, with `-t` timeout, with `-p` path, no args error, daemon error
- Unload: basic, with name
- Status: multi-model format, old single-model format, no model
- Start: already running
- Register: success (writes config), bad path error
- All commands: daemon not running error

### `search/`
- Brave search API (mock HTTP)
- Result formatting (empty, single, multiple, special characters)

## Coverage Targets

All packages maintain 70%+ statement coverage. Remaining uncovered code is primarily:
- `daemon.go:Run()` — full integration entry point (requires real socket, PID file, signal handling)
- `llama_engine.go` internals — require a real GGUF model file and GPU
- `cmd/start.go` — forks a real process
