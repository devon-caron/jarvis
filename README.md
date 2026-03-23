# Jarvis

A CLI tool and AI provider that puts control back into the hands of users.

While revolutionary projects like OpenClaw and its derivative products have given individuals modest control of the lightning-in-a-box power that AI brings, these projects have also proliferated user information, API keys, and also given AI agents the chance to go rogue on a moment's notice. Jarvis is the CLI tool and AI provider that puts control back into the hands of users while still allowing AI to fully tap into its capabilities.

Jarvis runs a lightweight background daemon that keeps your model loaded in GPU memory, giving you instant, low-latency access to local LLMs directly from your terminal — no cloud, no API keys leaving your machine, no surprises.

---

## Features

- **Persistent daemon** — keeps your model loaded in VRAM so every prompt is fast
- **((TODO)) Multi-turn conversations** — context is maintained per-shell session
- **Streaming responses** — tokens appear as they're generated
- **Web search augmentation** — enrich prompts with live search results with Brave (and others soon!)
- **Model registry** — name and configure your models once, use them by alias
- **Multi-GPU support** — layer-wise and row-wise tensor splitting
- **Benchmarking** — built-in performance measurement (tokens/s, TTFT)
- **Shell completions** — bash, zsh, fish, and PowerShell

---

## Installation

**Requirements:** Go 1.25.6+ and [llama-server](https://github.com/ggerganov/llama.cpp) on your PATH.

((Mac users, use `brew install llama-server` for a quick installation of llama-server.))

```bash
go install github.com/devon-caron/jarvis@latest
```

Or build from source:

```bash
git clone https://github.com/devon-caron/jarvis.git
cd jarvis
go build -o build/bin/jarvis .
```

---

## Quick Start

### 1. Initialize configuration

```bash
jarvis config init
```

This creates `~/.config/jarvis/config.yaml` with sensible defaults.

### 2. Register a model

```bash
jarvis models register mymodel /path/to/model.gguf
```

### 3. Start the daemon

```bash
jarvis start
```

### 4. Load a model

```bash
jarvis load mymodel
```

### 5. Chat

```bash
jarvis "What is the capital of France?"
```

---

## Usage

### Sending prompts

```bash
# Basic prompt
jarvis "Explain quicksort"

# With web search context
jarvis -w "Latest news on Go 1.25"

# Show performance stats
jarvis -s "Write a haiku"

# Batch mode (buffer full response, useful for shell substitution)
jarvis -b "Give me a UUID"

# Set max tokens
jarvis -n 200 "Summarize this concept"

# Override the system prompt
jarvis --system "You are a pirate." "Tell me about ships"

# Clear conversation history for this shell
jarvis -C
```

### Daemon management

```bash
jarvis start          # Start the daemon
jarvis start -d       # Start with debug logging
jarvis stop           # Stop the daemon
jarvis status         # Show daemon PID, loaded model, GPU info
```

### Model management

```bash
# Load by registered name
jarvis load mymodel

# Load by file path
jarvis load -p /path/to/model.gguf

# Load with options
jarvis load mymodel -g 0,1 -c 16384 -f -t 30m

# Unload the current model
jarvis unload
```

### Model registry

```bash
# List registered models
jarvis models ls

# Register with defaults
jarvis models register mymodel /path/to/model.gguf -c 8192 -f

# Remove a model
jarvis models unregister mymodel
```

### Shell completions

```bash
jarvis config completion --shell zsh >> ~/.zshrc
jarvis config completion --shell bash >> ~/.bashrc
```

---

## Configuration

Config lives at `~/.config/jarvis/config.yaml`. Key sections:

```yaml
default_model: mymodel
default_timeout: "30m"
default_gpu: 0

models:
  mymodel:
    path: /path/to/model.gguf
    context_size: 8192
    flash_attention: true

inference:
  context_size: 8192
  max_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  top_k: 40

system_prompt: "You are a helpful AI assistant."

search:
  provider: brave
  api_key: ""
  max_results: 5
```

---

## Architecture

```
┌──────────┐    Unix Socket    ┌──────────────┐    subprocess    ┌──────────────┐
│  jarvis   │ ◄──────────────► │    daemon     │ ◄─────────────► │ llama-server │
│  (CLI)    │     NDJSON       │  (persistent) │                 │  (inference) │
└──────────┘                   └──────────────┘                  └──────────────┘
```

- **CLI** sends requests over a Unix socket
- **Daemon** manages model lifecycle and routes inference
- **llama-server** handles the actual model execution in VRAM
- All communication stays local — nothing leaves your machine

---

## License

See [LICENSE](LICENSE) for details.