package internal

const (
	BUFFER_SIZE            = 5 * 1024 * 1024 // 5MB
	BUFFER_PAGE_SIZE       = 64 * 1024       // 64KB
	OPENAI_COMPAT_SSE_DONE = "[DONE]"
)

var VRAM_ERROR_PATTERNS = []string{
	"cudamalloc failed",
	"out of memory",
	"failed to allocate",
	"unable to allocate",
}

var CONTEXT_LENGTH_ERROR_PATTERNS = []string{
	"context length exceeded",
	"too many tokens",
	"maximum context length",
}

var (
	LogFileName string = "daemon.log"
)
