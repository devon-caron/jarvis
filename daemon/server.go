package daemon

import (
	"net"
)

// Server listens on a Unix socket and dispatches requests to a Handler.
type Server struct {
	socketPath string
	listener   net.Listener
	handler    *Handler
}

// NewServer creates a Server that will listen on the given socket path.
func NewServer(socketPath string, handler *Handler) *Server {
	return &Server{
		socketPath: socketPath,
		handler:    handler,
	}
}
