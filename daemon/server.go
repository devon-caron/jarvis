package daemon

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net"
	"os"

	"github.com/devon-caron/jarvis/protocol"
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

// Listen creates the Unix socket and starts listening.
func (s *Server) Listen() error {
	// Remove stale socket file if it exists
	os.Remove(s.socketPath)

	ln, err := net.Listen("unix", s.socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", s.socketPath, err)
	}
	s.listener = ln

	// Make socket accessible to the user
	os.Chmod(s.socketPath, 0700)
	return nil
}

// Serve accepts connections in a loop until the listener is closed.
func (s *Server) Serve() error {
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			// Check if we were closed intentionally
			select {
			case <-s.handler.StopCh:
				return nil
			default:
			}
			// If not a clean shutdown, check if it's a closed listener
			if opErr, ok := err.(*net.OpError); ok && opErr.Err.Error() == "use of closed network connection" {
				return nil
			}
			log.Printf("accept error: %v", err)
			continue
		}
		go s.handleConnection(conn)
	}
}

// Close shuts down the listener and removes the socket file.
func (s *Server) Close() error {
	if s.listener != nil {
		s.listener.Close()
	}
	os.Remove(s.socketPath)
	return nil
}

func (s *Server) handleConnection(conn net.Conn) {
	defer conn.Close()

	scanner := bufio.NewScanner(conn)
	// Allow up to 1MB messages
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	if !scanner.Scan() {
		return
	}
	line := scanner.Bytes()

	req, err := protocol.UnmarshalRequest(line)
	if err != nil {
		rw := NewResponseWriter(conn)
		rw.Write(protocol.ErrorResponse(fmt.Sprintf("invalid request: %v", err)))
		return
	}

	// Create a context that cancels when the client disconnects.
	// After the request line, the client sends no more data — it only waits
	// for responses. A read returning means the connection was closed (^C).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		buf := make([]byte, 1)
		conn.Read(buf)
		cancel()
	}()

	rw := NewResponseWriter(conn)
	s.handler.Handle(ctx, req, rw)
}
