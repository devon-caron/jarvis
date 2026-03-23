package daemon

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net"
	"os"

	"github.com/devon-caron/jarvis/internal"
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
	if err := os.Remove(s.socketPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove stale socket file: %w", err)
	}

	ln, err := net.Listen("unix", s.socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", s.socketPath, err)
	}
	s.listener = ln

	// Make socket accessible to the user
	os.Chmod(s.socketPath, 0700)
	return nil
}

func (s *Server) Serve() error {
	for {
		conn, err := s.listener.Accept()
		if err != nil {

			// Check if listener was intentionally closed
			select {
			case <-s.handler.StopCh:
				return nil
			default:
			}

			// If not a clean shutdown, check if it's a closed listener
			if opErr, ok := err.(*net.OpError); ok && opErr.Err.Error() == "use of closed network connection" {
				return nil
			}

			log.Println("Accept error: ", err)
			continue
		}
		go s.handleConnection(conn)
	}
}

func (s *Server) handleConnection(conn net.Conn) {
	defer conn.Close()

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, internal.BUFFER_PAGE_SIZE), internal.BUFFER_SIZE)
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

func (s *Server) Close() error {
	if s.listener != nil {
		if err := s.listener.Close(); err != nil {
			return err
		}
	}
	if err := os.Remove(s.socketPath); err != nil {
		return err
	}
	return nil
}
