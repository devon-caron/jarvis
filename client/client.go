package client

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"time"

	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/protocol"
)

// Client communicates with the jarvis daemon over a Unix socket.
type Client struct {
	conn    net.Conn
	scanner *bufio.Scanner
}

// Connect establishes a connection to the daemon socket.
func Connect() (*Client, error) {
	return ConnectTo(internal.SocketPath())
}

// ConnectTo establishes a connection to the daemon at the given socket path.
func ConnectTo(socketPath string) (*Client, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return &Client{conn: conn, scanner: scanner}, nil
}

// Close closes the connection.
func (c *Client) Close() error {
	return c.conn.Close()
}

// Send sends a request to the daemon.
func (c *Client) Send(req *protocol.Request) error {
	data, err := json.Marshal(req)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = c.conn.Write(data)
	return err
}

// ReadResponse reads a single NDJSON response line from the daemon.
func (c *Client) ReadResponse() (*protocol.Response, error) {
	if !c.scanner.Scan() {
		if err := c.scanner.Err(); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("connection closed")
	}
	return protocol.UnmarshalResponse(c.scanner.Bytes())
}

// StreamChat sends a chat request and calls onDelta for each streamed token.
// Returns the final error message if the daemon reports one, or nil on success.
func (c *Client) StreamChat(req *protocol.Request, onDelta func(string)) error {
	if err := c.Send(req); err != nil {
		return err
	}
	for {
		resp, err := c.ReadResponse()
		if err != nil {
			return err
		}
		switch resp.Type {
		case protocol.RespDelta:
			if resp.Delta != nil {
				onDelta(resp.Delta.Content)
			}
		case protocol.RespDone:
			return nil
		case protocol.RespError:
			if resp.Error != nil {
				return fmt.Errorf("%s", resp.Error.Message)
			}
			return fmt.Errorf("unknown error from daemon")
		default:
			return fmt.Errorf("unexpected response type: %s", resp.Type)
		}
	}
}

// SendAndWaitOK sends a request and waits for an OK or error response.
// A 3-minute read deadline prevents the client from hanging indefinitely
// if the daemon crashes or becomes unresponsive.
func (c *Client) SendAndWaitOK(req *protocol.Request) error {
	if err := c.Send(req); err != nil {
		return err
	}
	c.conn.SetReadDeadline(time.Now().Add(3 * time.Minute))
	defer c.conn.SetReadDeadline(time.Time{})
	resp, err := c.ReadResponse()
	if err != nil {
		return err
	}
	if resp.Type == protocol.RespError && resp.Error != nil {
		return fmt.Errorf("%s", resp.Error.Message)
	}
	if resp.Type != protocol.RespOK {
		return fmt.Errorf("unexpected response: %s", resp.Type)
	}
	return nil
}

// SendAndReadStatus sends a request and reads a status response.
func (c *Client) SendAndReadStatus(req *protocol.Request) (*protocol.StatusPayload, error) {
	if err := c.Send(req); err != nil {
		return nil, err
	}
	resp, err := c.ReadResponse()
	if err != nil {
		return nil, err
	}
	if resp.Type == protocol.RespError && resp.Error != nil {
		return nil, fmt.Errorf("%s", resp.Error.Message)
	}
	if resp.Type != protocol.RespStatus || resp.Status == nil {
		return nil, fmt.Errorf("unexpected response: %s", resp.Type)
	}
	return resp.Status, nil
}
