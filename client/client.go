package client

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"time"

	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/protocol"
)

type Client struct {
	conn    net.Conn
	scanner *bufio.Scanner
}

func Connect() (*Client, error) {
	return ConnectTo(internal.SocketPath())
}

// ConnectTo establishes a connection to the daemon at the given socket path.
// This function only runs after the daemon socket is created via the 'jarvis start' command.
func ConnectTo(socketPath string) (*Client, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("could not connect to daemon process (is it running?)")
	}

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, internal.BUFFER_PAGE_SIZE), internal.BUFFER_SIZE)
	return &Client{conn: conn, scanner: scanner}, nil
}

func (c *Client) Close() error {

	return fmt.Errorf("unimplmented")
}

func (c *Client) Send(req *protocol.Request) error {
	data, err := protocol.MarshalRequest(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}
	data = append(data, '\n')

	_, err = c.conn.Write(data)
	if err != nil {
		return fmt.Errorf("failed to write request: %w", err)
	}
	return nil
}

func (c *Client) SendAndWaitOK(req *protocol.Request) error {
	if err := c.Send(req); err != nil {
		return err
	}
	c.conn.SetReadDeadline(time.Now().Add(3 * time.Minute))
	defer c.conn.SetReadDeadline(time.Time{})
	resp, err := c.ReadResponse()
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}
	if resp.Type == protocol.RespError && resp.Error != nil {
		return fmt.Errorf("daemon error: %s", resp.Error.Message)
	}
	if resp.Type != protocol.RespOK {
		return fmt.Errorf("expected OK response, got %s", resp.Type)
	}
	return nil
}

func (c *Client) ReadResponse() (*protocol.Response, error) {
	if !c.scanner.Scan() {
		if c.scanner.Err() != nil {
			return nil, fmt.Errorf("scanner error: %w", c.scanner.Err())
		}
		return nil, fmt.Errorf("connection closed")
	}
	return protocol.UnmarshalResponse(c.scanner.Bytes())
}

func (c *Client) StreamChat(req *protocol.Request, onToken func(string)) error {
	log.Printf("streaming chat request: %v", req)
	return fmt.Errorf("unimplmented")
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
