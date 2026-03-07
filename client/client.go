package client

import (
	"bufio"
	"fmt"
	"net"

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
	scanner.Buffer(make([]byte, 0, 64*1024), 5*1024*1024)
	return &Client{conn: conn, scanner: scanner}, nil
}

func (c *Client) Close() error {

	return fmt.Errorf("unimplmented")
}

func (c *Client) SendAndWaitOK(req *protocol.Request) error {
	return fmt.Errorf("unimplmented")
}

func (c *Client) StreamChat(req *protocol.Request, onToken func(string)) error {
	return fmt.Errorf("unimplmented")
}

func (c *Client) SendAndReadStatus(req *protocol.Request) (*protocol.StatusPayload, error) {
	return nil, fmt.Errorf("unimplmented")
}
