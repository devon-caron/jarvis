package client

import (
	"fmt"

	"github.com/devon-caron/jarvis/protocol"
)

type Client struct{}

func Connect() (*Client, error) {
	return nil, fmt.Errorf("unimplmented")
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
