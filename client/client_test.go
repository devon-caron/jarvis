package client

import (
	"net"
	"path/filepath"
	"testing"
	"time"
)

// startMockDaemon creates a Unix socket server that responds to requests.
func startMockDaemon(t *testing.T, handler func(net.Conn)) string {
	t.Helper()
	dir := t.TempDir()
	sockPath := filepath.Join(dir, "test.sock")

	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go handler(conn)
		}
	}()

	t.Cleanup(func() { ln.Close() })
	time.Sleep(10 * time.Millisecond)
	return sockPath
}

// TestClient_ConnectTo verifies that ConnectTo can establish and close a
// connection to a Unix socket daemon.
func TestClient_ConnectTo(t *testing.T) {
	sockPath := startMockDaemon(t, func(conn net.Conn) {
		defer conn.Close()
	})

	c, err := ConnectTo(sockPath)
	if err != nil {
		t.Fatalf("ConnectTo: %v", err)
	}
	c.Close()
}
