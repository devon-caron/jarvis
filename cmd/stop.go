package cmd

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/protocol"
)

var stopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop the jarvis daemon and PTY shell",
	RunE:  runStop,
}

func init() {
	rootCmd.AddCommand(stopCmd)
}

func runStop(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	// Stop the daemon
	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	if err := c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqStop}); err != nil {
		return fmt.Errorf("stop failed: %w", err)
	}

	fmt.Println("jarvis daemon stopped")

	// Also stop PTY shell if running
	stopPTY()

	return nil
}

// stopPTY sends SIGTERM to the PTY process if its PID file exists.
func stopPTY() {
	ptyPIDPath := internal.PTYPIDPath()
	data, err := os.ReadFile(ptyPIDPath)
	if err != nil {
		return // no PTY running
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		os.Remove(ptyPIDPath)
		return
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		os.Remove(ptyPIDPath)
		return
	}

	proc.Signal(syscall.SIGTERM)
	os.Remove(ptyPIDPath)
	fmt.Println("jarvis PTY shell stopped")
}
