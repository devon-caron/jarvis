package cmd

import (
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/daemon"
	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/protocol"
)

var restartCmd = &cobra.Command{
	Use:   "restart",
	Short: "Restart the jarvis daemon",
	RunE:  runRestart,
}

func init() {
	rootCmd.AddCommand(restartCmd)
}

func runRestart(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	pidPath := internal.PIDPath()

	if !daemon.IsRunning(pidPath) {
		return fmt.Errorf("daemon is not running; use 'jarvis start' to start it")
	}

	// Stop the running daemon.
	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon: %w", err)
	}
	c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqStop})
	c.Close()

	// Wait for the old process to exit.
	for i := 0; i < 50; i++ {
		time.Sleep(100 * time.Millisecond)
		if !daemon.IsRunning(pidPath) {
			break
		}
	}
	if daemon.IsRunning(pidPath) {
		return fmt.Errorf("daemon did not stop within 5 seconds")
	}

	// Start a new daemon via the shared helper.
	return runStart(cmd, args)
}
