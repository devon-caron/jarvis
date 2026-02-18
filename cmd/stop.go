package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var stopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop the jarvis daemon",
	RunE:  runStop,
}

func init() {
	rootCmd.AddCommand(stopCmd)
}

func runStop(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	if err := c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqStop}); err != nil {
		return fmt.Errorf("stop failed: %w", err)
	}

	fmt.Println("jarvis daemon stopped")
	return nil
}
