package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var unloadCmd = &cobra.Command{
	Use:   "unload",
	Short: "Unload the current model from VRAM",
	RunE:  runUnload,
}

func init() {
	rootCmd.AddCommand(unloadCmd)
}

func runUnload(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	if err := c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqUnload}); err != nil {
		return fmt.Errorf("unload failed: %w", err)
	}

	fmt.Println("Model unloaded")
	return nil
}
