package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var unloadCmd = &cobra.Command{
	Use:   "unload [model-name]",
	Short: "Unload a model from VRAM",
	Long:  "Unload a specific model by name. If only one model is loaded, the name is optional.",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runUnload,
}

func init() {
	rootCmd.AddCommand(unloadCmd)
}

func runUnload(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	var name string
	if len(args) > 0 {
		name = args[0]
	}

	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	req := &protocol.Request{
		Type:   protocol.ReqUnload,
		Unload: &protocol.UnloadRequest{Name: name},
	}

	if err := c.SendAndWaitOK(req); err != nil {
		return fmt.Errorf("unload failed: %w", err)
	}

	fmt.Println("Model unloaded")
	return nil
}
