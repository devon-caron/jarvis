package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var unloadGPU int

var unloadCmd = &cobra.Command{
	Use:   "unload [model-name]",
	Short: "Unload a model from VRAM",
	Long:  "Unload a model by name, or by GPU with -g. If only one model is loaded, the name is optional.",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runUnload,
}

func init() {
	unloadCmd.Flags().IntVarP(&unloadGPU, "gpu", "g", -1, "Unload whichever model is on this GPU")
	rootCmd.AddCommand(unloadCmd)
}

func runUnload(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	unload := &protocol.UnloadRequest{}
	if unloadGPU >= 0 {
		g := unloadGPU
		unload.GPU = &g
	} else if len(args) > 0 {
		unload.Name = args[0]
	}

	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	if err := c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqUnload, Unload: unload}); err != nil {
		return fmt.Errorf("unload failed: %w", err)
	}

	fmt.Println("Model unloaded")
	return nil
}
