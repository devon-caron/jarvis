package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var loadGPULayers int

var loadCmd = &cobra.Command{
	Use:   "load <model-path-or-alias>",
	Short: "Load a model into VRAM",
	Args:  cobra.ExactArgs(1),
	RunE:  runLoad,
}

func init() {
	loadCmd.Flags().IntVarP(&loadGPULayers, "gpu-layers", "g", -1, "Number of layers to offload to GPU (-1 for all)")
	rootCmd.AddCommand(loadCmd)
}

func runLoad(cmd *cobra.Command, args []string) error {
	modelPath := args[0]

	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	fmt.Printf("Loading model: %s\n", modelPath)

	req := &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{
			ModelPath: modelPath,
			GPULayers: loadGPULayers,
		},
	}

	if err := c.SendAndWaitOK(req); err != nil {
		return fmt.Errorf("load failed: %w", err)
	}

	fmt.Println("Model loaded successfully")
	return nil
}
