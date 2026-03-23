package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show daemon and model status",
	RunE:  runStatus,
}

func init() {
	rootCmd.AddCommand(statusCmd)
}

func runStatus(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("daemon is not running")
	}
	defer c.Close()

	status, err := c.SendAndReadStatus(&protocol.Request{Type: protocol.ReqStatus})
	if err != nil {
		return err
	}

	fmt.Printf("Daemon:  running (pid %d)\n", status.PID)

	if status.ModelLoaded && status.Model != nil {
		fmt.Printf("Model:   %s\n", status.Model.Name)
		fmt.Printf("Path:    %s\n", status.Model.ModelPath)
		fmt.Printf("GPUs:    %v\n", status.Model.GPUs)
		for _, gpu := range status.Model.GPUInfo {
			fmt.Printf("GPU %d:   %s (%d/%d MB)\n",
				gpu.DeviceID, gpu.DeviceName,
				gpu.FreeMemoryMB, gpu.TotalMemoryMB)
		}
	} else {
		fmt.Printf("Model:   not loaded\n")
	}

	return nil
}
