package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var printContextModel string

var printContextCmd = &cobra.Command{
	Use:   "print-context",
	Short: "Print the model's conversation context",
	RunE:  runPrintContext,
}

func init() {
	printContextCmd.Flags().StringVarP(&printContextModel, "model", "m", "", "Target model name (when multiple models loaded)")
	modelsCmd.AddCommand(printContextCmd)
}

func runPrintContext(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	ctx, err := c.SendAndReadContext(&protocol.Request{
		Type: protocol.ReqGetContext,
		GetContext: &protocol.GetContextRequest{
			Model:    printContextModel,
			ShellPID: os.Getppid(),
		},
	})
	if err != nil {
		return err
	}

	fmt.Printf("Model: %s\n", ctx.Model)
	fmt.Printf("Messages: %d\n\n", len(ctx.Messages))

	for i, msg := range ctx.Messages {
		fmt.Printf("[%d] %s:\n%s\n\n", i+1, msg.Role, msg.Content)
	}

	return nil
}
