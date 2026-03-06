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
	Long: `Print the context that the LLM currently sees.

In PTY mode (jarvis start): prints the sanitized terminal transcript.
In daemon mode (jarvis start -b): prints the per-shell chat history from the daemon.`,
	RunE: runPrintContext,
}

func init() {
	printContextCmd.Flags().StringVarP(&printContextModel, "model", "m", "", "Target model name (when multiple models loaded)")
	modelsCmd.AddCommand(printContextCmd)
}

func runPrintContext(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	// PTY mode: read context from file
	if os.Getenv("JARVIS_PTY") == "1" {
		return printPTYContext()
	}

	// Daemon mode: query daemon for ModelSlot.history
	return printDaemonContext()
}

// printPTYContext reads the terminal transcript from the context file.
func printPTYContext() error {
	ctxFile := os.Getenv("JARVIS_CONTEXT_FILE")
	if ctxFile == "" {
		return fmt.Errorf("JARVIS_CONTEXT_FILE not set (are you inside a jarvis PTY shell?)")
	}

	data, err := os.ReadFile(ctxFile)
	if err != nil {
		return fmt.Errorf("cannot read context file %s: %w", ctxFile, err)
	}

	if len(data) == 0 {
		fmt.Println("(no terminal context yet)")
		return nil
	}

	fmt.Print(string(data))
	return nil
}

// printDaemonContext queries the daemon for per-shell chat history.
func printDaemonContext() error {
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
