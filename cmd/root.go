package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var (
	webSearch    bool
	batchMode    bool
	systemPrompt string
	maxTokens    int
	contextSize  int
	temperature  float64
	modelFlag    string
	gpuFlag      int
)

var rootCmd = &cobra.Command{
	Use:               "jarvis [prompt]",
	Short:             "Local LLM CLI with daemon-based model management",
	Long:              `Jarvis is a CLI tool that keeps LLM models loaded in VRAM via a background daemon, enabling fast multi-turn chat with direct memory control.`,
	Args:              cobra.ExactArgs(1),
	CompletionOptions: cobra.CompletionOptions{DisableDefaultCmd: true},
	RunE:              runChat,
}

func init() {
	rootCmd.Flags().BoolVarP(&webSearch, "web", "w", false, "Augment prompt with web search results")
	rootCmd.Flags().BoolVarP(&batchMode, "batch", "b", false, "Buffer full response before printing (for use in $())")
	rootCmd.Flags().StringVarP(&systemPrompt, "system", "s", "", "Override system prompt")
	rootCmd.Flags().IntVarP(&maxTokens, "max-tokens", "n", 0, "Max tokens to generate (0 = config default)")
	rootCmd.Flags().IntVarP(&contextSize, "context-size", "c", 8192, "Context window size in tokens (default 8192)")
	rootCmd.Flags().Float64VarP(&temperature, "temperature", "t", 0, "Temperature (0 = config default)")
	rootCmd.Flags().StringVarP(&modelFlag, "model", "m", "", "Target model name (when multiple models loaded)")
	rootCmd.Flags().IntVarP(&gpuFlag, "gpu", "g", -1, "Route to whichever model is loaded on this GPU")
}

// Execute runs the root command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func runChat(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	prompt := args[0]

	// cfg, _ := config.Load()

	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon. Is it running? Start with: jarvis start")
	}
	defer c.Close()

	opts := protocol.InferenceOpts{
		ContextSize: contextSize,
	}
	if maxTokens > 0 {
		opts.MaxTokens = maxTokens
	}
	if temperature > 0 {
		opts.Temperature = temperature
	}

	var gpuPtr *int
	if gpuFlag >= 0 {
		g := gpuFlag
		gpuPtr = &g
	}

	req := &protocol.Request{
		Type: protocol.ReqChat,
		Chat: &protocol.ChatRequest{
			Messages: []protocol.ChatMessage{
				{Role: "user", Content: prompt},
			},
			Model:        modelFlag,
			GPU:          gpuPtr,
			WebSearch:    webSearch,
			SystemPrompt: systemPrompt,
			Opts:         opts,
		},
	}

	// _ = cfg // config loaded for potential future use

	if batchMode {
		var buf strings.Builder
		err = c.StreamChat(req, func(token string) {
			buf.WriteString(token)
		})
		if err != nil {
			return err
		}
		fmt.Print(buf.String())
	} else {
		err = c.StreamChat(req, func(token string) {
			fmt.Print(token)
		})
		if err != nil {
			return err
		}
		fmt.Println()
	}
	return nil
}
