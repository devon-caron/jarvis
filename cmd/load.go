package cmd

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/protocol"
)

var (
	loadGPUs        string
	loadPath        string
	loadTimeout     string
	loadContextSize int
	loadNVLink      bool
	loadParallel    int
)

var loadCmd = &cobra.Command{
	Use:   "load [flags] <model-name>",
	Short: "Load a model into VRAM",
	Long: `Load a model by registered name or inline path.

Examples:
  jarvis load llama70b                    # load by name onto default GPU
  jarvis load -g 0,1 llama70b            # split across GPUs 0 and 1
  jarvis load -p /path/to/model.gguf     # load by path
  jarvis load -g 0 -t 30m llama70b       # load with 30min timeout`,
	Args: cobra.MaximumNArgs(1),
	RunE: runLoad,
}

func init() {
	loadCmd.Flags().StringVarP(&loadGPUs, "gpus", "g", "", "GPU device IDs (e.g. \"0\" or \"0,1\")")
	loadCmd.Flags().StringVarP(&loadPath, "path", "p", "", "Inline model path (instead of registered name)")
	loadCmd.Flags().StringVarP(&loadTimeout, "timeout", "t", "", "Inactivity timeout (e.g. \"30m\", \"1h\")")
	loadCmd.Flags().IntVarP(&loadContextSize, "context-size", "c", 0, "Context window size (0 = use registered default or 8192)")
	loadCmd.Flags().BoolVarP(&loadNVLink, "nvlink", "n", false, "Enable NVLink tensor parallelism (-sm graph)")
	loadCmd.Flags().IntVarP(&loadParallel, "parallel", "P", 0, "Number of parallel slots for concurrent requests (0 = single slot)")
	rootCmd.AddCommand(loadCmd)
}

func runLoad(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	var modelName string
	var modelPath string

	if loadPath != "" {
		modelPath = loadPath
		if len(args) > 0 {
			modelName = args[0]
		}
	} else if len(args) > 0 {
		modelName = args[0]
	} else {
		return fmt.Errorf("must specify a model name or --path")
	}

	// Parse GPU list
	var gpus []int
	if loadGPUs != "" {
		for _, s := range strings.Split(loadGPUs, ",") {
			s = strings.TrimSpace(s)
			id, err := strconv.Atoi(s)
			if err != nil {
				return fmt.Errorf("invalid GPU ID %q: %w", s, err)
			}
			gpus = append(gpus, id)
		}
	}

	c, err := client.Connect()
	if err != nil {
		return fmt.Errorf("cannot connect to daemon (is it running?): %w", err)
	}
	defer c.Close()

	displayName := modelName
	if modelPath != "" {
		displayName = modelPath
	}
	if loadContextSize > 0 {
		fmt.Printf("Loading model: %s (context: %d)\n", displayName, loadContextSize)
	} else {
		fmt.Printf("Loading model: %s\n", displayName)
	}

	req := &protocol.Request{
		Type: protocol.ReqLoad,
		Load: &protocol.LoadRequest{
			ModelPath:   modelPath,
			Name:        modelName,
			GPUs:        gpus,
			Timeout:     loadTimeout,
			ContextSize: loadContextSize,
			NVLink:      loadNVLink,
			Parallel:    loadParallel,
		},
	}

	if err := c.SendAndWaitOK(req); err != nil {
		return fmt.Errorf("load failed: %w", err)
	}

	fmt.Println("Model loaded successfully")
	return nil
}
