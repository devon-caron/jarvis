package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/internal"
)

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "Model management commands",
}

func init() {
	// models load — mirrors top-level load, shares RunE and flag vars
	modelsLoadCmd := &cobra.Command{
		Use:   "load [flags] <model-name>",
		Short: loadCmd.Short,
		Long:  loadCmd.Long,
		Args:  cobra.MaximumNArgs(1),
		RunE:  runLoad,
	}
	modelsLoadCmd.Flags().StringVarP(&loadGPUs, "gpus", "g", "", `GPU device IDs (e.g. "0" or "0,1")`)
	modelsLoadCmd.Flags().StringVarP(&loadPath, "path", "p", "", "Inline model path (instead of registered name)")
	modelsLoadCmd.Flags().StringVarP(&loadTimeout, "timeout", "t", "", `Inactivity timeout (e.g. "30m", "1h")`)
	modelsLoadCmd.Flags().IntVarP(&loadContextSize, "context-size", "c", 0, "Context window size (0 = use registered default or 8192)")
	modelsLoadCmd.Flags().BoolVarP(&loadNVLink, "nvlink", "n", false, "Enable NVLink tensor parallelism (-sm graph)")

	// models unload — mirrors top-level unload
	modelsUnloadCmd := &cobra.Command{
		Use:   "unload [model-name]",
		Short: unloadCmd.Short,
		Long:  unloadCmd.Long,
		Args:  cobra.MaximumNArgs(1),
		RunE:  runUnload,
	}
	modelsUnloadCmd.Flags().IntVarP(&unloadGPU, "gpu", "g", -1, "Unload whichever model is on this GPU")

	// models ls — lists registered models from config
	modelsLsCmd := &cobra.Command{
		Use:   "ls",
		Short: "List registered models",
		RunE:  runModelsLs,
	}

	modelsCmd.AddCommand(modelsLoadCmd)
	modelsCmd.AddCommand(modelsUnloadCmd)
	modelsCmd.AddCommand(modelsLsCmd)
	rootCmd.AddCommand(modelsCmd)
}

func runModelsLs(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	cfg, err := config.LoadFrom(internal.ConfigPath())
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	if len(cfg.Models) == 0 {
		fmt.Println("No models registered. Use 'jarvis models register' to add one.")
		return nil
	}
	for name, entry := range cfg.Models {
		flags := ""
		if entry.NVLink {
			flags += " (nvlink)"
		}
		if entry.ContextSize > 0 {
			fmt.Printf("  %-20s %s (ctx: %d)%s\n", name, entry.Path, entry.ContextSize, flags)
		} else {
			fmt.Printf("  %-20s %s%s\n", name, entry.Path, flags)
		}
	}
	return nil
}
