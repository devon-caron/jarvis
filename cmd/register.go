package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/daemon"
	"github.com/devon-caron/jarvis/internal"
)

var (
	registerContextSize    int
	registerSplitMode      string
	registerFlashAttention bool
	registerBatchSize      int
	registerTensorSplit    string
)

var registerCmd = &cobra.Command{
	Use:   "register <name> <path>",
	Short: "Register a named model alias in config",
	Args:  cobra.ExactArgs(2),
	RunE:  runRegister,
}

var unregisterCmd = &cobra.Command{
	Use:   "unregister <name>",
	Short: "Remove a named model alias from config",
	Args:  cobra.ExactArgs(1),
	RunE:  runUnregister,
}

func init() {
	registerCmd.Flags().IntVarP(&registerContextSize, "context-size", "c", 8192, "Default context window size")
	registerCmd.Flags().StringVarP(&registerSplitMode, "nvlink", "n", "", "Multi-GPU split mode: l(ayer), r(ow)")
	registerCmd.Flags().BoolVarP(&registerFlashAttention, "flash-attn", "f", false, "Enable flash attention by default for this model")
	registerCmd.Flags().IntVarP(&registerBatchSize, "batch-size", "B", 0, "Default micro-batch size (0 = server default)")
	registerCmd.Flags().StringVarP(&registerTensorSplit, "tensor-split", "T", "", "Default tensor split (e.g. \"1,1\")")
	modelsCmd.AddCommand(registerCmd)
	modelsCmd.AddCommand(unregisterCmd)
}

func runRegister(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	name := args[0]
	modelPath := args[1]

	// Resolve to absolute path so the daemon (different cwd) can find the file.
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return fmt.Errorf("invalid path %q: %w", modelPath, err)
	}

	// Validate that the file exists on disk.
	if _, err := os.Stat(absPath); err != nil {
		return fmt.Errorf("model path does not exist: %s", absPath)
	}

	// Validate and normalize split mode.
	splitMode, err := daemon.NormalizeSplitMode(registerSplitMode)
	if err != nil {
		return err
	}

	cfgPath := internal.ConfigPath()
	cfg, err := config.LoadFrom(cfgPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	cfg.AddModel(name, config.ModelEntry{
		Path:           absPath,
		ContextSize:    registerContextSize,
		SplitMode:      splitMode,
		FlashAttention: registerFlashAttention,
		BatchSize:      registerBatchSize,
		TensorSplit:    registerTensorSplit,
	})

	if err := cfg.Save(cfgPath); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	if splitMode != "" {
		fmt.Printf("Registered model %q -> %s (context: %d, split: %s)\n", name, absPath, registerContextSize, splitMode)
	} else {
		fmt.Printf("Registered model %q -> %s (context: %d)\n", name, absPath, registerContextSize)
	}
	return nil
}

func runUnregister(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	name := args[0]

	cfgPath := internal.ConfigPath()
	cfg, err := config.LoadFrom(cfgPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	if !cfg.RemoveModel(name) {
		return fmt.Errorf("model %q not found in registry", name)
	}

	if err := cfg.Save(cfgPath); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	fmt.Printf("Unregistered model %q\n", name)
	return nil
}
