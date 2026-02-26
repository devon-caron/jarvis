package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/internal"
)

var (
	registerContextSize int
	registerNVLink      bool
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
	registerCmd.Flags().BoolVarP(&registerNVLink, "nvlink", "n", false, "Enable NVLink tensor parallelism (-sm graph)")
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

	cfgPath := internal.ConfigPath()
	cfg, err := config.LoadFrom(cfgPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	cfg.AddModel(name, absPath, registerContextSize, registerNVLink)

	if err := cfg.Save(cfgPath); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	if registerNVLink {
		fmt.Printf("Registered model %q -> %s (context: %d, nvlink)\n", name, absPath, registerContextSize)
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
