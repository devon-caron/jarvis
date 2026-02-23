package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/internal"
)

var registerPath string

var registerCmd = &cobra.Command{
	Use:   "register <name>",
	Short: "Register a named model alias in config",
	Args:  cobra.ExactArgs(1),
	RunE:  runRegister,
}

func init() {
	registerCmd.Flags().StringVarP(&registerPath, "path", "p", "", "Path to the model file (required)")
	registerCmd.MarkFlagRequired("path")
	rootCmd.AddCommand(registerCmd)
}

func runRegister(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	name := args[0]

	// Resolve to absolute path so the daemon (different cwd) can find the file.
	absPath, err := filepath.Abs(registerPath)
	if err != nil {
		return fmt.Errorf("invalid path %q: %w", registerPath, err)
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

	cfg.AddModel(name, absPath)

	if err := cfg.Save(cfgPath); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	fmt.Printf("Registered model %q -> %s\n", name, absPath)
	return nil
}
