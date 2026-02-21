package cmd

import (
	"fmt"
	"os"

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

	// Validate that path exists on disk
	if _, err := os.Stat(registerPath); err != nil {
		return fmt.Errorf("model path does not exist: %s", registerPath)
	}

	cfgPath := internal.ConfigPath()
	cfg, err := config.LoadFrom(cfgPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	cfg.AddModel(name, registerPath)

	if err := cfg.Save(cfgPath); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	fmt.Printf("Registered model %q -> %s\n", name, registerPath)
	return nil
}
