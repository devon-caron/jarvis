package cmd

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/config"
)

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Configuration utilities",
}

var completionShell string

var completionCmd = &cobra.Command{
	Use:   "completion",
	Short: "Generate shell completion script",
	Long:  `Generate a shell completion script. Default is bash. Usage: eval "$(jarvis config completion)"`,
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		switch completionShell {
		case "bash":
			return rootCmd.GenBashCompletionV2(os.Stdout, true)
		case "zsh":
			return rootCmd.GenZshCompletion(os.Stdout)
		case "fish":
			return rootCmd.GenFishCompletion(os.Stdout, true)
		case "powershell":
			return rootCmd.GenPowerShellCompletionWithDesc(os.Stdout)
		default:
			return fmt.Errorf("unsupported shell: %s", completionShell)
		}
	},
}

var initConfigCmd = &cobra.Command{
	Use:   "init",
	Short: "Create a default config file",
	RunE: func(cmd *cobra.Command, args []string) error {
		cmd.SilenceUsage = true
		path, err := config.WriteDefault()
		if err != nil {
			if errors.Is(err, os.ErrExist) {
				return fmt.Errorf("config file already exists: %s", path)
			}
			return fmt.Errorf("failed to write config: %w", err)
		}
		fmt.Printf("Config file created: %s\n", path)
		return nil
	},
}

func init() {
	completionCmd.Flags().StringVar(&completionShell, "shell", "bash", "Shell type (bash, zsh, fish, powershell)")
	configCmd.AddCommand(completionCmd)
	configCmd.AddCommand(initConfigCmd)
	rootCmd.AddCommand(configCmd)
	rootCmd.CompletionOptions.DisableDefaultCmd = true
}
