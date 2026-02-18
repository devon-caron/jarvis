package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
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

func init() {
	completionCmd.Flags().StringVar(&completionShell, "shell", "bash", "Shell type (bash, zsh, fish, powershell)")
	configCmd.AddCommand(completionCmd)
	rootCmd.AddCommand(configCmd)
	rootCmd.CompletionOptions.DisableDefaultCmd = true
}
