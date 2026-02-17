package cmd

import (
	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/daemon"
)

// daemonCmd is the hidden subcommand invoked by "jarvis start" after forking.
var daemonCmd = &cobra.Command{
	Use:    "_daemon",
	Hidden: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		return daemon.Run()
	},
}

func init() {
	rootCmd.AddCommand(daemonCmd)
}
