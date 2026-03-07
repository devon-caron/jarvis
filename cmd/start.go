package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/daemon"
	"github.com/devon-caron/jarvis/internal"
)

var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the jarvis daemon",
	RunE:  runStart,
}

func init() {
	rootCmd.AddCommand(startCmd)
}

func runStart(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	pidPath := internal.PIDPath()

	// Check if already running
	if daemon.IsRunning(pidPath) {
		pid, _ := daemon.ReadPID(pidPath)
		return fmt.Errorf("daemon already running (pid %d)", pid)
	}

	// Self-fork: launch hidden _daemon subcommand
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("cannot find executable: %w", err)
	}

	daemonCmd := exec.Command(exe, "_daemon")
	daemonCmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid: true,
	}

	// Inherit necessary env vars for CUDA
	daemonCmd.Env = os.Environ()

	if err := daemonCmd.Start(); err != nil {
		return fmt.Errorf("failed to start daemon: %w", err)
	}

	// Wait for PID file to appear (confirms daemon is ready)
	for range 50 {
		time.Sleep(100 * time.Millisecond)
		if daemon.IsRunning(pidPath) {
			pid, _ := daemon.ReadPID(pidPath)
			fmt.Printf("jarvis daemon started (pid %d)\n", pid)
			return nil
		}
	}

	return fmt.Errorf("daemon did not start within 5 seconds")
}
