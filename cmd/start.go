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
	"github.com/devon-caron/jarvis/ptyshell"
)

var backgroundOnly bool

var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the jarvis daemon",
	Long:  `Start the jarvis daemon. Without -b, also launches an interactive AI shell with sanitized I/O for LLM context.`,
	RunE:  runStart,
}

func init() {
	startCmd.Flags().BoolVarP(&backgroundOnly, "background", "b", false, "Start daemon only (no interactive shell)")
	rootCmd.AddCommand(startCmd)
}

func runStart(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	// Ensure daemon is running
	if err := ensureDaemon(); err != nil {
		return err
	}

	// If -b flag, we're done (daemon-only mode, backward compat)
	if backgroundOnly {
		return nil
	}

	// Launch interactive PTY shell
	shell := ptyshell.New()
	return shell.Run()
}

// ensureDaemon starts the daemon if it is not already running.
func ensureDaemon() error {
	pidPath := internal.PIDPath()

	if daemon.IsRunning(pidPath) {
		pid, _ := daemon.ReadPID(pidPath)
		fmt.Printf("jarvis daemon already running (pid %d)\n", pid)
		return nil
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
	for i := 0; i < 50; i++ {
		time.Sleep(100 * time.Millisecond)
		if daemon.IsRunning(pidPath) {
			pid, _ := daemon.ReadPID(pidPath)
			fmt.Printf("jarvis daemon started (pid %d)\n", pid)
			return nil
		}
	}

	return fmt.Errorf("daemon did not start within 5 seconds")
}
