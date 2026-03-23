package cmd

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/daemon"
	"github.com/devon-caron/jarvis/internal"
)

var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the jarvis daemon",
	RunE:  runStart,
}

var debugLogFlag bool

func init() {
	startCmd.Flags().BoolVarP(&debugLogFlag, "debug", "d", false, "enable debug logger output to stdout")
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

			if debugLogFlag {
				debugLogger = logrus.New()
				debugLogger.SetLevel(logrus.DebugLevel)

				// initialize logger
				if err := os.MkdirAll(internal.LogDir(), 0755); err != nil {
					return fmt.Errorf("error creating log directories: %v", err)
				}
				debugLogger.Info("Initialized log directory: ", internal.LogDir())
				_, err := os.OpenFile(internal.LogPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
				if err != nil {
					return fmt.Errorf("error opening log file: %v", err)
				}
				debugLogger.Info("Initialized log file: ", internal.LogPath())

				// Follow the log file located at ~/.local/share/jarvis/daemon.log
				file, err := os.Open(internal.LogPath())
				if err != nil {
					debugLogger.Error(err)
					return err
				}
				debugLogger.Info("Following log file: ", internal.LogPath())
				defer file.Close()
				s := bufio.NewScanner(file)
				for {
					if s.Scan() {
						debugLogger.Info(s.Text())
					} else {
						if s.Err() != nil {
							debugLogger.Error("scanner error: " + s.Err().Error())
							break
						}
						// No new data, wait a bit and try again
						time.Sleep(1 * time.Second)
						s = bufio.NewScanner(file) // Recreate scanner to reset state
					}
				}
			}
			return nil
		}
	}

	return fmt.Errorf("daemon did not start within 5 seconds")
}
