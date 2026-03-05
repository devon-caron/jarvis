package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/client"
	"github.com/devon-caron/jarvis/daemon"
	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/protocol"
	jarvispty "github.com/devon-caron/jarvis/pty"
)

var backgroundFlag bool

var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the jarvis daemon and PTY shell",
	Long:  "Start the jarvis daemon and open an interactive PTY shell with terminal context capture.\nUse -b/--background for daemon-only mode (current behavior).",
	RunE:  runStart,
}

func init() {
	startCmd.Flags().BoolVarP(&backgroundFlag, "background", "b", false, "Start daemon only (background mode)")
	rootCmd.AddCommand(startCmd)
}

func runStart(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	if backgroundFlag {
		return startDaemonProcess()
	}

	return runStartPTY()
}

// startDaemonProcess starts the daemon as a background process (original behavior).
func startDaemonProcess() error {
	pidPath := internal.PIDPath()

	if daemon.IsRunning(pidPath) {
		pid, _ := daemon.ReadPID(pidPath)
		return fmt.Errorf("daemon already running (pid %d)", pid)
	}

	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("cannot find executable: %w", err)
	}

	daemonCmd := exec.Command(exe, "_daemon")
	daemonCmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid: true,
	}
	daemonCmd.Env = os.Environ()

	if err := daemonCmd.Start(); err != nil {
		return fmt.Errorf("failed to start daemon: %w", err)
	}

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

// runStartPTY starts the daemon (if needed), writes the PTY PID file, and
// enters an interactive PTY shell with terminal context capture.
func runStartPTY() error {
	pidPath := internal.PIDPath()
	daemonStartedByUs := false

	// Start daemon if not already running
	if !daemon.IsRunning(pidPath) {
		if err := startDaemonProcess(); err != nil {
			return err
		}
		daemonStartedByUs = true
	} else {
		pid, _ := daemon.ReadPID(pidPath)
		fmt.Printf("jarvis daemon already running (pid %d)\n", pid)
	}

	// Write PTY PID file
	ptyPIDPath := internal.PTYPIDPath()
	if err := os.WriteFile(ptyPIDPath, []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
		return fmt.Errorf("failed to write PTY PID file: %w", err)
	}
	defer os.Remove(ptyPIDPath)

	// Create and run the PTY session
	contextPath := internal.PTYContextPath()
	session := jarvispty.NewSession(contextPath)

	fmt.Println("jarvis PTY shell started (type 'exit' to quit)")
	err := session.Run()

	// Clean up context file
	os.Remove(contextPath)

	// Stop daemon if we started it
	if daemonStartedByUs {
		fmt.Println("stopping jarvis daemon...")
		stopDaemonQuietly()
	}

	return err
}

// stopDaemonQuietly attempts to stop the daemon via the socket protocol,
// suppressing errors (used during PTY cleanup).
func stopDaemonQuietly() {
	c, err := client.Connect()
	if err != nil {
		return
	}
	defer c.Close()
	c.SendAndWaitOK(&protocol.Request{Type: protocol.ReqStop})
}
