package cmd

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/daemon"
)

var (
	workerSocket string
	workerPath   string
	workerName   string
)

var workerCmd = &cobra.Command{
	Use:    "_worker",
	Hidden: true,
	RunE:   runWorker,
}

func init() {
	workerCmd.Flags().StringVar(&workerSocket, "socket", "", "Unix socket path to listen on")
	workerCmd.Flags().StringVar(&workerPath, "path", "", "Model file path to load")
	workerCmd.Flags().StringVar(&workerName, "name", "", "Model name for registry (defaults to path)")
	workerCmd.MarkFlagRequired("socket")
	workerCmd.MarkFlagRequired("path")
	rootCmd.AddCommand(workerCmd)
}

func runWorker(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true

	name := workerName
	if name == "" {
		name = workerPath
	}

	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	// Use LlamaBackend directly — the worker is the leaf that does real inference.
	newBackend := func(c *config.Config) daemon.ModelBackend {
		return daemon.NewLlamaBackend(c)
	}
	registry := daemon.NewModelRegistry(cfg, newBackend)
	defer registry.Shutdown()

	// Load the model; CUDA_VISIBLE_DEVICES already restricts which GPUs are visible.
	if err := registry.Load(name, workerPath, nil, 0); err != nil {
		return fmt.Errorf("worker failed to load model: %v", err)
	}

	// Signal readiness to the parent via fd 3.
	readyFd := os.NewFile(3, "ready")
	readyFd.Write([]byte{1})
	readyFd.Close()

	// Serve requests until ReqStop or a signal.
	stopCh := make(chan struct{}, 1)
	handler := daemon.NewHandler(registry, cfg, nil, stopCh)
	server := daemon.NewServer(workerSocket, handler)

	if err := server.Listen(); err != nil {
		return err
	}
	defer server.Close()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.Serve()
	}()

	select {
	case <-stopCh:
	case <-sigCh:
	case err := <-errCh:
		return err
	}

	return nil
}
