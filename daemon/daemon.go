package daemon

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/devon-caron/jarvis/config"
	"github.com/devon-caron/jarvis/internal"
	"github.com/devon-caron/jarvis/search"
)

// Run is the daemon entry point. It sets up the PID file, socket server,
// signal handling, and auto-loads the default model if configured.
func Run() error {
	// Load config
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	// Set up logging
	logPath := internal.LogPath()
	if err := os.MkdirAll(filepath.Dir(logPath), 0755); err != nil {
		return fmt.Errorf("failed to create log directory: %w", err)
	}
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}
	defer logFile.Close()
	log.SetOutput(logFile)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Write PID file
	pidPath := internal.PIDPath()
	if err := WritePID(pidPath); err != nil {
		return fmt.Errorf("failed to write PID file: %w", err)
	}
	defer RemovePID(pidPath)

	log.Printf("jarvis daemon starting (pid=%d)", os.Getpid())

	// Create model manager with real llama backend
	backend := NewLlamaBackend(cfg)
	manager := NewModelManager(backend)
	defer manager.Shutdown()

	// Set up search
	var searcher search.Searcher
	if apiKey := cfg.SearchAPIKey(); apiKey != "" {
		searcher = search.NewBraveSearcher(apiKey, cfg.Search.MaxResults)
	}

	// Create handler and server
	stopCh := make(chan struct{}, 1)
	handler := NewHandler(manager, cfg, searcher, stopCh)
	server := NewServer(internal.SocketPath(), handler)

	if err := server.Listen(); err != nil {
		return err
	}
	defer server.Close()

	log.Printf("listening on %s", internal.SocketPath())

	// Auto-load default model if configured
	if cfg.DefaultModel != "" {
		modelPath := cfg.ResolveModel(cfg.DefaultModel)
		log.Printf("auto-loading default model: %s", modelPath)
		if err := manager.Load(modelPath, cfg.ModelOptions.GPULayers); err != nil {
			log.Printf("warning: failed to auto-load model: %v", err)
		} else {
			log.Printf("default model loaded successfully")
		}
	}

	// Signal handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Serve in a goroutine
	errCh := make(chan error, 1)
	go func() {
		errCh <- server.Serve()
	}()

	// Wait for stop signal or serve error
	select {
	case <-stopCh:
		log.Printf("stop requested, shutting down")
	case sig := <-sigCh:
		log.Printf("received signal %v, shutting down", sig)
	case err := <-errCh:
		if err != nil {
			log.Printf("server error: %v", err)
			return err
		}
	}

	server.Close()
	manager.Shutdown()
	log.Printf("daemon stopped")
	return nil
}
