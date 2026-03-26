package pty

import (
	"fmt"
	"time"
)

type Session struct {
	ContextPath   string
	RingSize      int
	FlushInterval time.Duration

	ring *RingBuffer
}

func NewSession(contextPath string) *Session {
	return &Session{
		ContextPath:   contextPath,
		RingSize:      DefaultRingSize,
		FlushInterval: 1 * time.Second,
	}
}

func (s *Session) Run() error {
	// TODO: Implement PTY session logic
	return fmt.Errorf("session.Run(): unimplemented")
}
