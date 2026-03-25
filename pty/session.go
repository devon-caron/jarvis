package pty

import "fmt"

type Session struct {
	contextPath string
}

func NewSession(contextPath string) *Session {
	return &Session{
		contextPath: contextPath,
	}
}

func (s *Session) Run() error {
	// TODO: Implement PTY session logic
	return fmt.Errorf("session.Run(): unimplemented")
}
