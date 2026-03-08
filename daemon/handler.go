package daemon

import "io"

type ResponseWriter struct {
	w io.Writer
}

func NewResponseWriter(w io.Writer) *ResponseWriter {
	return &ResponseWriter{w: w}
}
