package pty

import "sync"

const DefaultRingSize = 64 * 1024 // 64KB

// RingBuffer is a thread-safe circular byte buffer that implements io.Writer.
// When full, new writes overwrite the oldest data.
type RingBuffer struct {
	mu   sync.Mutex
	buf  []byte
	size int
	w    int  // next write position
	full bool // buffer has wrapped at least once
}

// NewRingBuffer creates a ring buffer with the given capacity.
func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{
		buf:  make([]byte, size),
		size: size,
	}
}

// Write appends p to the ring buffer, overwriting oldest data if necessary.
func (r *RingBuffer) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	n := len(p)
	if n >= r.size {
		copy(r.buf, p[n-r.size:])
		r.w = 0
		r.full = true
		return n, nil
	}

	space := r.size - r.w
	if n <= space {
		copy(r.buf[r.w:], p)
	} else {
		copy(r.buf[r.w:], p[:space])
		copy(r.buf, p[space:])
		r.full = true
	}

	oldW := r.w
	r.w = (r.w + n) % r.size
	// Detect wrap: if write position moved backwards or we exactly filled to end
	if r.w <= oldW && n > 0 {
		r.full = true
	}

	return n, nil
}

// Bytes returns the current contents of the buffer in order (oldest to newest).
func (r *RingBuffer) Bytes() []byte {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.full {
		out := make([]byte, r.w)
		copy(out, r.buf[:r.w])
		return out
	}

	out := make([]byte, r.size)
	n := copy(out, r.buf[r.w:])
	copy(out[n:], r.buf[:r.w])
	return out
}

// Reset clears the buffer.
func (r *RingBuffer) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.w = 0
	r.full = false
}

// Len returns the number of bytes currently stored.
func (r *RingBuffer) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.full {
		return r.size
	}
	return r.w
}
