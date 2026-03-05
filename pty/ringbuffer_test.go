package pty

import (
	"io"
	"sync"
	"testing"
)

func TestRingBuffer_BasicWrite(t *testing.T) {
	rb := NewRingBuffer(16)
	rb.Write([]byte("hello"))
	if got := string(rb.Bytes()); got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
	if rb.Len() != 5 {
		t.Errorf("Len() = %d, want 5", rb.Len())
	}
}

func TestRingBuffer_ExactCapacity(t *testing.T) {
	rb := NewRingBuffer(5)
	rb.Write([]byte("12345"))
	if got := string(rb.Bytes()); got != "12345" {
		t.Errorf("got %q, want %q", got, "12345")
	}
	if rb.Len() != 5 {
		t.Errorf("Len() = %d, want 5", rb.Len())
	}
}

func TestRingBuffer_Overflow(t *testing.T) {
	rb := NewRingBuffer(5)
	rb.Write([]byte("1234567"))
	// Should keep last 5 bytes
	if got := string(rb.Bytes()); got != "34567" {
		t.Errorf("got %q, want %q", got, "34567")
	}
}

func TestRingBuffer_MultipleWrites(t *testing.T) {
	rb := NewRingBuffer(10)
	rb.Write([]byte("hello"))
	rb.Write([]byte("world"))
	if got := string(rb.Bytes()); got != "helloworld" {
		t.Errorf("got %q, want %q", got, "helloworld")
	}
}

func TestRingBuffer_WrapAround(t *testing.T) {
	rb := NewRingBuffer(8)
	rb.Write([]byte("abcde"))   // [a b c d e _ _ _], w=5
	rb.Write([]byte("fghij"))   // wraps: [i j c d e f g h], w=2, keeps oldest from wrap
	got := string(rb.Bytes())
	if got != "cdefghij" {
		t.Errorf("got %q, want %q", got, "cdefghij")
	}
}

func TestRingBuffer_ConcurrentAccess(t *testing.T) {
	rb := NewRingBuffer(1024)
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rb.Write([]byte("data"))
			rb.Bytes()
			rb.Len()
		}()
	}
	wg.Wait()
	// No race condition = pass
}

func TestRingBuffer_LargeWrite(t *testing.T) {
	rb := NewRingBuffer(16)
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte('a' + (i % 26))
	}
	n, err := rb.Write(data)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1024 {
		t.Errorf("Write returned %d, want 1024", n)
	}
	got := rb.Bytes()
	if len(got) != 16 {
		t.Errorf("Bytes() len = %d, want 16", len(got))
	}
	// Should contain last 16 bytes of data
	want := string(data[1024-16:])
	if string(got) != want {
		t.Errorf("got %q, want %q", string(got), want)
	}
}

func TestRingBuffer_Reset(t *testing.T) {
	rb := NewRingBuffer(16)
	rb.Write([]byte("hello"))
	rb.Reset()
	if rb.Len() != 0 {
		t.Errorf("Len() after Reset = %d, want 0", rb.Len())
	}
	if got := rb.Bytes(); len(got) != 0 {
		t.Errorf("Bytes() after Reset = %q, want empty", string(got))
	}
}

func TestRingBuffer_IOWriterInterface(t *testing.T) {
	rb := NewRingBuffer(64)
	var w io.Writer = rb
	_, err := w.Write([]byte("test"))
	if err != nil {
		t.Fatal(err)
	}
	if got := string(rb.Bytes()); got != "test" {
		t.Errorf("got %q, want %q", got, "test")
	}
}
