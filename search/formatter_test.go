package search

import (
	"strings"
	"testing"
)

func TestFormatResults_Empty(t *testing.T) {
	result := FormatResults(nil)
	if result != "" {
		t.Errorf("FormatResults(nil) = %q, want empty", result)
	}

	result = FormatResults([]Result{})
	if result != "" {
		t.Errorf("FormatResults([]) = %q, want empty", result)
	}
}

func TestFormatResults_Single(t *testing.T) {
	results := []Result{
		{Title: "Go Programming", URL: "https://go.dev", Description: "The Go programming language"},
	}

	formatted := FormatResults(results)

	if !strings.Contains(formatted, "Wikipedia search results:") {
		t.Error("should contain header")
	}
	if !strings.Contains(formatted, "1. Go Programming") {
		t.Error("should contain numbered title")
	}
	if !strings.Contains(formatted, "URL: https://go.dev") {
		t.Error("should contain URL")
	}
	if !strings.Contains(formatted, "The Go programming language") {
		t.Error("should contain description")
	}
	if !strings.Contains(formatted, "Wikipedia articles") {
		t.Error("should contain Wikipedia reference")
	}
}

func TestFormatResults_Multiple(t *testing.T) {
	results := []Result{
		{Title: "Result 1", URL: "https://a.com", Description: "First"},
		{Title: "Result 2", URL: "https://b.com", Description: "Second"},
		{Title: "Result 3", URL: "https://c.com", Description: "Third"},
	}

	formatted := FormatResults(results)

	if !strings.Contains(formatted, "1. Result 1") {
		t.Error("should contain result 1")
	}
	if !strings.Contains(formatted, "2. Result 2") {
		t.Error("should contain result 2")
	}
	if !strings.Contains(formatted, "3. Result 3") {
		t.Error("should contain result 3")
	}
}

func TestFormatResults_SpecialCharacters(t *testing.T) {
	results := []Result{
		{Title: "Go <1.25> & more", URL: "https://test.com/a?b=c&d=e", Description: "Test with \"quotes\""},
	}

	formatted := FormatResults(results)

	if !strings.Contains(formatted, "Go <1.25> & more") {
		t.Error("should preserve special characters in title")
	}
	if !strings.Contains(formatted, "https://test.com/a?b=c&d=e") {
		t.Error("should preserve URL with query params")
	}
}
