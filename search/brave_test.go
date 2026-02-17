package search

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestBraveSearcher_Search(t *testing.T) {
	// Create mock Brave API server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Header.Get("X-Subscription-Token") != "test-key" {
			t.Errorf("missing or wrong API key header")
		}
		if r.URL.Query().Get("q") != "test query" {
			t.Errorf("query = %q, want 'test query'", r.URL.Query().Get("q"))
		}
		if r.URL.Query().Get("count") != "3" {
			t.Errorf("count = %q, want 3", r.URL.Query().Get("count"))
		}

		resp := braveResponse{
			Web: braveWebResults{
				Results: []braveWebResult{
					{Title: "Result 1", URL: "https://a.com", Description: "First result"},
					{Title: "Result 2", URL: "https://b.com", Description: "Second result"},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	searcher := &BraveSearcher{
		apiKey:     "test-key",
		maxResults: 3,
		client:     server.Client(),
	}

	// Override URL for testing by using a custom client with transport
	originalURL := braveSearchURL
	// We need to intercept the URL, so let's create a custom transport
	searcher.client = &http.Client{
		Transport: &testTransport{
			base:      http.DefaultTransport,
			serverURL: server.URL,
		},
	}
	_ = originalURL

	results, err := searcher.Search(context.Background(), "test query")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if results[0].Title != "Result 1" {
		t.Errorf("results[0].Title = %q", results[0].Title)
	}
	if results[1].URL != "https://b.com" {
		t.Errorf("results[1].URL = %q", results[1].URL)
	}
}

// testTransport redirects requests to the test server.
type testTransport struct {
	base      http.RoundTripper
	serverURL string
}

func (t *testTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Redirect to test server, preserving query and headers
	req.URL.Scheme = "http"
	req.URL.Host = t.serverURL[len("http://"):]
	return t.base.RoundTrip(req)
}

func TestBraveSearcher_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte("invalid key"))
	}))
	defer server.Close()

	searcher := &BraveSearcher{
		apiKey:     "bad-key",
		maxResults: 5,
		client: &http.Client{
			Transport: &testTransport{
				base:      http.DefaultTransport,
				serverURL: server.URL,
			},
		},
	}

	_, err := searcher.Search(context.Background(), "test")
	if err == nil {
		t.Error("expected error for 401 response")
	}
}

func TestBraveSearcher_EmptyResults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := braveResponse{Web: braveWebResults{Results: nil}}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	searcher := &BraveSearcher{
		apiKey:     "test-key",
		maxResults: 5,
		client: &http.Client{
			Transport: &testTransport{
				base:      http.DefaultTransport,
				serverURL: server.URL,
			},
		},
	}

	results, err := searcher.Search(context.Background(), "no results")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestNewBraveSearcher_DefaultMaxResults(t *testing.T) {
	s := NewBraveSearcher("key", 0)
	if s.maxResults != 5 {
		t.Errorf("maxResults = %d, want 5", s.maxResults)
	}

	s = NewBraveSearcher("key", -1)
	if s.maxResults != 5 {
		t.Errorf("maxResults = %d, want 5", s.maxResults)
	}
}
