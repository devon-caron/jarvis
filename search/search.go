package search

import "context"

// Result represents a single search result.
type Result struct {
	Title       string `json:"title"`
	URL         string `json:"url"`
	Description string `json:"description"`
}

// Searcher performs web searches.
type Searcher interface {
	Search(ctx context.Context, query string) ([]Result, error)
}
