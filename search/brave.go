package search

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

const braveSearchURL = "https://api.search.brave.com/res/v1/web/search"

// BraveSearcher queries the Brave Search API.
type BraveSearcher struct {
	apiKey     string
	maxResults int
	client     *http.Client
}

// NewBraveSearcher creates a new BraveSearcher with the given API key.
func NewBraveSearcher(apiKey string, maxResults int) *BraveSearcher {
	if maxResults <= 0 {
		maxResults = 5
	}
	return &BraveSearcher{
		apiKey:     apiKey,
		maxResults: maxResults,
		client:     http.DefaultClient,
	}
}

func (b *BraveSearcher) Search(ctx context.Context, query string) ([]Result, error) {
	u, err := url.Parse(braveSearchURL)
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Set("q", query)
	q.Set("count", fmt.Sprintf("%d", b.maxResults))
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Accept-Encoding", "gzip")
	req.Header.Set("X-Subscription-Token", b.apiKey)

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search API returned %d: %s", resp.StatusCode, string(body))
	}

	var result braveResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode search response: %w", err)
	}

	var results []Result
	for _, r := range result.Web.Results {
		results = append(results, Result{
			Title:       r.Title,
			URL:         r.URL,
			Description: r.Description,
		})
	}
	return results, nil
}

// braveResponse is the top-level Brave Search API response.
type braveResponse struct {
	Web braveWebResults `json:"web"`
}

type braveWebResults struct {
	Results []braveWebResult `json:"results"`
}

type braveWebResult struct {
	Title       string `json:"title"`
	URL         string `json:"url"`
	Description string `json:"description"`
}
