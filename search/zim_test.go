package search

import (
	"context"
	"os"
	"strings"
	"testing"
)

const testZimPath = "/Users/devoncaron/ai/wikipedia/wikipedia_en_top_nopic_2025-12.zim"

func requireZimFile(t *testing.T) *ZimSearcher {
	t.Helper()
	if _, err := os.Stat(testZimPath); os.IsNotExist(err) {
		t.Skipf("ZIM file not found at %s", testZimPath)
	}
	s, err := NewZimSearcher([]string{testZimPath}, 5)
	if err != nil {
		t.Fatalf("NewZimSearcher: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

// --- Unit tests (no ZIM file required) ---

func TestQueryPrefixes_Empty(t *testing.T) {
	if got := queryPrefixes(""); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
	if got := queryPrefixes("  "); got != nil {
		t.Errorf("expected nil for whitespace, got %v", got)
	}
}

func TestQueryPrefixes_SingleWord(t *testing.T) {
	got := queryPrefixes("photosynthesis")
	if len(got) != 1 || got[0] != "Photosynthesis" {
		t.Errorf("queryPrefixes(photosynthesis) = %v, want [Photosynthesis]", got)
	}
}

func TestQueryPrefixes_MultiWord(t *testing.T) {
	got := queryPrefixes("stock market crash")
	// Longest combos first: Stock_market_crash, Market_crash, Crash, Stock_market, Market, Stock
	if len(got) == 0 {
		t.Fatal("expected non-empty prefixes")
	}
	if got[0] != "Stock_Market_Crash" {
		t.Errorf("first prefix = %q, want Stock_Market_Crash (each word capitalized)", got[0])
	}
	// All three individual words must appear somewhere
	prefixSet := make(map[string]bool)
	for _, p := range got {
		prefixSet[p] = true
	}
	for _, want := range []string{"Stock", "Market", "Crash"} {
		if !prefixSet[want] {
			t.Errorf("missing individual word prefix %q", want)
		}
	}
}

func TestQueryPrefixes_CapitalizesFirstLetter(t *testing.T) {
	got := queryPrefixes("albert einstein")
	if len(got) == 0 {
		t.Fatal("expected prefixes")
	}
	if got[0] != "Albert_Einstein" {
		t.Errorf("first prefix = %q, want Albert_Einstein (each word capitalized)", got[0])
	}
}

func TestQueryPrefixes_SkipsShortWords(t *testing.T) {
	// Single-character words are skipped (len < 2); multi-char words like "the" are kept.
	got := queryPrefixes("a stock b")
	prefixSet := make(map[string]bool)
	for _, p := range got {
		prefixSet[p] = true
	}
	if prefixSet["A"] || prefixSet["B"] {
		t.Error("should skip single-character words")
	}
	if !prefixSet["Stock"] {
		t.Error("should include 'Stock'")
	}
}

func TestScoreEntry_ExactMatch(t *testing.T) {
	score := scoreEntry("Stock market crash", "stock market crash")
	// All 3 words match exactly (2 pts each) = 6
	if score != 6 {
		t.Errorf("score = %d, want 6", score)
	}
}

func TestScoreEntry_StemMatch(t *testing.T) {
	// "crashes" → stem "crash" → matches "crash" in title
	score := scoreEntry("Stock market crash", "stock market crashes")
	// "stock"=2, "market"=2, "crashes"→stem→1 = 5
	if score != 5 {
		t.Errorf("score = %d, want 5", score)
	}
}

func TestScoreEntry_NoMatch(t *testing.T) {
	score := scoreEntry("Quantum mechanics", "photosynthesis")
	if score != 0 {
		t.Errorf("score = %d, want 0", score)
	}
}

func TestExtractDescription_StripsStyle(t *testing.T) {
	html := "<html><head><style>.foo { color: red; }</style></head><body><p>Hello world.</p></body></html>"
	desc := extractDescription(html, 500)
	if strings.Contains(desc, ".foo") {
		t.Error("description should not contain CSS rules")
	}
	if !strings.Contains(desc, "Hello world") {
		t.Errorf("description should contain article text, got: %q", desc)
	}
}

func TestNewZimSearcher_BadPath(t *testing.T) {
	_, err := NewZimSearcher([]string{"/nonexistent/file.zim"}, 5)
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

// --- Integration tests (require ZIM file) ---

func TestZimSearch_Photosynthesis(t *testing.T) {
	s := requireZimFile(t)
	results, err := s.Search(context.Background(), "photosynthesis")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}
	top := results[0]
	if top.Title != "Photosynthesis" {
		t.Errorf("top result title = %q, want Photosynthesis", top.Title)
	}
	if top.URL != "Photosynthesis" {
		t.Errorf("top result URL = %q, want Photosynthesis", top.URL)
	}
	if len(top.Description) == 0 {
		t.Error("description should be non-empty")
	}
	if strings.Contains(top.Description, ".mw-parser-output") {
		t.Errorf("description contains raw CSS, extraction failed: %q", top.Description[:min(100, len(top.Description))])
	}
}

func TestZimSearch_AlbertEinstein(t *testing.T) {
	s := requireZimFile(t)
	results, err := s.Search(context.Background(), "Albert Einstein")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.Title == "Albert Einstein" {
			found = true
			if len(r.Description) == 0 {
				t.Error("Albert Einstein description should be non-empty")
			}
		}
	}
	if !found {
		titles := make([]string, len(results))
		for i, r := range results {
			titles[i] = r.Title
		}
		t.Errorf("Albert Einstein not in results: %v", titles)
	}
}

func TestZimSearch_StockMarketCrash(t *testing.T) {
	s := requireZimFile(t)
	results, err := s.Search(context.Background(), "stock market crash")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.Title == "Stock market crash" {
			found = true
			if len(r.Description) == 0 {
				t.Error("Stock market crash description should be non-empty")
			}
			if strings.Contains(r.Description, ".mw-parser-output") {
				t.Errorf("description contains raw CSS: %q", r.Description[:min(100, len(r.Description))])
			}
			break
		}
	}
	if !found {
		titles := make([]string, len(results))
		for i, r := range results {
			titles[i] = r.Title
		}
		t.Errorf("'Stock market crash' not found; got: %v", titles)
	}
}

func TestZimSearch_HistoricalStockCrashes(t *testing.T) {
	// This phrasing doesn't match any article title exactly — the searcher must
	// use stemming ("crashes"→"crash") and redirect-following to surface
	// the canonical "Stock market crash" article.
	s := requireZimFile(t)
	results, err := s.Search(context.Background(), "historical stock crashes")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if strings.Contains(strings.ToLower(r.Title), "stock market crash") ||
			strings.Contains(strings.ToLower(r.Title), "stock market crash") {
			found = true
			break
		}
	}
	if !found {
		titles := make([]string, len(results))
		for i, r := range results {
			titles[i] = r.Title
		}
		t.Errorf("no stock market crash variant in results for 'historical stock crashes': %v", titles)
	}
}

func TestZimSearch_MaxResults(t *testing.T) {
	s := requireZimFile(t)
	results, err := s.Search(context.Background(), "world")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) > 5 {
		t.Errorf("got %d results, want at most 5 (maxResults)", len(results))
	}
}

func TestZimSearch_ContextCancelled(t *testing.T) {
	s := requireZimFile(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately
	// Should return without hanging, possibly empty results.
	_, err := s.Search(ctx, "photosynthesis")
	if err != nil {
		t.Errorf("cancelled context should not return error, got: %v", err)
	}
}

func TestZimSearch_EmptyQuery(t *testing.T) {
	s := requireZimFile(t)
	results, err := s.Search(context.Background(), "")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("empty query should return 0 results, got %d", len(results))
	}
}

func TestZimSearch_ResultsHaveDescriptions(t *testing.T) {
	s := requireZimFile(t)
	results, err := s.Search(context.Background(), "Albert Einstein")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range results {
		if r.Title == "Albert Einstein" && len(r.Description) < 50 {
			t.Errorf("Albert Einstein description too short (%d chars): %q", len(r.Description), r.Description)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
