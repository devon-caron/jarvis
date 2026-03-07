package search

import (
	"context"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"unicode"

	zim "github.com/tim-st/go-zim"
)

// zimNS is the namespace for article content in ZIM v6 files.
// ZIM v5 used 'A' (NamespaceArticles), but v6 moved all content to 'C'.
const zimNS = zim.Namespace('C')

// candidatePool is how many prefix-search results we gather per prefix before
// scoring. Large enough to find articles well past common redirect clusters.
const candidatePool = 200

// ZimSearcher searches Wikipedia ZIM archives.
type ZimSearcher struct {
	files      []*zim.File
	maxResults int
}

// NewZimSearcher opens the given ZIM files and returns a ZimSearcher.
func NewZimSearcher(paths []string, maxResults int) (*ZimSearcher, error) {
	if maxResults <= 0 {
		maxResults = 5
	}
	files := make([]*zim.File, 0, len(paths))
	for _, p := range paths {
		f, err := zim.Open(p)
		if err != nil {
			for _, opened := range files {
				opened.Close()
			}
			return nil, fmt.Errorf("failed to open ZIM file %q: %w", p, err)
		}
		files = append(files, f)
	}
	return &ZimSearcher{files: files, maxResults: maxResults}, nil
}

// Close releases all open ZIM file handles.
func (z *ZimSearcher) Close() {
	for _, f := range z.files {
		f.Close()
	}
}

var (
	htmlTagRE = regexp.MustCompile(`<[^>]+>`)
	styleRE   = regexp.MustCompile(`(?si)<style[^>]*>.*?</style>`)
	scriptRE  = regexp.MustCompile(`(?si)<script[^>]*>.*?</script>`)
)

func stripHTML(s string) string {
	s = htmlTagRE.ReplaceAllString(s, " ")
	return strings.Join(strings.Fields(s), " ")
}

// extractDescription removes inline style/script blocks, locates the article
// body, then returns a plain-text excerpt.
func extractDescription(html string, maxLen int) string {
	html = styleRE.ReplaceAllString(html, "")
	html = scriptRE.ReplaceAllString(html, "")

	lower := strings.ToLower(html)
	if bodyIdx := strings.Index(lower, "<body"); bodyIdx >= 0 {
		html = html[bodyIdx:]
		lower = lower[bodyIdx:]
	}
	if pIdx := strings.Index(lower, "<p"); pIdx >= 0 {
		html = html[pIdx:]
	}

	text := stripHTML(html)
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen] + "..."
}

// queryPrefixes returns URL prefixes to try for the query, longest first.
// Wikipedia URLs use underscores for spaces with the first letter capitalized.
// All contiguous word subsequences are returned so multi-word matches are tried
// before individual words.
func queryPrefixes(query string) []string {
	raw := strings.Fields(query)
	var words []string
	for _, w := range raw {
		w = strings.TrimFunc(w, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
		if len(w) < 2 {
			continue
		}
		runes := []rune(w)
		runes[0] = unicode.ToUpper(runes[0])
		words = append(words, string(runes))
	}
	if len(words) == 0 {
		return nil
	}

	seen := make(map[string]struct{})
	var prefixes []string
	for length := len(words); length >= 1; length-- {
		for start := 0; start+length <= len(words); start++ {
			p := strings.Join(words[start:start+length], "_")
			if _, ok := seen[p]; !ok {
				seen[p] = struct{}{}
				prefixes = append(prefixes, p)
			}
		}
	}
	return prefixes
}

// scoreEntry counts how many query words match the title.
// Exact substring matches score 2; stemmed matches (shared 4-char prefix) score 1.
func scoreEntry(title, query string) int {
	titleLower := strings.ToLower(title)
	score := 0
	for _, w := range strings.Fields(query) {
		w = strings.ToLower(strings.TrimFunc(w, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}))
		if len(w) < 2 {
			continue
		}
		if strings.Contains(titleLower, w) {
			score += 2
			continue
		}
		// Fuzzy stem: "crashes"→"crash", "crashing"→"crash"
		stem := w
		if len(stem) > 5 {
			stem = stem[:len(stem)-2]
		}
		if len(stem) >= 4 && strings.Contains(titleLower, stem) {
			score++
		}
	}
	return score
}

type scoredResult struct {
	result Result
	score  int
}

// Search finds Wikipedia articles matching the query across all ZIM files.
// It follows redirects so that alias URLs (e.g. "Stock_market_crashes") resolve
// to the canonical article ("Stock market crash") before scoring.
func (z *ZimSearcher) Search(ctx context.Context, query string) ([]Result, error) {
	prefixes := queryPrefixes(query)
	if len(prefixes) == 0 {
		return nil, nil
	}

	// seenURL deduplicates by the *resolved* article URL to avoid counting the
	// same article multiple times when reached via different redirect aliases.
	seenURL := make(map[string]struct{})
	var candidates []scoredResult

	for _, f := range z.files {
		if ctx.Err() != nil {
			break
		}
		for _, prefix := range prefixes {
			entries := f.EntriesWithURLPrefix(zimNS, []byte(prefix), candidatePool)
			for i := range entries {
				e := &entries[i]
				if e.IsDeletedEntry() || e.IsLinkTarget() {
					continue
				}

				// Follow redirects to get the canonical article.
				resolved := *e
				if e.IsRedirect() {
					target, err := f.FollowRedirect(e)
					if err != nil || target.IsRedirect() || target.IsDeletedEntry() {
						continue
					}
					resolved = target
				}

				url := string(resolved.URL())
				if _, dup := seenURL[url]; dup {
					continue
				}
				seenURL[url] = struct{}{}

				title := string(resolved.Title())
				score := scoreEntry(title, query)

				var desc string
				reader, _, err := f.BlobReader(&resolved)
				if err == nil {
					raw, readErr := io.ReadAll(io.LimitReader(reader, 32768))
					if readErr == nil {
						desc = extractDescription(string(raw), 500)
					}
				}

				candidates = append(candidates, scoredResult{
					result: Result{Title: title, URL: url, Description: desc},
					score:  score,
				})
			}
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].result.Title < candidates[j].result.Title
	})

	results := make([]Result, 0, z.maxResults)
	for _, c := range candidates {
		if len(results) >= z.maxResults {
			break
		}
		results = append(results, c.result)
	}
	return results, nil
}
