package search

import (
	"fmt"
	"strings"
)

// FormatResults formats search results into a system message for the model.
func FormatResults(results []Result) string {
	if len(results) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("Wikipedia search results:\n\n")
	for i, r := range results {
		b.WriteString(fmt.Sprintf("%d. %s\n   URL: %s\n   %s\n\n", i+1, r.Title, r.URL, r.Description))
	}
	b.WriteString("Use the above Wikipedia articles to inform your response.")
	return b.String()
}
