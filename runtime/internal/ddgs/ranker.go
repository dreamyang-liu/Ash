package ddgs

import (
	"regexp"
	"strings"
)

const minTokenLength = 3

var tokenSplitter = regexp.MustCompile(`\W+`)

// simpleFilterRanker mirrors ddgs.similarity.SimpleFilterRanker:
//
//  1. Pull any doc with 'wikipedia.org' in its href to the top.
//  2. Bucket the rest according to where query tokens appear:
//     both title & body/description, title only, body only, neither.
//  3. Return wikipedia-top + both + title-only + body-only + neither.
type simpleFilterRanker struct{}

func (simpleFilterRanker) extractTokens(query string) []string {
	seen := map[string]bool{}
	var tokens []string
	for _, tok := range tokenSplitter.Split(strings.ToLower(query), -1) {
		if len(tok) >= minTokenLength && !seen[tok] {
			seen[tok] = true
			tokens = append(tokens, tok)
		}
	}
	return tokens
}

func (simpleFilterRanker) hasAnyToken(text string, tokens []string) bool {
	lower := strings.ToLower(text)
	for _, tok := range tokens {
		if strings.Contains(lower, tok) {
			return true
		}
	}
	return false
}

func (r simpleFilterRanker) rank(docs []Result, query string) []Result {
	tokens := r.extractTokens(query)

	var wikiHits, both, titleOnly, bodyOnly, neither []Result
	for _, doc := range docs {
		href := doc.getString("href")
		title := doc.getString("title")
		body := doc.getString("body")
		if body == "" {
			body = doc.getString("description")
		}

		// Skip Wikimedia category pages
		if strings.Contains(title, "Category:") && strings.Contains(title, "Wikimedia") {
			continue
		}

		if strings.Contains(href, "wikipedia.org") {
			wikiHits = append(wikiHits, doc)
			continue
		}

		hitTitle := r.hasAnyToken(title, tokens)
		hitBody := r.hasAnyToken(body, tokens)
		switch {
		case hitTitle && hitBody:
			both = append(both, doc)
		case hitTitle:
			titleOnly = append(titleOnly, doc)
		case hitBody:
			bodyOnly = append(bodyOnly, doc)
		default:
			neither = append(neither, doc)
		}
	}

	ranked := make([]Result, 0, len(docs))
	ranked = append(ranked, wikiHits...)
	ranked = append(ranked, both...)
	ranked = append(ranked, titleOnly...)
	ranked = append(ranked, bodyOnly...)
	ranked = append(ranked, neither...)
	return ranked
}
