package ddgs

// Encyclopedia engines: wikipedia and grokipedia.
// Mirrors ddgs/engines/wikipedia.py and ddgs/engines/grokipedia.py.
// Called from registry.go (engine factory table); user instruction:
// "帮我写一个golang的library，主要就是把 https://github.com/deedy5/ddgs 的功能复刻出来".

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// newWikipedia mirrors ddgs.engines.wikipedia.Wikipedia (priority 2).
func newWikipedia(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "wikipedia",
		category:     "text",
		provider:     "wikipedia",
		priority:     2,
		searchMethod: "GET",
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.searchURLFn = func(p searchParams) string {
		_, lang := splitRegion(p.Region)
		return fmt.Sprintf(
			"https://%s.wikipedia.org/w/api.php?action=opensearch&profile=fuzzy&limit=1&search=%s",
			lang, url.QueryEscape(p.Query),
		)
	}
	e.buildPayload = func(_ context.Context, _ searchParams) (map[string]string, error) {
		return map[string]string{}, nil
	}
	e.extractResults = func(ctx context.Context, text string, p searchParams) ([]Result, error) {
		// opensearch response: [query, [titles], [descriptions], [urls]]
		var payload []any
		if err := json.Unmarshal([]byte(text), &payload); err != nil {
			return nil, fmt.Errorf("%w: parsing wikipedia json: %v", ErrDDGS, err)
		}
		if len(payload) < 4 {
			return nil, nil
		}
		titles, _ := payload[1].([]any)
		urls, _ := payload[3].([]any)
		if len(titles) == 0 || len(urls) == 0 {
			return nil, nil
		}

		result := newResult("text")
		title, _ := titles[0].(string)
		href, _ := urls[0].(string)
		result.Set("title", title)
		result.Set("href", href)

		// Fetch the article extract as body.
		_, lang := splitRegion(p.Region)
		extractURL := fmt.Sprintf(
			"https://%s.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles=%s&explaintext=0&exintro=0&redirects=1",
			lang, url.QueryEscape(result.getString("title")),
		)
		respText, err := e.requestText(ctx, "GET", extractURL, nil, nil)
		if err == nil && respText != "" {
			var extractPayload struct {
				Query struct {
					Pages map[string]struct {
						Extract string `json:"extract"`
					} `json:"pages"`
				} `json:"query"`
			}
			if json.Unmarshal([]byte(respText), &extractPayload) == nil {
				for _, page := range extractPayload.Query.Pages {
					result.Set("body", page.Extract)
					break
				}
			}
		}
		if strings.Contains(result.getString("body"), "may refer to:") {
			return nil, nil
		}
		return []Result{result}, nil
	}
	return e, nil
}

// newGrokipedia mirrors ddgs.engines.grokipedia.Grokipedia (priority 1.9).
func newGrokipedia(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "grokipedia",
		category:     "text",
		provider:     "grokipedia",
		priority:     1.9,
		searchURL:    "https://grokipedia.com/api/typeahead",
		searchMethod: "GET",
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		return map[string]string{"query": p.Query, "limit": "1"}, nil
	}
	e.extractResults = func(_ context.Context, text string, _ searchParams) ([]Result, error) {
		var payload struct {
			Results []struct {
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
				Slug    string `json:"slug"`
			} `json:"results"`
		}
		if err := json.Unmarshal([]byte(text), &payload); err != nil {
			return nil, fmt.Errorf("%w: parsing grokipedia json: %v", ErrDDGS, err)
		}
		if len(payload.Results) == 0 {
			return nil, nil
		}
		item := payload.Results[0]
		result := newResult("text")
		result.Set("title", strings.Trim(item.Title, "_"))
		body := item.Snippet
		if idx := strings.Index(body, "\n\n"); idx >= 0 {
			body = body[idx+2:]
		}
		result.Set("body", body)
		result.Set("href", "https://grokipedia.com/page/"+item.Slug)
		return []Result{result}, nil
	}
	return e, nil
}
