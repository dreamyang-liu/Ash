package tools

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWebFetchFormatsAndHeaders(t *testing.T) {
	var sawUA, sawHeader bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawUA = strings.HasPrefix(r.Header.Get("User-Agent"), "AshRuntime/")
		sawHeader = r.Header.Get("X-Test") == "ash"
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, `<html><body><nav>skip nav</nav><h1>Hello</h1><p>Alpha <strong>Beta</strong></p><script>hidden()</script></body></html>`)
	}))
	defer server.Close()

	tool := &WebFetchTool{}

	html := tool.Execute(map[string]any{
		"url":     server.URL,
		"format":  "html",
		"headers": map[string]any{"X-Test": "ash"},
	})
	if !html.Success {
		t.Fatalf("html fetch failed: %s", html.Error)
	}
	if !sawUA || !sawHeader {
		t.Fatalf("server did not receive expected headers: user-agent=%v x-test=%v", sawUA, sawHeader)
	}
	if !strings.Contains(html.Output, "[status] 200 OK") ||
		!strings.Contains(html.Output, "[content_type] text/html") ||
		!strings.Contains(html.Output, "<h1>Hello</h1>") {
		t.Fatalf("unexpected html output:\n%s", html.Output)
	}

	text := tool.Execute(map[string]any{"url": server.URL, "format": "text"})
	if !text.Success {
		t.Fatalf("text fetch failed: %s", text.Error)
	}
	if !strings.Contains(text.Output, "Hello") ||
		!strings.Contains(text.Output, "Alpha Beta") ||
		strings.Contains(text.Output, "hidden()") ||
		strings.Contains(text.Output, "skip nav") {
		t.Fatalf("unexpected text output:\n%s", text.Output)
	}

	markdown := tool.Execute(map[string]any{"url": server.URL, "format": "markdown"})
	if !markdown.Success {
		t.Fatalf("markdown fetch failed: %s", markdown.Error)
	}
	if !strings.Contains(markdown.Output, "Hello") || !strings.Contains(markdown.Output, "Alpha") {
		t.Fatalf("unexpected markdown output:\n%s", markdown.Output)
	}
}

func TestWebFetchValidationAndTruncation(t *testing.T) {
	tool := &WebFetchTool{}

	missingURL := tool.Execute(map[string]any{})
	if missingURL.Success || !strings.Contains(missingURL.Error, "url is required") {
		t.Fatalf("expected missing url error, got: %#v", missingURL)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, strings.Repeat("x", 40))
	}))
	defer server.Close()

	truncated := tool.Execute(map[string]any{
		"url":        server.URL,
		"format":     "text",
		"max_length": float64(5),
	})
	if !truncated.Success {
		t.Fatalf("truncated fetch failed: %s", truncated.Error)
	}
	if !strings.Contains(truncated.Output, "xxxxx") || !strings.Contains(truncated.Output, "... (truncated,") {
		t.Fatalf("expected truncated output, got:\n%s", truncated.Output)
	}

	invalidFormat := tool.Execute(map[string]any{"url": server.URL, "format": "pdf"})
	if invalidFormat.Success || !strings.Contains(invalidFormat.Error, "invalid format: pdf") {
		t.Fatalf("expected invalid format error, got: %#v", invalidFormat)
	}
}

func TestWebSearchValidationBackendSelectionAndClamp(t *testing.T) {
	origSearch := ddgsTextSearch
	defer func() { ddgsTextSearch = origSearch }()

	type call struct {
		query   string
		backend string
		max     int
	}
	var calls []call
	ddgsTextSearch = func(query, backend string, maxResults int) ([]searchResult, error) {
		calls = append(calls, call{query: query, backend: backend, max: maxResults})
		results := make([]searchResult, 25)
		for i := range results {
			results[i] = searchResult{
				Title: fmt.Sprintf("%s title %02d", backend, i+1),
				URL:   fmt.Sprintf("https://example.com/%s/%02d", backend, i+1),
				Body:  "body",
			}
		}
		return results, nil
	}

	tool := &WebSearchTool{}

	missingQuery := tool.Execute(map[string]any{})
	if missingQuery.Success || !strings.Contains(missingQuery.Error, "query is required") {
		t.Fatalf("expected missing query error, got: %#v", missingQuery)
	}

	invalidBackend := tool.Execute(map[string]any{"query": "ash", "backend": "bad"})
	if invalidBackend.Success || !strings.Contains(invalidBackend.Error, "invalid backend: bad") {
		t.Fatalf("expected invalid backend error, got: %#v", invalidBackend)
	}
	if len(calls) != 0 {
		t.Fatalf("invalid backend should not reach the search layer: %#v", calls)
	}

	result := tool.Execute(map[string]any{
		"query":       "ash runtime",
		"backend":     "duckduckgo",
		"max_results": float64(200),
	})
	if !result.Success {
		t.Fatalf("web search failed: %s", result.Error)
	}
	if len(calls) != 1 || calls[0] != (call{query: "ash runtime", backend: "duckduckgo", max: 20}) {
		t.Fatalf("unexpected search calls: %#v", calls)
	}
	if !strings.Contains(result.Output, "1. duckduckgo title 01") ||
		!strings.Contains(result.Output, "20. duckduckgo title 20") ||
		strings.Contains(result.Output, "21. duckduckgo title 21") {
		t.Fatalf("unexpected search output:\n%s", result.Output)
	}

	// New ddgs backends are accepted by schema validation.
	for _, backend := range []string{"wikipedia", "startpage", "yandex"} {
		r := tool.Execute(map[string]any{"query": "ash", "backend": backend})
		if !r.Success {
			t.Fatalf("backend %s rejected: %s", backend, r.Error)
		}
	}
}

func TestWebSearchErrorPropagation(t *testing.T) {
	origSearch := ddgsTextSearch
	defer func() { ddgsTextSearch = origSearch }()

	ddgsTextSearch = func(string, string, int) ([]searchResult, error) {
		return nil, fmt.Errorf("boom")
	}
	tool := &WebSearchTool{}
	r := tool.Execute(map[string]any{"query": "ash"})
	if r.Success || !strings.Contains(r.Error, "all search engines failed: boom") {
		t.Fatalf("expected propagated error, got: %#v", r)
	}
}
