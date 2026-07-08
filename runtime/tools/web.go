package tools

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/PuerkitoBio/goquery"
)

const defaultUA = "AshRuntime/1.0 (+https://github.com/dreamyang-liu/ash)"

const (
	defaultWebTimeoutSeconds = 15
	maxWebTimeoutSeconds     = 60
	defaultWebMaxLength      = 10000
	maxWebMaxLength          = 200000
)

// ---- WebFetchTool ----

type WebFetchTool struct{}

func (w *WebFetchTool) Name() string { return "web_fetch" }

func (w *WebFetchTool) Description() string {
	return "Fetch a URL and return its content in the specified format"
}

func (w *WebFetchTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url":        map[string]any{"type": "string", "description": "URL to fetch"},
			"format":     map[string]any{"type": "string", "enum": []string{"html", "text", "markdown"}, "default": "markdown"},
			"headers":    map[string]any{"type": "object", "description": "Additional HTTP headers"},
			"timeout":    map[string]any{"type": "integer", "minimum": 1, "maximum": maxWebTimeoutSeconds, "default": defaultWebTimeoutSeconds, "description": "Request timeout in seconds"},
			"max_length": map[string]any{"type": "integer", "minimum": 1, "maximum": maxWebMaxLength, "default": defaultWebMaxLength, "description": "Maximum returned characters"},
		},
		"required": []string{"url"},
	}
}

func (w *WebFetchTool) Execute(args map[string]any) Result {
	rawURL, _ := args["url"].(string)
	if rawURL == "" {
		return Err("url is required")
	}
	format := "markdown"
	if f, ok := args["format"].(string); ok && f != "" {
		format = f
	}
	maxLen := defaultWebMaxLength
	if m, ok := args["max_length"].(float64); ok && int(m) > 0 {
		maxLen = clampInt(int(m), 1, maxWebMaxLength)
	}
	timeoutSeconds := defaultWebTimeoutSeconds
	if t, ok := args["timeout"].(float64); ok && int(t) > 0 {
		timeoutSeconds = clampInt(int(t), 1, maxWebTimeoutSeconds)
	}

	client := &http.Client{Timeout: time.Duration(timeoutSeconds) * time.Second}
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return Err("invalid url: " + err.Error())
	}
	req.Header.Set("User-Agent", defaultUA)
	if hdrs, ok := args["headers"].(map[string]any); ok {
		for k, v := range hdrs {
			if s, ok := v.(string); ok {
				req.Header.Set(k, s)
			}
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return Err("fetch failed: " + err.Error())
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxLen)+1))
	if err != nil {
		return Err("read body: " + err.Error())
	}
	html := string(body)

	var output string
	switch format {
	case "html":
		output = html
	case "text":
		output = stripHTML(html)
	case "markdown":
		output = toMarkdown(html)
	default:
		return Err("invalid format: " + format)
	}

	prefix := fmt.Sprintf("[status] %s\n", resp.Status)
	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		prefix += fmt.Sprintf("[content_type] %s\n", contentType)
	}
	return Ok(prefix + truncate(output, maxLen))
}

func stripHTML(s string) string {
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(s))
	if err != nil {
		// fallback: naive tag stripping
		var b strings.Builder
		inTag := false
		for _, r := range s {
			if r == '<' {
				inTag = true
			} else if r == '>' {
				inTag = false
			} else if !inTag {
				b.WriteRune(r)
			}
		}
		return strings.Join(strings.Fields(b.String()), " ")
	}
	doc.Find("script, style, noscript, nav, footer, header").Remove()
	// Insert newlines before block elements so text doesn't run together
	doc.Find("p, h1, h2, h3, h4, h5, h6, li, br, div, tr, blockquote, pre").Each(func(_ int, s *goquery.Selection) {
		s.PrependHtml("\n")
	})
	text := doc.Find("body").Text()
	lines := strings.Split(text, "\n")
	var out []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return strings.Join(out, "\n")
}

func toMarkdown(html string) string {
	md, err := htmltomarkdown.ConvertString(html)
	if err != nil {
		return stripHTML(html)
	}
	return strings.TrimSpace(md)
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + fmt.Sprintf("\n... (truncated, %d total chars)", len(s))
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// ---- WebSearchTool ----

type WebSearchTool struct{}

func (w *WebSearchTool) Name() string { return "web_search" }

func (w *WebSearchTool) Description() string {
	return "Search the web and return results from multiple engines"
}

func (w *WebSearchTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query":       map[string]any{"type": "string", "description": "Search query"},
			"backend":     map[string]any{"type": "string", "enum": []string{"auto", "duckduckgo", "brave", "google"}, "default": "auto"},
			"max_results": map[string]any{"type": "integer", "minimum": 1, "maximum": 20, "default": 5, "description": "Maximum results"},
		},
		"required": []string{"query"},
	}
}

type searchResult struct {
	Title string
	URL   string
	Body  string
}

func (w *WebSearchTool) Execute(args map[string]any) Result {
	query, _ := args["query"].(string)
	if query == "" {
		return Err("query is required")
	}
	backend := "auto"
	if b, ok := args["backend"].(string); ok && b != "" {
		backend = b
	}
	maxResults := 5
	if m, ok := args["max_results"].(float64); ok && int(m) > 0 {
		maxResults = clampInt(int(m), 1, 20)
	}

	engines := []func(string, int) ([]searchResult, error){searchGoogle, searchDuckDuckGo, searchBrave}
	if backend != "auto" {
		switch backend {
		case "google":
			engines = engines[:1]
		case "duckduckgo":
			engines = []func(string, int) ([]searchResult, error){searchDuckDuckGo}
		case "brave":
			engines = []func(string, int) ([]searchResult, error){searchBrave}
		default:
			return Err("invalid backend: " + backend)
		}
	}

	var results []searchResult
	var lastErr error
	for _, eng := range engines {
		results, lastErr = eng(query, maxResults)
		if lastErr == nil && len(results) > 0 {
			break
		}
	}
	if len(results) == 0 {
		msg := "all search engines failed"
		if lastErr != nil {
			msg += ": " + lastErr.Error()
		}
		return Err(msg)
	}

	if len(results) > maxResults {
		results = results[:maxResults]
	}

	var b strings.Builder
	for i, r := range results {
		fmt.Fprintf(&b, "%d. %s\n   %s\n", i+1, r.Title, r.URL)
		if r.Body != "" {
			fmt.Fprintf(&b, "   %s\n", r.Body)
		}
		b.WriteString("\n")
	}
	return Ok(strings.TrimSpace(b.String()))
}

func searchDuckDuckGo(query string, max int) ([]searchResult, error) {
	form := url.Values{"q": {query}, "b": {""}, "l": {"us-en"}, "s": {""}, "df": {""}}
	client := &http.Client{Timeout: 15 * time.Second}
	req, _ := http.NewRequest("POST", "https://html.duckduckgo.com/html/", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", defaultUA)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	var results []searchResult
	doc.Find("a.result__a").Each(func(i int, s *goquery.Selection) {
		if i >= max {
			return
		}
		title := strings.TrimSpace(s.Text())
		href, _ := s.Attr("href")
		href = decodeDDGURL(href)
		body := ""
		parent := s.Closest(".result")
		if parent.Length() > 0 {
			snippet := parent.Find("a.result__snippet, span.result__snippet").First()
			body = strings.TrimSpace(snippet.Text())
		}
		results = append(results, searchResult{Title: title, URL: href, Body: body})
	})
	return results, nil
}

func decodeDDGURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	if uddg := u.Query().Get("uddg"); uddg != "" {
		decoded, err := url.QueryUnescape(uddg)
		if err == nil {
			return decoded
		}
	}
	return raw
}

func searchGoogle(query string, max int) ([]searchResult, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	u := "https://www.google.com/search?q=" + url.QueryEscape(query) + "&num=10&hl=en"
	req, _ := http.NewRequest("GET", u, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36")
	req.Header.Set("Cookie", "CONSENT=YES+")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	var results []searchResult
	doc.Find("a[href]").Each(func(i int, s *goquery.Selection) {
		if len(results) >= max {
			return
		}
		href, _ := s.Attr("href")
		if !strings.Contains(href, "/url?q=") {
			return
		}
		parsed, err := url.Parse(href)
		if err != nil {
			return
		}
		realURL := parsed.Query().Get("q")
		if realURL == "" || !strings.HasPrefix(realURL, "http") {
			return
		}
		title := strings.TrimSpace(s.Text())
		if title == "" {
			return
		}
		results = append(results, searchResult{Title: title, URL: realURL})
	})
	return results, nil
}

func searchBrave(query string, max int) ([]searchResult, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	u := "https://search.brave.com/search?q=" + url.QueryEscape(query) + "&source=web"
	req, _ := http.NewRequest("GET", u, nil)
	req.Header.Set("User-Agent", defaultUA)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	var results []searchResult
	doc.Find("[data-type=\"web\"]").Each(func(i int, s *goquery.Selection) {
		if len(results) >= max {
			return
		}
		link := s.Find("a[href]").First()
		href, _ := link.Attr("href")
		if !strings.HasPrefix(href, "http") {
			return
		}
		title := s.Find(".title, h2, h3").First().Text()
		if title == "" {
			title = link.Text()
		}
		title = strings.TrimSpace(title)
		body := strings.TrimSpace(s.Find(".snippet").First().Text())
		results = append(results, searchResult{Title: title, URL: href, Body: body})
	})
	return results, nil
}
