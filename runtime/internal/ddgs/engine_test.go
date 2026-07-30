package ddgs

// Unit tests for engine.go XPath extraction, engine_misc.go date helpers,
// engine_text.go URL helpers, and ddgs.go orchestration (via httptest
// servers). Run by `go test ./...`; part of the deedy5/ddgs Go port.

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const ddgHTMLFixture = `<html><body>
<div class="links_main">
  <div class="body">
    <a href="https://example.com/1">Example One body snippet golang</a>
    <h2>Result One golang</h2>
  </div>
  <div class="body">
    <a href="https://duckduckgo.com/y.js?ad=1">Ad result</a>
    <h2>Advert</h2>
  </div>
  <div class="body">
    <a href="https://example.com/2">Example Two body</a>
    <h2>Result Two</h2>
  </div>
</div>
</body></html>`

func TestExtractByXPathDuckduckgo(t *testing.T) {
	e := &baseEngine{spec: engineSpec{
		category:   "text",
		itemsXPath: "//div[contains(@class, 'body')]",
		elementsXPath: map[string]string{
			"title": ".//h2//text()",
			"href":  "./a/@href",
			"body":  "./a//text()",
		},
	}}
	results, err := e.extractByXPath(ddgHTMLFixture)
	if err != nil {
		t.Fatalf("extractByXPath: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	if got := results[0].getString("title"); got != "Result One golang" {
		t.Errorf("title = %q", got)
	}
	if got := results[0].getString("href"); got != "https://example.com/1" {
		t.Errorf("href = %q", got)
	}
}

func TestDDGJSONExtract(t *testing.T) {
	payload := `{"results": [
	  {"title": "T1", "image": "https://i.com/1.jpg", "thumbnail": "https://t.com/1.jpg",
	   "url": "https://p.com/1", "height": 100, "width": 200, "source": "Bing"}
	]}`
	results, err := ddgJSONExtract("images", payload, map[string]string{
		"title": "title", "image": "image", "thumbnail": "thumbnail",
		"url": "url", "height": "height", "width": "width", "source": "source",
	})
	if err != nil {
		t.Fatalf("ddgJSONExtract: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if got := results[0].getString("image"); got != "https://i.com/1.jpg" {
		t.Errorf("image = %q", got)
	}
	if got, ok := results[0]["height"].(float64); !ok || got != 100 {
		t.Errorf("height = %v", results[0]["height"])
	}
}

func TestYahooExtractURL(t *testing.T) {
	in := "https://r.search.yahoo.com/_ylt=x/RU=https%3a%2f%2fexample.com%2fpage/RK=2/RS=abc"
	if got := yahooExtractURL(in); got != "https://example.com/page" {
		t.Errorf("yahooExtractURL = %q", got)
	}
}

func TestYahooNewsExtractDate(t *testing.T) {
	got := yahooNewsExtractDate("2 hours ago")
	then, err := time.Parse("2006-01-02T15:04:05-07:00", got)
	if err != nil {
		t.Fatalf("unparseable date %q: %v", got, err)
	}
	diff := time.Since(then)
	if diff < 119*time.Minute || diff > 121*time.Minute {
		t.Errorf("date %q not ~2h ago (diff %v)", got, diff)
	}
	if got := yahooNewsExtractDate("no date here"); got != "no date here" {
		t.Errorf("passthrough = %q", got)
	}
}

func TestBingNewsExtractDate(t *testing.T) {
	if got := bingNewsExtractDate("15.06.2024"); !strings.HasPrefix(got, "2024-06-15") {
		t.Errorf("dotted date = %q", got)
	}
	got := bingNewsExtractDate("3 days")
	then, err := time.Parse("2006-01-02T15:04:05-07:00", got)
	if err != nil {
		t.Fatalf("unparseable date %q: %v", got, err)
	}
	if diff := time.Since(then); diff < 71*time.Hour || diff > 73*time.Hour {
		t.Errorf("relative date %q not ~3d ago", got)
	}
}

// newTestEngine builds a stub engine that serves the given HTML from a
// httptest server through the standard XPath pipeline.
func newTestEngine(t *testing.T, name, provider, htmlBody string) searchEngine {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, htmlBody)
	}))
	t.Cleanup(server.Close)

	e, err := newBaseEngine(engineSpec{
		name:         name,
		category:     "text",
		provider:     provider,
		searchURL:    server.URL,
		searchMethod: "GET",
		itemsXPath:   "//div[contains(@class, 'body')]",
		elementsXPath: map[string]string{
			"title": ".//h2//text()",
			"href":  "./a/@href",
			"body":  "./a//text()",
		},
	}, "", 5*time.Second, Verify{})
	if err != nil {
		t.Fatalf("newBaseEngine: %v", err)
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		return map[string]string{"q": p.Query}, nil
	}
	return e
}

func TestEngineSearchEndToEnd(t *testing.T) {
	e := newTestEngine(t, "stub", "stubprov", ddgHTMLFixture)
	results, err := e.Search(context.Background(), searchParams{Query: "golang", Page: 1})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
}

func TestDDGSSearchAggregatesAndRanks(t *testing.T) {
	e := newTestEngine(t, "stub", "stubprov", ddgHTMLFixture)
	d, err := New(nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Inject the stub engine directly into the registry for this test.
	saved := engineRegistry["text"]
	engineRegistry["text"] = map[string]engineFactory{
		"stub": func(string, time.Duration, Verify) (searchEngine, error) { return e, nil },
	}
	defer func() { engineRegistry["text"] = saved }()

	results, err := d.Text(context.Background(), "golang", &SearchOptions{Backend: "stub"})
	if err != nil {
		t.Fatalf("Text: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	// Ranker puts the golang-matching result first.
	if got := results[0].getString("href"); got != "https://example.com/1" {
		t.Errorf("first ranked = %q, want example.com/1", got)
	}
}

func TestDDGSEmptyQuery(t *testing.T) {
	d, _ := New(nil)
	if _, err := d.Text(context.Background(), "", nil); err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestDDGSUnknownBackendFallsBackToAuto(t *testing.T) {
	e := newTestEngine(t, "stub", "stubprov", ddgHTMLFixture)
	d, _ := New(nil)
	saved := engineRegistry["text"]
	engineRegistry["text"] = map[string]engineFactory{
		"stub": func(string, time.Duration, Verify) (searchEngine, error) { return e, nil },
	}
	defer func() { engineRegistry["text"] = saved }()

	// Nonexistent backend -> falls back to auto -> uses stub.
	results, err := d.Text(context.Background(), "golang", &SearchOptions{Backend: "nonexistent"})
	if err != nil {
		t.Fatalf("Text with bad backend: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected fallback-to-auto results")
	}
}

func TestDDGSMaxResults(t *testing.T) {
	e := newTestEngine(t, "stub", "stubprov", ddgHTMLFixture)
	d, _ := New(nil)
	saved := engineRegistry["text"]
	engineRegistry["text"] = map[string]engineFactory{
		"stub": func(string, time.Duration, Verify) (searchEngine, error) { return e, nil },
	}
	defer func() { engineRegistry["text"] = saved }()

	results, err := d.Text(context.Background(), "golang", &SearchOptions{Backend: "stub", MaxResults: 2})
	if err != nil {
		t.Fatalf("Text: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("got %d results, want 2", len(results))
	}
}

func TestHTMLToText(t *testing.T) {
	page := `<html><head><script>bad()</script></head><body>
	<h1>Title</h1><p>Para one.</p><ul><li>Item A</li><li>Item B</li></ul></body></html>`
	plain := htmlToText(page)
	if !strings.Contains(plain, "Title") || !strings.Contains(plain, "Item A") {
		t.Errorf("plain text missing content: %q", plain)
	}
	if strings.Contains(plain, "bad()") {
		t.Errorf("script content leaked: %q", plain)
	}
	if strings.Contains(plain, "#") {
		t.Errorf("plain text contains markdown: %q", plain)
	}
}

func TestHTMLToMarkdown(t *testing.T) {
	page := `<html><head><script>bad()</script></head><body>
	<h1>Title</h1>
	<p>Para with <a href="https://x.com">a link</a> and <b>bold</b>.</p>
	<ul><li>Item A</li><li>Item B</li></ul>
	<table><tr><th>Col1</th><th>Col2</th></tr><tr><td>a</td><td>b</td></tr></table>
	</body></html>`
	md := htmlToMarkdown(page)
	for _, want := range []string{
		"# Title",
		"[a link](https://x.com)", // link URL preserved
		"**bold**",
		"Item A",
		"| Col1 | Col2 |", // table structure preserved
	} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown missing %q:\n%s", want, md)
		}
	}
	if strings.Contains(md, "bad()") {
		t.Errorf("script content leaked: %q", md)
	}
}
