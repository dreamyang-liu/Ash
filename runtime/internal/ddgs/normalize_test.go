package ddgs

// Unit tests exercising normalize.go, ranker.go and results.go (the pure
// functions ported from ddgs/utils.py, similarity.py, results.py).
// Run by `go test ./...`; user instruction: 复刻 deedy5/ddgs 为 Go library.

import (
	"strings"
	"testing"
)

func TestNormalizeText(t *testing.T) {
	tests := []struct {
		name, in, want string
	}{
		{"strips html tags", "<b>hello</b> world", "hello world"},
		{"unescapes entities", "a &amp; b &lt;c&gt;", "a & b <c>"},
		{"collapses whitespace", "  a  b  c  ", "a b c"},
		{"removes zero width space", "a​b", "ab"},
		{"empty input", "", ""},
		// C-category chars (\n, \t) are deleted, not converted to spaces —
		// matches Python's translate({C: None}).
		{"newline deleted like python", "line1\nline2", "line1line2"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeText(tt.in); got != tt.want {
				t.Errorf("normalizeText(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestNormalizeURL(t *testing.T) {
	tests := []struct {
		name, in, want string
	}{
		{"unquotes percent encoding", "https://x.com/a%20b", "https://x.com/a+b"},
		{"replaces spaces with plus", "https://x.com/a b", "https://x.com/a+b"},
		{"empty input", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeURL(tt.in); got != tt.want {
				t.Errorf("normalizeURL(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestNormalizeDate(t *testing.T) {
	if got := normalizeDate(0); !strings.HasPrefix(got, "1970-01-01T00:00:00") {
		t.Errorf("normalizeDate(0) = %q, want 1970-01-01 prefix", got)
	}
	if got := normalizeDate("2024-01-01"); got != "2024-01-01" {
		t.Errorf("normalizeDate(string) = %q, want passthrough", got)
	}
	if got := normalizeDate(float64(86400)); !strings.HasPrefix(got, "1970-01-02T00:00:00") {
		t.Errorf("normalizeDate(86400.0) = %q, want 1970-01-02 prefix", got)
	}
}

func TestExtractVQD(t *testing.T) {
	tests := []struct {
		name, html, want string
		wantErr          bool
	}{
		{"double quotes", `x vqd="4-12345" y`, "4-12345", false},
		{"query param", `href="?q=x&vqd=4-999&other=1"`, "4-999", false},
		{"single quotes", `vqd='4-abc'`, "4-abc", false},
		{"missing", `no token here`, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractVQD([]byte(tt.html), "q")
			if (err != nil) != tt.wantErr {
				t.Fatalf("extractVQD error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("extractVQD = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestExpandProxyTBAlias(t *testing.T) {
	if got := expandProxyTBAlias("tb"); got != "socks5://127.0.0.1:9150" {
		t.Errorf("tb alias = %q", got)
	}
	if got := expandProxyTBAlias("http://x:1"); got != "http://x:1" {
		t.Errorf("passthrough = %q", got)
	}
}

func TestSimpleFilterRankerOrdering(t *testing.T) {
	docs := []Result{
		{"title": "unrelated", "href": "https://a.com", "body": "nothing"},
		{"title": "", "href": "https://b.com", "body": "golang tutorial"},
		{"title": "golang", "href": "https://c.com", "body": "nothing"},
		{"title": "golang", "href": "https://d.com", "body": "golang docs"},
		{"title": "Go", "href": "https://en.wikipedia.org/wiki/Go", "body": "x"},
	}
	ranked := simpleFilterRanker{}.rank(docs, "golang")
	wantOrder := []string{
		"https://en.wikipedia.org/wiki/Go", // wikipedia first
		"https://d.com",                    // both
		"https://c.com",                    // title only
		"https://b.com",                    // body only
		"https://a.com",                    // neither
	}
	if len(ranked) != len(wantOrder) {
		t.Fatalf("got %d results, want %d", len(ranked), len(wantOrder))
	}
	for i, want := range wantOrder {
		if got := ranked[i].getString("href"); got != want {
			t.Errorf("position %d = %q, want %q", i, got, want)
		}
	}
}

func TestSimpleFilterRankerSkipsWikimediaCategories(t *testing.T) {
	docs := []Result{
		{"title": "Category: Foo - Wikimedia Commons", "href": "https://x.com", "body": ""},
	}
	if ranked := (simpleFilterRanker{}).rank(docs, "foo"); len(ranked) != 0 {
		t.Errorf("expected wikimedia category page to be skipped, got %d", len(ranked))
	}
}

func TestResultsAggregatorDedupAndFrequency(t *testing.T) {
	agg := newResultsAggregator("text")
	r1 := Result{"title": "a", "href": "https://a.com", "body": "short"}
	r2 := Result{"title": "a", "href": "https://a.com", "body": "a longer body text"}
	r3 := Result{"title": "b", "href": "https://b.com", "body": "b"}
	agg.extend([]Result{r3, r1, r2})

	if agg.len() != 2 {
		t.Fatalf("len = %d, want 2", agg.len())
	}
	out := agg.extract()
	// a.com appeared twice -> first; longer body wins.
	if got := out[0].getString("href"); got != "https://a.com" {
		t.Errorf("first result = %q, want a.com (higher frequency)", got)
	}
	if got := out[0].getString("body"); got != "a longer body text" {
		t.Errorf("body = %q, want longer body kept", got)
	}
}

func TestResultSetNormalizes(t *testing.T) {
	r := newResult("text")
	r.Set("title", "<b>Hello</b>&nbsp;World")
	if got := r.getString("title"); got != "Hello World" {
		t.Errorf("title = %q, want normalized", got)
	}
	r.Set("href", "https://x.com/a%20b")
	if got := r.getString("href"); got != "https://x.com/a+b" {
		t.Errorf("href = %q, want unquoted", got)
	}
}

func TestNewResultDefaults(t *testing.T) {
	r := newResult("videos")
	if len(r) != len(resultFields["videos"]) {
		t.Errorf("videos result has %d fields, want %d", len(r), len(resultFields["videos"]))
	}
	if _, ok := r["images"].(map[string]any); !ok {
		t.Errorf("videos images field should be a map")
	}
}
