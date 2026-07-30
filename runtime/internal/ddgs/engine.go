package ddgs

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/antchfx/htmlquery"
	"golang.org/x/net/html"
)

// searchParams carries the per-call search parameters shared by every
// engine, mirroring the keyword arguments of BaseSearchEngine.search.
type searchParams struct {
	Query      string
	Region     string            // e.g. "us-en"
	SafeSearch string            // "on", "moderate", "off"
	TimeLimit  string            // "d", "w", "m", "y" or ""
	Page       int               // 1-based
	Extra      map[string]string // engine-specific kwargs (size, color, resolution, ...)
}

// searchEngine is the interface every backend implements, mirroring
// ddgs.base.BaseSearchEngine.
type searchEngine interface {
	Name() string
	Category() string
	Provider() string
	Priority() float64
	Search(ctx context.Context, p searchParams) ([]Result, error)
}

// engineSpec holds the declarative parts of an engine, mirroring the
// ClassVar attributes of BaseSearchEngine subclasses.
type engineSpec struct {
	name          string
	category      string
	provider      string
	priority      float64 // default 1
	searchURL     string
	searchMethod  string // "GET" or "POST"
	headersUpdate map[string]string
	itemsXPath    string
	elementsXPath map[string]string // result field -> xpath (relative to item)
}

// baseEngine provides shared behavior for all engines: HTTP access,
// payload building hooks, HTML parsing and XPath field extraction.
type baseEngine struct {
	spec engineSpec
	http *httpClient

	// buildPayload builds request params for a search. Required.
	buildPayload func(ctx context.Context, p searchParams) (map[string]string, error)
	// searchURLFn optionally overrides the search URL per request.
	searchURLFn func(p searchParams) string
	// preProcessHTML optionally rewrites the raw response text before parsing.
	preProcessHTML func(string) string
	// extractResults optionally replaces the default XPath extraction
	// (e.g. JSON APIs).
	extractResults func(ctx context.Context, text string, p searchParams) ([]Result, error)
	// postExtract optionally post-processes extracted results.
	postExtract func([]Result) []Result
}

func (e *baseEngine) Name() string     { return e.spec.name }
func (e *baseEngine) Category() string { return e.spec.category }
func (e *baseEngine) Provider() string { return e.spec.provider }
func (e *baseEngine) Priority() float64 {
	if e.spec.priority == 0 {
		return 1
	}
	return e.spec.priority
}

// newBaseEngine constructs a baseEngine with its own HTTP client, applying
// the engine's default header overrides.
func newBaseEngine(spec engineSpec, proxy string, timeout time.Duration, verify Verify) (*baseEngine, error) {
	client, err := newHTTPClient(proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	client.headersUpdate(spec.headersUpdate)
	return &baseEngine{spec: spec, http: client}, nil
}

// requestText performs the request and returns the body text for HTTP 200,
// mirroring BaseSearchEngine.request.
func (e *baseEngine) requestText(ctx context.Context, method, rawURL string, params, data map[string]string) (string, error) {
	resp, err := e.http.request(ctx, method, rawURL, params, data)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != 200 {
		return "", nil
	}
	return resp.Text(), nil
}

// extractByXPath implements the default BaseSearchEngine.extract_results:
// select item nodes with itemsXPath, then for each result field join the
// text of the nodes matched by its relative XPath.
func (e *baseEngine) extractByXPath(text string) ([]Result, error) {
	doc, err := htmlquery.Parse(strings.NewReader(text))
	if err != nil {
		return nil, fmt.Errorf("%w: parsing html: %v", ErrDDGS, err)
	}
	items, err := htmlquery.QueryAll(doc, e.spec.itemsXPath)
	if err != nil {
		return nil, fmt.Errorf("%w: items xpath: %v", ErrDDGS, err)
	}
	results := make([]Result, 0, len(items))
	for _, item := range items {
		result := newResult(e.spec.category)
		for field, xpath := range e.spec.elementsXPath {
			nodes, err := htmlquery.QueryAll(item, xpath)
			if err != nil {
				return nil, fmt.Errorf("%w: xpath %q: %v", ErrDDGS, xpath, err)
			}
			var sb strings.Builder
			for _, n := range nodes {
				sb.WriteString(nodeText(n))
			}
			result.Set(field, strings.Join(strings.Fields(sb.String()), " "))
		}
		results = append(results, result)
	}
	return results, nil
}

// nodeText returns the text content of a node; attribute and text nodes
// return their value directly (like lxml xpath returning strings).
func nodeText(n *html.Node) string {
	if n == nil {
		return ""
	}
	if n.Type == html.TextNode {
		return n.Data
	}
	return htmlquery.InnerText(n)
}

// Search runs a full search request cycle, mirroring BaseSearchEngine.search.
func (e *baseEngine) Search(ctx context.Context, p searchParams) ([]Result, error) {
	if e.buildPayload == nil {
		return nil, fmt.Errorf("%w: engine %s has no payload builder", ErrDDGS, e.spec.name)
	}
	payload, err := e.buildPayload(ctx, p)
	if err != nil {
		return nil, err
	}

	searchURL := e.spec.searchURL
	if e.searchURLFn != nil {
		searchURL = e.searchURLFn(p)
	}

	var text string
	if e.spec.searchMethod == "POST" {
		text, err = e.requestText(ctx, "POST", searchURL, nil, payload)
	} else {
		text, err = e.requestText(ctx, "GET", searchURL, payload, nil)
	}
	if err != nil {
		return nil, err
	}
	if text == "" {
		return nil, nil
	}

	if e.preProcessHTML != nil {
		text = e.preProcessHTML(text)
	}

	var results []Result
	if e.extractResults != nil {
		results, err = e.extractResults(ctx, text, p)
	} else {
		results, err = e.extractByXPath(text)
	}
	if err != nil {
		return nil, err
	}

	if e.postExtract != nil {
		results = e.postExtract(results)
	}
	return results, nil
}

// splitRegion splits "us-en" into country "us" and lang "en".
func splitRegion(region string) (country, lang string) {
	parts := strings.SplitN(strings.ToLower(region), "-", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return parts[0], "en"
}
