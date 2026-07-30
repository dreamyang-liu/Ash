package ddgs

// Extract mirrors DDGS.extract() (ddgs/ddgs.py L245-272): fetch a URL and
// return its content in a chosen format. It is a public method on the DDGS
// struct defined in ddgs.go. Python delegates HTML→markdown conversion to
// primp (Rust html2text); here text_markdown uses the battle-tested
// html-to-markdown library (links, emphasis and tables preserved) and
// text_plain uses a lightweight DOM walker.

import (
	"context"
	"fmt"
	"strings"

	"github.com/JohannesKaufmann/html-to-markdown/v2/converter"
	"github.com/JohannesKaufmann/html-to-markdown/v2/plugin/base"
	"github.com/JohannesKaufmann/html-to-markdown/v2/plugin/commonmark"
	"github.com/JohannesKaufmann/html-to-markdown/v2/plugin/table"
	"golang.org/x/net/html"
)

// ExtractFormat selects the output format of Extract, mirroring the Python
// `fmt` parameter values.
type ExtractFormat string

const (
	// FormatTextMarkdown converts the page to Markdown-ish plain text (default).
	FormatTextMarkdown ExtractFormat = "text_markdown"
	// FormatTextPlain converts the page to plain text.
	FormatTextPlain ExtractFormat = "text_plain"
	// FormatText returns the raw HTML text.
	FormatText ExtractFormat = "text"
	// FormatContent returns the raw response bytes.
	FormatContent ExtractFormat = "content"
)

// Extracted is the return value of Extract, mirroring the Python
// {"url": ..., "content": ...} dict.
type Extracted struct {
	URL     string
	Content string // textual formats
	Bytes   []byte // FormatContent
}

// Extract fetches a URL and extracts its content. Mirrors DDGS.extract().
func (d *DDGS) Extract(ctx context.Context, rawURL string, format ExtractFormat) (*Extracted, error) {
	client, err := newHTTPClient(d.proxy, d.timeout, d.verify)
	if err != nil {
		return nil, err
	}
	resp, err := client.get(ctx, rawURL, nil)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%w: failed to fetch %s: HTTP %d", ErrDDGS, rawURL, resp.StatusCode)
	}

	switch format {
	case FormatContent:
		return &Extracted{URL: rawURL, Bytes: resp.Content}, nil
	case FormatText:
		return &Extracted{URL: rawURL, Content: resp.Text()}, nil
	case FormatTextPlain:
		return &Extracted{URL: rawURL, Content: htmlToText(resp.Text())}, nil
	default: // FormatTextMarkdown and unknown values, like Python's else-branch
		return &Extracted{URL: rawURL, Content: htmlToMarkdown(resp.Text())}, nil
	}
}

// markdownConverter is the shared HTML→Markdown converter (goroutine-safe),
// configured with commonmark + table support.
var markdownConverter = converter.NewConverter(
	converter.WithPlugins(
		base.NewBasePlugin(),
		commonmark.NewCommonmarkPlugin(),
		table.NewTablePlugin(),
	),
)

// htmlToMarkdown converts HTML to Markdown, preserving links, emphasis,
// lists and tables. Falls back to plain text on conversion failure.
func htmlToMarkdown(rawHTML string) string {
	md, err := markdownConverter.ConvertString(rawHTML)
	if err != nil {
		return htmlToText(rawHTML)
	}
	return md
}

// skipTags are elements whose content is never text.
var skipTags = map[string]bool{
	"script": true, "style": true, "noscript": true,
	"head": true, "iframe": true, "svg": true,
}

// blockTags are elements that force a paragraph break.
var blockTags = map[string]bool{
	"p": true, "div": true, "section": true, "article": true, "header": true,
	"footer": true, "ul": true, "ol": true, "table": true, "tr": true,
	"blockquote": true, "pre": true, "br": true, "hr": true,
	"h1": true, "h2": true, "h3": true, "h4": true, "h5": true, "h6": true,
	"li": true,
}

// htmlToText converts HTML to plain text (Python's text_plain equivalent).
func htmlToText(rawHTML string) string {
	doc, err := html.Parse(strings.NewReader(rawHTML))
	if err != nil {
		return normalizeText(rawHTML)
	}
	var sb strings.Builder
	walkHTMLText(doc, &sb)
	// Collapse runs of blank lines.
	lines := strings.Split(sb.String(), "\n")
	var out []string
	blank := false
	for _, line := range lines {
		line = strings.Join(strings.Fields(line), " ")
		if line == "" {
			if !blank && len(out) > 0 {
				out = append(out, "")
			}
			blank = true
			continue
		}
		blank = false
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

func walkHTMLText(n *html.Node, sb *strings.Builder) {
	switch n.Type {
	case html.TextNode:
		sb.WriteString(n.Data)
		return
	case html.ElementNode:
		if skipTags[n.Data] {
			return
		}
		if blockTags[n.Data] {
			sb.WriteString("\n")
		}
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walkHTMLText(c, sb)
	}
	if n.Type == html.ElementNode && blockTags[n.Data] {
		sb.WriteString("\n")
	}
}
