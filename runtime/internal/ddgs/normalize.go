package ddgs

import (
	"fmt"
	"html"
	"net/url"
	"regexp"
	"strings"
	"time"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

var regexStripTags = regexp.MustCompile("<.*?>")

// normalizeURL unquotes a URL and replaces spaces with '+'.
// Mirrors ddgs.utils._normalize_url.
func normalizeURL(rawURL string) string {
	if rawURL == "" {
		return ""
	}
	unquoted, err := url.QueryUnescape(rawURL)
	if err != nil {
		unquoted = rawURL
	}
	return strings.ReplaceAll(unquoted, " ", "+")
}

// normalizeText strips HTML tags, unescapes HTML entities, normalizes
// Unicode (NFC), removes "C" category characters and collapses whitespace.
// Mirrors ddgs.utils._normalize_text.
func normalizeText(raw string) string {
	if raw == "" {
		return ""
	}
	text := regexStripTags.ReplaceAllString(raw, "")
	text = html.UnescapeString(text)
	text = norm.NFC.String(text)
	// Remove all "C" category characters entirely (including \n, \t),
	// exactly like Python's translate({C-category: None}).
	text = strings.Map(func(r rune) rune {
		if unicode.In(r, unicode.C) {
			return -1
		}
		return r
	}, text)
	return strings.Join(strings.Fields(text), " ")
}

// normalizeDate converts a numeric UNIX timestamp to an ISO-8601 string.
// String input is returned unchanged. Mirrors ddgs.utils._normalize_date.
func normalizeDate(date any) string {
	switch v := date.(type) {
	case nil:
		return ""
	case string:
		return v
	case int:
		return time.Unix(int64(v), 0).UTC().Format("2006-01-02T15:04:05-07:00")
	case int64:
		return time.Unix(v, 0).UTC().Format("2006-01-02T15:04:05-07:00")
	case float64:
		return time.Unix(int64(v), 0).UTC().Format("2006-01-02T15:04:05-07:00")
	default:
		return fmt.Sprintf("%v", v)
	}
}

// extractVQD extracts the vqd token from a DuckDuckGo HTML response.
// Mirrors ddgs.utils._extract_vqd.
func extractVQD(body []byte, query string) (string, error) {
	text := string(body)
	type marker struct {
		start string
		end   byte
	}
	for _, m := range []marker{
		{`vqd="`, '"'},
		{`vqd=`, '&'},
		{`vqd='`, '\''},
	} {
		if idx := strings.Index(text, m.start); idx >= 0 {
			start := idx + len(m.start)
			if end := strings.IndexByte(text[start:], m.end); end >= 0 {
				return text[start : start+end], nil
			}
		}
	}
	return "", fmt.Errorf("%w: extractVQD() query=%q could not extract vqd", ErrDDGS, query)
}

// expandProxyTBAlias expands "tb" to the Tor Browser proxy URL.
// Mirrors ddgs.utils._expand_proxy_tb_alias.
func expandProxyTBAlias(proxy string) string {
	if proxy == "tb" {
		return "socks5://127.0.0.1:9150"
	}
	return proxy
}
