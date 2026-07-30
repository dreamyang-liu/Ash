package ddgs

// Misc engines: bing images, bing news, yahoo news, annas archive (books).
// Mirrors ddgs/engines/bing_images.py, bing_news.py, yahoo_news.py,
// annasarchive.py. Called from registry.go (engine factory table).

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/antchfx/htmlquery"
)

// newBingImages mirrors ddgs.engines.bing_images.BingImages.
func newBingImages(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "bing",
		category:     "images",
		provider:     "bing",
		searchURL:    "https://www.bing.com/images/async",
		searchMethod: "GET",
		itemsXPath:   "//div[./div[@class='imgpt']/a[@m] and ./div[@class='infopt']]",
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		count := 35
		if v := p.Extra["max_results"]; v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > count {
				count = n
			}
		}
		payload := map[string]string{
			"q":     p.Query,
			"async": "1",
			"first": strconv.Itoa((p.Page-1)*count + 1),
			"count": strconv.Itoa(count),
		}
		if p.TimeLimit != "" {
			minutes := map[string]int{"day": 1440, "week": 10080, "month": 44640, "year": 525600}[p.TimeLimit]
			payload["qft"] = fmt.Sprintf("filterui:age-lt%d", minutes)
		}
		return payload, nil
	}
	e.extractResults = func(_ context.Context, text string, _ searchParams) ([]Result, error) {
		doc, err := htmlquery.Parse(strings.NewReader(text))
		if err != nil {
			return nil, fmt.Errorf("%w: parsing bing images html: %v", ErrDDGS, err)
		}
		items, err := htmlquery.QueryAll(doc, e.spec.itemsXPath)
		if err != nil {
			return nil, fmt.Errorf("%w: bing images xpath: %v", ErrDDGS, err)
		}
		var results []Result
		for _, item := range items {
			metaNode, err := htmlquery.Query(item, ".//a[@class='iusc']/@m")
			if err != nil || metaNode == nil {
				continue
			}
			var m map[string]any
			if json.Unmarshal([]byte(nodeText(metaNode)), &m) != nil {
				continue
			}
			result := newResult("images")
			result.Set("title", m["t"])
			result.Set("image", m["murl"])
			result.Set("thumbnail", m["turl"])
			result.Set("url", m["purl"])
			if dim, err := htmlquery.Query(item, ".//div[contains(@class, 'img_info')][./span]/span[@class='nowrap']/text()"); err == nil && dim != nil {
				dimText := strings.ReplaceAll(nodeText(dim), "×", "x")
				if parts := strings.SplitN(dimText, "x", 2); len(parts) == 2 {
					result.Set("width", strings.TrimSpace(parts[0]))
					if fields := strings.Fields(parts[1]); len(fields) > 0 {
						result.Set("height", fields[0])
					}
				}
			}
			if src, err := htmlquery.Query(item, ".//div[@class='lnkw']//a/text()"); err == nil && src != nil {
				result.Set("source", nodeText(src))
			}
			results = append(results, result)
		}
		return results, nil
	}
	return e, nil
}

var bingNewsDateRE = regexp.MustCompile(`(?i)\b(\d+)\s*(days|tagen|jours|giorni|dias|días|дн\.|день)?\b`)

// bingNewsExtractDate mirrors ddgs.engines.bing_news.extract_date.
func bingNewsExtractDate(pubDateStr string) string {
	for _, layout := range []string{"02.01.2006", "01/02/2006", "02/01/2006"} {
		if t, err := time.Parse(layout, pubDateStr); err == nil {
			return t.UTC().Format("2006-01-02T15:04:05-07:00")
		}
	}
	if m := bingNewsDateRE.FindStringSubmatch(pubDateStr); m != nil {
		if daysAgo, err := strconv.Atoi(m[1]); err == nil {
			return time.Now().UTC().AddDate(0, 0, -daysAgo).Truncate(time.Second).Format("2006-01-02T15:04:05-07:00")
		}
	}
	return pubDateStr
}

// newBingNews mirrors ddgs.engines.bing_news.BingNews.
func newBingNews(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "bing",
		category:     "news",
		provider:     "bing",
		searchURL:    "https://www.bing.com/news/infinitescrollajax",
		searchMethod: "GET",
		itemsXPath:   "//div[contains(@class, 'newsitem')]",
		elementsXPath: map[string]string{
			"date":   ".//span[@aria-label]//@aria-label",
			"title":  "@data-title",
			"body":   ".//div[@class='snippet']//text()",
			"url":    "@url",
			"image":  ".//a[contains(@class, 'image')]//@src",
			"source": "@data-author",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		country, lang := splitRegion(p.Region)
		payload := map[string]string{
			"q":              p.Query,
			"InfiniteScroll": "1",
			"first":          strconv.Itoa(p.Page*10 + 1),
			"SFX":            strconv.Itoa(p.Page),
			"cc":             country,
			"setlang":        lang,
		}
		if p.TimeLimit != "" {
			payload["qft"] = map[string]string{
				"d": `interval="4"`,
				"w": `interval="7"`,
				"m": `interval="9"`,
				"y": `interval="9"`,
			}[p.TimeLimit]
		}
		return payload, nil
	}
	e.postExtract = func(results []Result) []Result {
		for _, r := range results {
			r["date"] = bingNewsExtractDate(r.getString("date"))
			if img := r.getString("image"); img != "" {
				r.Set("image", "https://www.bing.com"+strings.SplitN(img, "&", 2)[0])
			}
		}
		return results
	}
	return e, nil
}

var yahooNewsDateRE = regexp.MustCompile(`(?i)\b(\d+)\s*(year|month|week|day|hour|minute)s?\b`)

// yahooNewsExtractDate mirrors ddgs.engines.yahoo_news.extract_date.
func yahooNewsExtractDate(pubDateStr string) string {
	m := yahooNewsDateRE.FindStringSubmatch(pubDateStr)
	if m == nil {
		return pubDateStr
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return pubDateStr
	}
	var delta time.Duration
	switch strings.ToLower(m[2]) {
	case "minute":
		delta = time.Duration(n) * time.Minute
	case "hour":
		delta = time.Duration(n) * time.Hour
	case "day":
		delta = time.Duration(n) * 24 * time.Hour
	case "week":
		delta = time.Duration(n) * 7 * 24 * time.Hour
	case "month":
		delta = time.Duration(n) * 30 * 24 * time.Hour
	case "year":
		delta = time.Duration(n) * 365 * 24 * time.Hour
	}
	return time.Now().UTC().Add(-delta).Truncate(time.Second).Format("2006-01-02T15:04:05-07:00")
}

// yahooNewsExtractURL mirrors ddgs.engines.yahoo_news.extract_url.
func yahooNewsExtractURL(u string) string {
	parts := strings.SplitN(u, "/RU=", 2)
	if len(parts) < 2 {
		return u
	}
	t := strings.SplitN(parts[1], "/RK=", 2)[0]
	t = strings.SplitN(t, "?", 2)[0]
	return normalizeURL(strings.ReplaceAll(t, "+", " "))
}

// newYahooNews mirrors ddgs.engines.yahoo_news.YahooNews.
func newYahooNews(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "yahoo",
		category:     "news",
		provider:     "yahoo",
		searchURL:    "https://news.search.yahoo.com/search",
		searchMethod: "GET",
		itemsXPath:   "//div[@id='web']//li[a]",
		elementsXPath: map[string]string{
			"date":   ".//span[contains(@class, 'time')]//text()",
			"title":  ".//h4//text()",
			"body":   ".//p//text()",
			"url":    ".//h4/a/@href",
			"image":  "(.//img/@data-src | .//img/@src)[1]",
			"source": ".//span[contains(@class, 'source')]//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		payload := map[string]string{"p": p.Query}
		if p.Page > 1 {
			payload["b"] = strconv.Itoa((p.Page-1)*10 + 1)
		}
		if p.TimeLimit != "" {
			payload["btf"] = p.TimeLimit
		}
		return payload, nil
	}
	e.postExtract = func(results []Result) []Result {
		for _, r := range results {
			r["date"] = yahooNewsExtractDate(r.getString("date"))
			if u := r.getString("url"); strings.Contains(u, "/RU=") {
				r.Set("url", yahooNewsExtractURL(u))
			}
			if img := r.getString("image"); img != "" {
				if idx := strings.Index(img, "-/"); idx != -1 {
					r.Set("image", img[idx+2:])
				}
			}
			if src := r.getString("source"); src != "" {
				r.Set("source", strings.SplitN(src, " · via Yahoo", 2)[0])
			}
		}
		return results
	}
	return e, nil
}

// annasArchiveTLDs mirrors random.choice(['gd', 'gl', 'pk']).
var annasArchiveTLDs = []string{"gd", "gl", "pk"}

// newAnnasArchive mirrors ddgs.engines.annasarchive.AnnasArchive.
func newAnnasArchive(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	tld := annasArchiveTLDs[randInt(0, int64(len(annasArchiveTLDs)-1))]
	baseURL := "https://annas-archive." + tld
	e, err := newBaseEngine(engineSpec{
		name:         "annasarchive",
		category:     "books",
		provider:     "annasarchive",
		searchURL:    baseURL + "/search",
		searchMethod: "GET",
		itemsXPath:   "//div[contains(@class, 'record-list-outer')]/div",
		elementsXPath: map[string]string{
			"title":     ".//a[contains(@class, 'text-lg')]//text()",
			"author":    ".//a[span[contains(@class, 'user')]]//text()",
			"publisher": ".//a[span[contains(@class, 'company')]]//text()",
			"info":      ".//div[contains(@class, 'text-gray-800')]/text()",
			"url":       "./a/@href",
			"thumbnail": ".//img/@src",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		return map[string]string{"q": p.Query, "page": strconv.Itoa(p.Page)}, nil
	}
	e.preProcessHTML = func(text string) string {
		// Anna's Archive hides results inside HTML comments.
		return strings.ReplaceAll(strings.ReplaceAll(text, "<!--", ""), "-->", "")
	}
	e.postExtract = func(results []Result) []Result {
		for _, r := range results {
			r.Set("url", baseURL+r.getString("url"))
		}
		return results
	}
	return e, nil
}
