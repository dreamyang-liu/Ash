package ddgs

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// fetchVQD is the vqd fetcher used by the DuckDuckGo JSON engines;
// a package variable so parity tests can stub the network call.
var fetchVQD = ddgVQD

// ddgVQD fetches the vqd token for a query, mirroring the _get_vqd helper
// shared by the DuckDuckGo JSON engines.
func ddgVQD(ctx context.Context, client *httpClient, query string) (string, error) {
	resp, err := client.get(ctx, "https://duckduckgo.com", map[string]string{"q": query})
	if err != nil {
		return "", err
	}
	return extractVQD(resp.Content, query)
}

// newDuckduckgoText mirrors ddgs.engines.duckduckgo.Duckduckgo (text, POST html endpoint).
func newDuckduckgoText(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "duckduckgo",
		category:     "text",
		provider:     "bing",
		searchURL:    "https://html.duckduckgo.com/html/",
		searchMethod: "POST",
		itemsXPath:   "//div[contains(@class, 'body')]",
		elementsXPath: map[string]string{
			"title": ".//h2//text()",
			"href":  "./a/@href",
			"body":  "./a//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		payload := map[string]string{"q": p.Query, "b": "", "l": p.Region}
		if p.Page > 1 {
			payload["s"] = strconv.Itoa(10 + (p.Page-2)*15)
		}
		if p.TimeLimit != "" {
			payload["df"] = p.TimeLimit
		}
		return payload, nil
	}
	e.postExtract = func(results []Result) []Result {
		out := make([]Result, 0, len(results))
		for _, r := range results {
			if !strings.HasPrefix(r.getString("href"), "https://duckduckgo.com/y.js?") {
				out = append(out, r)
			}
		}
		return out
	}
	return e, nil
}

// ddgJSONExtract parses a DuckDuckGo JSON API response ("results" array)
// into Results using srcKey->dstField mapping (elements_replace in Python).
func ddgJSONExtract(category string, text string, fieldMap map[string]string) ([]Result, error) {
	var payload struct {
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		return nil, fmt.Errorf("%w: parsing duckduckgo json: %v", ErrDDGS, err)
	}
	results := make([]Result, 0, len(payload.Results))
	for _, item := range payload.Results {
		result := newResult(category)
		for srcKey, dstField := range fieldMap {
			// item.get(key) semantics: missing keys store nil, like Python.
			result.Set(dstField, item[srcKey])
		}
		results = append(results, result)
	}
	return results, nil
}

// newDuckduckgoImages mirrors ddgs.engines.duckduckgo_images.DuckduckgoImages.
func newDuckduckgoImages(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "duckduckgo",
		category:     "images",
		provider:     "bing",
		searchURL:    "https://duckduckgo.com/i.js",
		searchMethod: "GET",
		headersUpdate: map[string]string{
			"Accept":          "*/*",
			"Accept-Language": "en-US,en;q=0.5",
			"Referer":         "https://duckduckgo.com/",
			"Sec-GPC":         "1",
			"Connection":      "keep-alive",
			"Sec-Fetch-Dest":  "empty",
			"Sec-Fetch-Mode":  "cors",
			"Sec-Fetch-Site":  "same-origin",
			"Priority":        "u=4",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(ctx context.Context, p searchParams) (map[string]string, error) {
		vqd, err := fetchVQD(ctx, e.http, p.Query)
		if err != nil {
			return nil, err
		}
		safesearch := map[string]string{"on": "1", "moderate": "1", "off": "-1"}
		timelimitNames := map[string]string{"d": "Day", "w": "Week", "m": "Month", "y": "Year"}

		filter := func(prefix, key string) string {
			if v := p.Extra[key]; v != "" {
				return prefix + ":" + v
			}
			return ""
		}
		timelimit := ""
		if p.TimeLimit != "" {
			timelimit = "time:" + timelimitNames[p.TimeLimit]
		}
		size := filter("size", "size")
		color := filter("color", "color")
		typeImage := filter("type", "type_image")
		layout := filter("layout", "layout")
		licenseImage := filter("license", "license_image")

		payload := map[string]string{
			"o":   "json",
			"q":   p.Query,
			"l":   p.Region,
			"vqd": vqd,
			"p":   safesearch[strings.ToLower(p.SafeSearch)],
			"ct":  "AT",
		}
		if timelimit != "" || size != "" || color != "" || typeImage != "" || layout != "" || licenseImage != "" {
			payload["f"] = strings.Join([]string{timelimit, size, color, typeImage, layout, licenseImage}, ",")
		}
		if p.Page > 1 {
			payload["s"] = strconv.Itoa((p.Page - 1) * 100)
		}
		return payload, nil
	}
	e.extractResults = func(_ context.Context, text string, _ searchParams) ([]Result, error) {
		return ddgJSONExtract("images", text, map[string]string{
			"title":     "title",
			"image":     "image",
			"thumbnail": "thumbnail",
			"url":       "url",
			"height":    "height",
			"width":     "width",
			"source":    "source",
		})
	}
	return e, nil
}

// newDuckduckgoNews mirrors ddgs.engines.duckduckgo_news.DuckduckgoNews.
func newDuckduckgoNews(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "duckduckgo",
		category:     "news",
		provider:     "bing",
		searchURL:    "https://duckduckgo.com/news.js",
		searchMethod: "GET",
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(ctx context.Context, p searchParams) (map[string]string, error) {
		vqd, err := fetchVQD(ctx, e.http, p.Query)
		if err != nil {
			return nil, err
		}
		safesearch := map[string]string{"on": "1", "moderate": "-1", "off": "-2"}
		payload := map[string]string{
			"l":     p.Region,
			"o":     "json",
			"noamp": "1",
			"q":     p.Query,
			"vqd":   vqd,
			"p":     safesearch[strings.ToLower(p.SafeSearch)],
		}
		if p.TimeLimit != "" {
			payload["df"] = p.TimeLimit
		}
		if p.Page > 1 {
			payload["s"] = strconv.Itoa((p.Page - 1) * 30)
		}
		return payload, nil
	}
	e.extractResults = func(_ context.Context, text string, _ searchParams) ([]Result, error) {
		return ddgJSONExtract("news", text, map[string]string{
			"date":    "date",
			"title":   "title",
			"excerpt": "body",
			"url":     "url",
			"image":   "image",
			"source":  "source",
		})
	}
	return e, nil
}

// newDuckduckgoVideos mirrors ddgs.engines.duckduckgo_videos.DuckduckgoVideos.
func newDuckduckgoVideos(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "duckduckgo",
		category:     "videos",
		provider:     "bing",
		searchURL:    "https://duckduckgo.com/v.js",
		searchMethod: "GET",
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(ctx context.Context, p searchParams) (map[string]string, error) {
		vqd, err := fetchVQD(ctx, e.http, p.Query)
		if err != nil {
			return nil, err
		}
		safesearch := map[string]string{"on": "1", "moderate": "-1", "off": "-2"}
		timelimit := ""
		if p.TimeLimit != "" {
			timelimit = "publishedAfter:" + p.TimeLimit
		}
		filter := func(prefix, key string) string {
			if v := p.Extra[key]; v != "" {
				return prefix + ":" + v
			}
			return ""
		}
		resolution := filter("videoDefinition", "resolution")
		duration := filter("videoDuration", "duration")
		licenseVideos := filter("videoLicense", "license_videos")

		payload := map[string]string{
			"l":   p.Region,
			"o":   "json",
			"q":   p.Query,
			"vqd": vqd,
			"f":   strings.Join([]string{timelimit, resolution, duration, licenseVideos}, ","),
			"p":   safesearch[strings.ToLower(p.SafeSearch)],
		}
		if p.Page > 1 {
			payload["s"] = strconv.Itoa((p.Page - 1) * 60)
		}
		return payload, nil
	}
	e.extractResults = func(_ context.Context, text string, _ searchParams) ([]Result, error) {
		fields := map[string]string{}
		for _, f := range resultFields["videos"] {
			fields[f] = f
		}
		return ddgJSONExtract("videos", text, fields)
	}
	return e, nil
}
