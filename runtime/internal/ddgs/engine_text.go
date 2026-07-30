package ddgs

// Text engines: google, brave, mojeek, yahoo, yandex, startpage.
// Each mirrors the corresponding module in ddgs/engines/.

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"math/big"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/antchfx/htmlquery"
)

func randInt(min, max int64) int64 {
	n, err := rand.Int(rand.Reader, big.NewInt(max-min+1))
	if err != nil {
		return min
	}
	return min + n.Int64()
}

// googleUA returns one random Android Google App User-Agent string,
// mirroring ddgs.engines.google.get_ua (including the "NSTNWV" suffix).
func googleUA() string {
	devices := []struct {
		androidVer, device   string
		chromeMin, chromeMax int64
	}{
		{"5.0", "SM-G900P Build/LRX21T", 39, 60},
		{"6.0", "Nexus 5 Build/MRA58N", 39, 60},
		{"8.0", "Pixel 2 Build/OPD3.170816.012", 39, 60},
	}
	d := devices[randInt(0, int64(len(devices)-1))]
	return fmt.Sprintf(
		"Mozilla/5.0 (Linux; Android %s; %s) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/%d.0.%d.%d Mobile Safari/537.36NSTNWV",
		d.androidVer, d.device,
		randInt(d.chromeMin, d.chromeMax), randInt(1000, 9999), randInt(1000, 1999),
	)
}

// newGoogle mirrors ddgs.engines.google.Google.
func newGoogle(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:          "google",
		category:      "text",
		provider:      "google",
		searchURL:     "https://www.google.com/search",
		searchMethod:  "GET",
		headersUpdate: map[string]string{"User-Agent": googleUA()},
		itemsXPath:    "//div[@data-hveid][.//h3]",
		elementsXPath: map[string]string{
			"title": ".//h3//text()",
			"href":  ".//a[.//h3]/@href",
			"body":  "./div/div[last()]//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		e.http.setCookies("https://google.com", map[string]string{"CONSENT": "YES+"})
		safesearch := map[string]string{"on": "2", "moderate": "1", "off": "0"}
		country, lang := splitRegion(p.Region)
		payload := map[string]string{
			"q":      p.Query,
			"filter": safesearch[strings.ToLower(p.SafeSearch)],
			"start":  strconv.Itoa((p.Page - 1) * 10),
			"hl":     fmt.Sprintf("%s-%s", lang, strings.ToUpper(country)),
			"lr":     "lang_" + lang,
			"cr":     "country" + strings.ToUpper(country),
		}
		if p.TimeLimit != "" {
			payload["tbs"] = "qdr:" + p.TimeLimit
		}
		return payload, nil
	}
	e.postExtract = func(results []Result) []Result {
		out := make([]Result, 0, len(results))
		for _, r := range results {
			href := r.getString("href")
			if strings.HasPrefix(href, "/url?q=") {
				href = strings.SplitN(strings.SplitN(href, "?q=", 2)[1], "&", 2)[0]
				r.Set("href", href)
				href = r.getString("href")
			}
			if r.getString("title") != "" && strings.HasPrefix(href, "http") {
				out = append(out, r)
			}
		}
		return out
	}
	return e, nil
}

// newBrave mirrors ddgs.engines.brave.Brave.
func newBrave(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "brave",
		category:     "text",
		provider:     "brave",
		searchURL:    "https://search.brave.com/search",
		searchMethod: "GET",
		itemsXPath:   "//div[@data-type='web']",
		elementsXPath: map[string]string{
			"title": ".//div[(contains(@class,'title') or contains(@class,'sitename-container')) and position()=last()]//text()",
			"href":  ".//a[div[contains(@class, 'title')]]/@href",
			"body":  ".//div[contains(@class, 'snippet')]//div[contains(@class, 'content')]//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		payload := map[string]string{"q": p.Query, "source": "web"}
		country, _ := splitRegion(p.Region)
		cookies := map[string]string{country: country, "useLocation": "0"}
		if p.SafeSearch != "moderate" {
			if p.SafeSearch == "on" {
				cookies["safesearch"] = "strict"
			} else {
				cookies["safesearch"] = "off"
			}
		}
		e.http.setCookies("https://search.brave.com", cookies)
		if p.TimeLimit != "" {
			payload["tf"] = map[string]string{"d": "pd", "w": "pw", "m": "pm", "y": "py"}[p.TimeLimit]
		}
		if p.Page > 1 {
			payload["offset"] = strconv.Itoa(p.Page - 1)
		}
		return payload, nil
	}
	return e, nil
}

// newMojeek mirrors ddgs.engines.mojeek.Mojeek.
func newMojeek(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "mojeek",
		category:     "text",
		provider:     "mojeek",
		searchURL:    "https://www.mojeek.com/search",
		searchMethod: "GET",
		itemsXPath:   "//ul[contains(@class, 'results')]/li",
		elementsXPath: map[string]string{
			"title": ".//h2//text()",
			"href":  ".//h2/a/@href",
			"body":  ".//p[@class='s']//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		country, lang := splitRegion(p.Region)
		e.http.setCookies("https://www.mojeek.com", map[string]string{"arc": country, "lb": lang})
		payload := map[string]string{"q": p.Query}
		if p.SafeSearch == "on" {
			payload["safe"] = "1"
		}
		if p.Page > 1 {
			payload["s"] = strconv.Itoa((p.Page-1)*10 + 1)
		}
		return payload, nil
	}
	return e, nil
}

// yahooExtractURL sanitizes a Yahoo redirect URL, mirroring
// ddgs.engines.yahoo.extract_url.
func yahooExtractURL(u string) string {
	parts := strings.SplitN(u, "/RU=", 2)
	if len(parts) < 2 {
		return u
	}
	t := parts[1]
	t = strings.SplitN(t, "/RK=", 2)[0]
	t = strings.SplitN(t, "/RS=", 2)[0]
	decoded, err := url.QueryUnescape(strings.ReplaceAll(t, "+", " "))
	if err != nil {
		return t
	}
	return decoded
}

// tokenURLSafe generates a URL-safe random token of n bytes, mirroring
// Python secrets.token_urlsafe.
func tokenURLSafe(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return strings.Repeat("A", n)
	}
	return base64.RawURLEncoding.EncodeToString(b)
}

// newYahoo mirrors ddgs.engines.yahoo.Yahoo.
func newYahoo(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "yahoo",
		category:     "text",
		provider:     "bing",
		searchURL:    "https://search.yahoo.com/search",
		searchMethod: "GET",
		itemsXPath:   "//div[contains(@class, 'relsrch')]",
		elementsXPath: map[string]string{
			"title": ".//div[contains(@class, 'Title')]//h3//text()",
			"href":  ".//div[contains(@class, 'Title')]//a/@href",
			"body":  ".//div[contains(@class, 'Text')]//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.searchURLFn = func(_ searchParams) string {
		return fmt.Sprintf("https://search.yahoo.com/search;_ylt=%s;_ylu=%s", tokenURLSafe(18), tokenURLSafe(35))
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		payload := map[string]string{"p": p.Query}
		if p.Page > 1 {
			payload["b"] = strconv.Itoa((p.Page-1)*7 + 1)
		}
		if p.TimeLimit != "" {
			payload["btf"] = p.TimeLimit
		}
		return payload, nil
	}
	e.postExtract = func(results []Result) []Result {
		out := make([]Result, 0, len(results))
		for _, r := range results {
			href := r.getString("href")
			if strings.HasPrefix(href, "https://www.bing.com/aclick?") {
				continue
			}
			if strings.Contains(href, "/RU=") {
				r.Set("href", yahooExtractURL(href))
			}
			out = append(out, r)
		}
		return out
	}
	return e, nil
}

// newYandex mirrors ddgs.engines.yandex.Yandex.
func newYandex(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:         "yandex",
		category:     "text",
		provider:     "yandex",
		searchURL:    "https://yandex.com/search/site/",
		searchMethod: "GET",
		itemsXPath:   "//li[contains(@class, 'serp-item')]",
		elementsXPath: map[string]string{
			"title": ".//h3//text()",
			"href":  ".//h3//a/@href",
			"body":  ".//div[contains(@class, 'text')]//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(_ context.Context, p searchParams) (map[string]string, error) {
		payload := map[string]string{
			"text":     p.Query,
			"web":      "1",
			"searchid": strconv.FormatInt(randInt(1000000, 9999999), 10),
		}
		if p.Page > 1 {
			payload["p"] = strconv.Itoa(p.Page - 1)
		}
		return payload, nil
	}
	return e, nil
}

// newStartpage mirrors ddgs.engines.startpage.Startpage.
func newStartpage(proxy string, timeout time.Duration, verify Verify) (searchEngine, error) {
	e, err := newBaseEngine(engineSpec{
		name:          "startpage",
		category:      "text",
		provider:      "google",
		searchURL:     "https://www.startpage.com/sp/search",
		searchMethod:  "POST",
		headersUpdate: map[string]string{"Referer": "https://www.startpage.com/"},
		itemsXPath:    "//div[contains(@class, 'result')][./a]",
		elementsXPath: map[string]string{
			"title": ".//h2//text()",
			"href":  "./a/@href",
			"body":  ".//p//text()",
		},
	}, proxy, timeout, verify)
	if err != nil {
		return nil, err
	}
	e.buildPayload = func(ctx context.Context, p searchParams) (map[string]string, error) {
		// Fetch the "sc" anti-bot token from the home page.
		resp, err := e.http.get(ctx, "https://www.startpage.com/", nil)
		if err != nil {
			return nil, err
		}
		sc := ""
		if doc, err := htmlquery.Parse(strings.NewReader(resp.Text())); err == nil {
			if node, err := htmlquery.Query(doc, `//form[@id="search"]//input[@name="sc"]/@value`); err == nil && node != nil {
				sc = nodeText(node)
			}
		}
		country, lang := splitRegion(p.Region)
		safesearch := map[string]string{"on": "heavy", "moderate": "moderate", "off": "none"}
		payload := map[string]string{
			"query":    p.Query,
			"cat":      "web",
			"t":        "device",
			"sc":       sc,
			"lui":      "english",
			"language": "english",
			"abp":      "1",
			"abd":      "0",
			"abe":      "0",
			"qsr":      fmt.Sprintf("%s_%s", lang, strings.ToUpper(country)),
			"qadf":     safesearch[strings.ToLower(p.SafeSearch)],
			"segment":  "organic",
		}
		if p.Page > 1 {
			payload["page"] = strconv.Itoa(p.Page)
		}
		if p.TimeLimit != "" {
			payload["with_date"] = p.TimeLimit
		}
		return payload, nil
	}
	return e, nil
}
