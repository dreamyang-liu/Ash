package ddgs

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"strings"
	"time"

	"golang.org/x/net/publicsuffix"
)

// defaultTimeout mirrors the Python DDGS default timeout of 5 seconds.
const defaultTimeout = 5 * time.Second

// browserUserAgents is a small pool of current desktop browser User-Agent
// strings, standing in for primp's TLS impersonation ("impersonate=random").
var browserUserAgents = []string{
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
}

// httpResponse mirrors ddgs.http_client.Response.
type httpResponse struct {
	StatusCode int
	Content    []byte
}

func (r *httpResponse) Text() string { return string(r.Content) }

// httpClient mirrors ddgs.http_client.HttpClient: one client per engine
// instance, with its own cookie jar, default browser-like headers, proxy,
// timeout and TLS-verification settings.
type httpClient struct {
	client    *http.Client
	headers   http.Header
	userAgent string
}

// newHTTPClient creates an httpClient.
//
//	proxy:   "" or a http/https/socks5 proxy URL.
//	timeout: zero means defaultTimeout.
//	verify:  TLS verification options (skip verification or custom CA file).
func newHTTPClient(proxy string, timeout time.Duration, verify Verify) (*httpClient, error) {
	if timeout <= 0 {
		timeout = defaultTimeout
	}

	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
	}
	if proxy != "" {
		proxyURL, err := url.Parse(proxy)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid proxy %q: %v", ErrDDGS, proxy, err)
		}
		transport.Proxy = http.ProxyURL(proxyURL)
	}

	tlsConfig := &tls.Config{}
	if verify.SkipVerify {
		tlsConfig.InsecureSkipVerify = true
	}
	if verify.CACertFile != "" {
		pem, err := os.ReadFile(verify.CACertFile)
		if err != nil {
			return nil, fmt.Errorf("%w: reading CA cert file: %v", ErrDDGS, err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("%w: no certificates found in %s", ErrDDGS, verify.CACertFile)
		}
		tlsConfig.RootCAs = pool
	}
	transport.TLSClientConfig = tlsConfig

	jar, err := cookiejar.New(&cookiejar.Options{PublicSuffixList: publicsuffix.List})
	if err != nil {
		return nil, fmt.Errorf("%w: creating cookie jar: %v", ErrDDGS, err)
	}

	ua := browserUserAgents[rand.Intn(len(browserUserAgents))]
	headers := http.Header{}
	headers.Set("User-Agent", ua)
	headers.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
	headers.Set("Accept-Language", "en-US,en;q=0.9")

	return &httpClient{
		client: &http.Client{
			Transport: transport,
			Jar:       jar,
			Timeout:   timeout,
		},
		headers:   headers,
		userAgent: ua,
	}, nil
}

// headersUpdate merges headers into the client defaults, mirroring
// primp Client.headers_update.
func (c *httpClient) headersUpdate(h map[string]string) {
	for k, v := range h {
		c.headers.Set(k, v)
	}
}

// setCookies sets cookies for a URL, mirroring primp Client.set_cookies.
func (c *httpClient) setCookies(rawURL string, cookies map[string]string) {
	if !strings.Contains(rawURL, "://") {
		rawURL = "https://" + rawURL
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return
	}
	list := make([]*http.Cookie, 0, len(cookies))
	for k, v := range cookies {
		list = append(list, &http.Cookie{Name: k, Value: v})
	}
	c.client.Jar.SetCookies(u, list)
}

// request performs an HTTP request.
//
//	method:  "GET" or "POST"
//	rawURL:  target URL
//	params:  query params (GET) — appended to the URL
//	data:    form data (POST) — sent urlencoded in the body
func (c *httpClient) request(
	ctx context.Context,
	method, rawURL string,
	params, data map[string]string,
) (*httpResponse, error) {
	if len(params) > 0 {
		q := url.Values{}
		for k, v := range params {
			q.Set(k, v)
		}
		sep := "?"
		if strings.Contains(rawURL, "?") {
			sep = "&"
		}
		rawURL += sep + q.Encode()
	}

	var body io.Reader
	if len(data) > 0 {
		form := url.Values{}
		for k, v := range data {
			form.Set(k, v)
		}
		body = strings.NewReader(form.Encode())
	}

	req, err := http.NewRequestWithContext(ctx, method, rawURL, body)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrDDGS, err)
	}
	for k, vals := range c.headers {
		for _, v := range vals {
			req.Header.Set(k, v)
		}
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	resp, err := c.client.Do(req)
	if err != nil {
		if isTimeoutError(err) {
			return nil, fmt.Errorf("%w: %v", ErrTimeout, err)
		}
		return nil, fmt.Errorf("%w: %v", ErrDDGS, err)
	}
	defer resp.Body.Close()

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		if isTimeoutError(err) {
			return nil, fmt.Errorf("%w: %v", ErrTimeout, err)
		}
		return nil, fmt.Errorf("%w: reading body: %v", ErrDDGS, err)
	}
	return &httpResponse{StatusCode: resp.StatusCode, Content: content}, nil
}

func (c *httpClient) get(ctx context.Context, rawURL string, params map[string]string) (*httpResponse, error) {
	return c.request(ctx, http.MethodGet, rawURL, params, nil)
}

func isTimeoutError(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr interface{ Timeout() bool }
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	// http.Client wraps timeouts in url.Error with "Client.Timeout exceeded"
	return strings.Contains(err.Error(), "Client.Timeout exceeded")
}
