package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
)

type cfg struct {
	ListenAddr         string         // Listen address, default :8080
	SessionHeader      string         // Request header to get UUID from, default X-MCP-Session-ID
	RedisAddr          string         // 127.0.0.1:6379 or redis:6379
	RedisPassword      string         // Optional
	RedisDB            int            // Default 0
	RedisKeyPrefix     string         // Route table key prefix, default mcp:route:
	DefaultScheme      string         // Protocol to use when only host:port is given, default http
	RedisLookupTimeout time.Duration  // Redis lookup timeout, default 300ms
	AllowedHostRegex   *regexp.Regexp // Allowed target host regex (prevents SSRF), default only intranet/localhost
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
func getenvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
func getenvDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func loadCfg() *cfg {
	pat := getenv("ALLOWED_HOST_PATTERN", `^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|localhost$|127\.0\.0\.1$|\[::1\])`)
	re, err := regexp.Compile(pat)
	if err != nil {
		log.Fatalf("bad ALLOWED_HOST_PATTERN: %v", err)
	}
	return &cfg{
		ListenAddr:         getenv("LISTEN_ADDR", ":8080"),
		SessionHeader:      getenv("SESSION_HEADER", "X-MCP-Session-ID"),
		RedisAddr:          getenv("REDIS_ADDR", "127.0.0.1:6379"),
		RedisPassword:      os.Getenv("REDIS_PASSWORD"),
		RedisDB:            getenvInt("REDIS_DB", 0),
		RedisKeyPrefix:     getenv("ROUTE_KEY_PREFIX", "sandbox:"),
		DefaultScheme:      getenv("DEFAULT_SCHEME", "http"),
		RedisLookupTimeout: getenvDur("REDIS_LOOKUP_TIMEOUT", 300*time.Millisecond),
		AllowedHostRegex:   re,
	}
}

var (
	rdb       *redis.Client
	cfgv      *cfg
	targetKey = &struct{}{} // context key
)

type routeJSON struct {
	Host       string `json:"host"`
	Port       int    `json:"port"`
	UUID       string `json:"uuid"`
	Status     string `json:"status"`
	ExpireTime string `json:"expire_time"`
}

func parseRouteValue(val string) (*url.URL, error) {
	val = strings.TrimSpace(val)
	if val == "" {
		return nil, errors.New("empty route")
	}
	if strings.HasPrefix(val, "http://") || strings.HasPrefix(val, "https://") {
		return url.Parse(val)
	}
	if strings.Contains(val, "://") {

		return url.Parse(val)
	}

	if (strings.HasPrefix(val, "{") && strings.HasSuffix(val, "}")) || strings.Contains(val, `"host"`) {
		var r routeJSON
		if err := json.Unmarshal([]byte(val), &r); err == nil {
			return url.Parse(fmt.Sprintf("http://%s:%d/mcp", r.Host, r.Port))
		}
	}

	if _, _, err := net.SplitHostPort(val); err == nil {
		return url.Parse(fmt.Sprintf("%s://%s", cfgv.DefaultScheme, val))
	}
	return nil, fmt.Errorf("unrecognized route value: %q", val)
}

func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	h, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return h
}

func lookupTarget(ctx context.Context, uuid string) (*url.URL, error) {
	// key := cfgv.RedisKeyPrefix + uuid

	// // 先 Get
	// val, err := rdb.Get(ctx, key).Result()
	// if err == nil {
	// 	return parseRouteValue(val)
	// }
	// if err != redis.Nil {
	// 	return nil, err
	// }

	// hm, err := rdb.HGetAll(ctx, key).Result()
	// if err == nil && len(hm) > 0 {
	// 	if rawURL := hm["url"]; rawURL != "" {
	// 		return parseRouteValue(rawURL)
	// 	}
	// 	if host := hm["host"]; host != "" {
	// 		port := hm["port"]
	// 		if port == "" {
	// 			port = "80"
	// 		}
	// 		scheme := hm["scheme"]
	// 		if scheme == "" {
	// 			scheme = cfgv.DefaultScheme
	// 		}
	// 		return url.Parse(fmt.Sprintf("%s://%s:%s%s", scheme, host, port, hm["path"]))
	// 	}
	// }
	log.Printf("[redis] lookup %s not found", uuid)
	return url.Parse("http://127.0.0.1:3000")
	// return nil, redis.Nil
}

func main() {
	cfgv = loadCfg()
	log.Printf("[cfg] listen=%s sessionHeader=%s redis=%s db=%d prefix=%s defaultScheme=%s",
		cfgv.ListenAddr, cfgv.SessionHeader, cfgv.RedisAddr, cfgv.RedisDB, cfgv.RedisKeyPrefix, cfgv.DefaultScheme)

	rdb = redis.NewClient(&redis.Options{
		Addr:     cfgv.RedisAddr,
		Password: cfgv.RedisPassword,
		DB:       cfgv.RedisDB,
	})

	if err := rdb.Ping(context.Background()).Err(); err != nil {
		log.Fatalf("redis ping failed: %v", err)
	}

	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = http.ProxyFromEnvironment
	transport.MaxIdleConns = 256
	transport.MaxIdleConnsPerHost = 128
	transport.IdleConnTimeout = 90 * time.Second

	proxy := &httputil.ReverseProxy{
		Director: func(r *http.Request) {

			u, _ := r.Context().Value(targetKey).(*url.URL)
			if u == nil {
				return
			}

			origHost := r.Host
			r.URL.Scheme = u.Scheme
			r.URL.Host = u.Host

			if u.Path != "" && u.Path != "/" {

				if !strings.HasPrefix(r.URL.Path, u.Path) {
					r.URL.Path = singleJoin(u.Path, r.URL.Path)
				}
			}
			r.Host = u.Host

			ip := clientIP(r)
			if prior := r.Header.Get("X-Forwarded-For"); prior != "" {
				r.Header.Set("X-Forwarded-For", prior+", "+ip)
			} else {
				r.Header.Set("X-Forwarded-For", ip)
			}
			r.Header.Set("X-Forwarded-Host", origHost)
		},
		Transport:     transport,
		FlushInterval: 50 * time.Millisecond,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("[proxy] upstream error: %v", err)
			http.Error(w, "bad gateway", http.StatusBadGateway)
		},
	}

	http.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	http.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {

		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		if err := rdb.Ping(ctx).Err(); err != nil {
			http.Error(w, "redis not ready", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ready"))
	})

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		uuid := strings.TrimSpace(r.Header.Get(cfgv.SessionHeader))
		log.Printf("[gateway] Headers %s", uuid)
		if uuid == "" {
			http.Error(w, "missing session header", http.StatusBadRequest)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), cfgv.RedisLookupTimeout)
		defer cancel()
		u, err := lookupTarget(ctx, uuid)
		log.Printf("[gateway] Host: %s", u.Host)
		if err != nil {
			if errors.Is(err, redis.Nil) {
				http.Error(w, "route not found", http.StatusNotFound)
				return
			}
			log.Printf("[redis] lookup error: %v", err)
			http.Error(w, "route lookup error", http.StatusBadGateway)
			return
		}

		host := u.Hostname()
		if !cfgv.AllowedHostRegex.MatchString(host) {
			http.Error(w, "forbidden target host", http.StatusForbidden)
			return
		}

		ctx = context.WithValue(r.Context(), targetKey, u)
		proxy.ServeHTTP(w, r.WithContext(ctx))
	})

	srv := &http.Server{
		Addr:              cfgv.ListenAddr,
		ReadHeaderTimeout: 30 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	log.Printf("[gateway] listening on %s", cfgv.ListenAddr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server error: %v", err)
	}
}

func singleJoin(a, b string) string {
	if a == "" || a == "/" {
		return b
	}
	if b == "" || b == "/" {
		return a
	}
	as := strings.HasSuffix(a, "/")
	bs := strings.HasPrefix(b, "/")
	switch {
	case as && bs:
		return a + b[1:]
	case !as && !bs:
		return a + "/" + b
	default:
		return a + b
	}
}
