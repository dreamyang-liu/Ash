package main

import (
	"context"
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
		ListenAddr:         getenv("LISTEN_ADDR", ":80"),
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

type SandboxStatus string

var ErrNotFound = errors.New("not found")

const (
	StatusStarting SandboxStatus = "starting"
	StatusReady    SandboxStatus = "ready"
	StatusFailed   SandboxStatus = "failed"
	StatusStopped  SandboxStatus = "stopped"
)

type SandboxRecord struct {
	UUID   string
	IP     string
	Port   int
	Status SandboxStatus
	MaxTTL time.Duration // 以秒为单位存储时，这里换算为 Duration
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
	key := cfgv.RedisKeyPrefix + uuid

	m, err := rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("redis HGETALL: %w", err)
	}
	// HGetAll 对不存在的 key 返回空 map，不是 redis.Nil
	if len(m) == 0 {
		return nil, ErrNotFound
	}

	// 2) 解析各字段
	var rec SandboxRecord
	rec.IP = m["host"]
	// rec.Status = SandboxStatus(m["status"])

	// port
	if p := m["port"]; p != "" {
		port, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("bad port %q: %w", p, err)
		}
		rec.Port = port
	}
	log.Printf("[lookup] UUID %s -> IP %s, Port %d", uuid, rec.IP, rec.Port)
	return url.Parse(fmt.Sprintf("%s://%s:%d/mcp", cfgv.DefaultScheme, rec.IP, rec.Port))
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
			// 取目标
			u, _ := r.Context().Value(targetKey).(*url.URL)
			if u == nil {
				log.Printf("[director] no target URL in context (skip) method=%s path=%q", r.Method, r.URL.Path)
				return
			}

			// 记录进入前信息
			origHost := r.Host
			origPath := r.URL.Path
			origQuery := r.URL.RawQuery
			xffBefore := r.Header.Get("X-Forwarded-For")

			log.Printf("[director][before] method=%s origHost=%s path=%q rawQuery=%q xff=%q target=%s",
				r.Method, origHost, origPath, origQuery, xffBefore, u.String())

			// 设置 scheme/host（目标）
			r.URL.Scheme = u.Scheme
			r.URL.Host = u.Host

			// 路径处理决策
			decision := "noop"
			if u.Path != "" && u.Path != "/" {
				if !strings.HasPrefix(r.URL.Path, u.Path) {
					newPath := singleJoin(u.Path, r.URL.Path)
					decision = fmt.Sprintf("join(%q, %q) -> %q", u.Path, origPath, newPath)
					r.URL.Path = newPath
				} else {
					decision = fmt.Sprintf("keep (already prefixed by %q)", u.Path)
				}
			} else {
				decision = "skip (u.Path empty or /)"
			}

			// Host 头改为上游主机
			r.Host = u.Host

			// 维护 XFF
			ip := clientIP(r)
			if xffBefore != "" {
				r.Header.Set("X-Forwarded-For", xffBefore+", "+ip)
			} else {
				r.Header.Set("X-Forwarded-For", ip)
			}
			r.Header.Set("X-Forwarded-Host", origHost)

			log.Printf("[director][after]  forwardTo scheme=%s host=%s path=%q rawQuery=%q HostHdr=%s decision=%s xff=%q",
				r.URL.Scheme, r.URL.Host, r.URL.Path, r.URL.RawQuery, r.Host, decision, r.Header.Get("X-Forwarded-For"))
		},

		Transport:     transport,
		FlushInterval: 50 * time.Millisecond,

		// 可选：记录上游响应状态与最终 URL
		ModifyResponse: func(resp *http.Response) error {
			// resp.Request.URL 是已经改写后的目标 URL
			log.Printf("[proxy][resp] status=%d url=%s", resp.StatusCode, resp.Request.URL.String())
			return nil
		},

		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			// r.Context() 里仍然能拿到目标
			if u, _ := r.Context().Value(targetKey).(*url.URL); u != nil {
				log.Printf("[proxy][error] upstream error: %v target=%s method=%s path=%q",
					err, u.String(), r.Method, r.URL.Path)
			} else {
				log.Printf("[proxy][error] upstream error: %v (no target) method=%s path=%q",
					err, r.Method, r.URL.Path)
			}
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
		log.Printf("[gateway] Host: %s, err: %v", u.Host, err)
		if err != nil {
			if errors.Is(err, redis.Nil) {
				http.Error(w, "route not found", http.StatusNotFound)
				return
			}
			log.Printf("[redis] lookup error: %v", err)
			http.Error(w, "route lookup error", http.StatusBadGateway)
			return
		}

		// host := u.Hostname()
		// if !cfgv.AllowedHostRegex.MatchString(host) {
		// 	http.Error(w, "forbidden target host", http.StatusForbidden)
		// 	return
		// }

		ctx = context.WithValue(r.Context(), targetKey, u)
		log.Printf("[gateway] routing request: method=%s path=%q target=%s", r.Method, r.URL.Path, u.String())
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
