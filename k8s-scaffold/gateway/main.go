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
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/go-redis/redis/v8"
)

// Common errors
var (
	ErrNotFound = errors.New("not found")
)

// Configuration structure
type Config struct {
	ListenAddr         string        // Listen address, default :80
	SessionHeader      string        // Request header to get UUID from, default X-MCP-Session-ID
	RedisAddr          string        // Redis address, default 127.0.0.1:6379
	RedisPassword      string        // Redis password, optional
	RedisDB            int           // Redis database, default 0
	RedisKeyPrefix     string        // Route table key prefix, default sandbox:
	DefaultScheme      string        // Protocol to use when only host:port is given, default http
	RedisLookupTimeout time.Duration // Redis lookup timeout, default 300ms
	ReadTimeout        time.Duration // HTTP server read timeout
	WriteTimeout       time.Duration // HTTP server write timeout
	IdleTimeout        time.Duration // HTTP server idle timeout
}

// SandboxRecord represents a sandbox record in Redis
type SandboxRecord struct {
	UUID   string
	Host   string
	Port   int
	Status string
}

// Helper functions for environment variables
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

// Load configuration from environment variables
func loadConfig() *Config {
	return &Config{
		ListenAddr:         getenv("LISTEN_ADDR", ":8080"),
		SessionHeader:      getenv("SESSION_HEADER", "X-MCP-Session-ID"),
		RedisAddr:          getenv("REDIS_ADDR", "127.0.0.1:6379"),
		RedisPassword:      os.Getenv("REDIS_PASSWORD"),
		RedisDB:            getenvInt("REDIS_DB", 0),
		RedisKeyPrefix:     getenv("ROUTE_KEY_PREFIX", "sandbox:"),
		DefaultScheme:      getenv("DEFAULT_SCHEME", "http"),
		RedisLookupTimeout: getenvDur("REDIS_LOOKUP_TIMEOUT", 300*time.Millisecond),
		ReadTimeout:        getenvDur("READ_TIMEOUT", 900*time.Second),
		WriteTimeout:       getenvDur("WRITE_TIMEOUT", 60*time.Second),
		IdleTimeout:        getenvDur("IDLE_TIMEOUT", 120*time.Second),
	}
}

var (
	rdb       *redis.Client
	config    *Config
	targetKey = &struct{}{} // context key for storing target URL
)

// Get client IP from request

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

// Look up target URL from Redis based on UUID
func lookupTarget(ctx context.Context, uuid string) (*url.URL, error) {
	key := config.RedisKeyPrefix + uuid

	// Use Redis pipeline for efficiency
	pipe := rdb.Pipeline()
	getHostCmd := pipe.HGet(ctx, key, "host")
	getPortCmd := pipe.HGet(ctx, key, "port")

	// Execute pipeline
	_, err := pipe.Exec(ctx)
	if err != nil && err != redis.Nil {
		return nil, fmt.Errorf("redis pipeline error: %w", err)
	}

	// Get host
	host, err := getHostCmd.Result()
	if err == redis.Nil || host == "" {
		return nil, ErrNotFound
	}

	// Get port
	portStr, err := getPortCmd.Result()
	if err == redis.Nil || portStr == "" {
		// Default to port 3000 if not specified
		portStr = "3000"
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, fmt.Errorf("invalid port %q: %w", portStr, err)
	}

	log.Printf("[lookup] UUID %s -> Host %s, Port %d", uuid, host, port)
	return url.Parse(fmt.Sprintf("%s://%s:%d/mcp", config.DefaultScheme, host, port))
}

func main() {
	// Load configuration
	config = loadConfig()
	log.Printf("[config] listen=%s sessionHeader=%s redis=%s db=%d prefix=%s defaultScheme=%s",
		config.ListenAddr, config.SessionHeader, config.RedisAddr, config.RedisDB,
		config.RedisKeyPrefix, config.DefaultScheme)

	// Initialize Redis client
	rdb = redis.NewClient(&redis.Options{
		Addr:         config.RedisAddr,
		Password:     config.RedisPassword,
		DB:           config.RedisDB,
		DialTimeout:  5 * time.Second,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
		PoolSize:     10,
		MinIdleConns: 5,
	})

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("redis ping failed: %v", err)
	}

	// Configure transport for reverse proxy
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = http.ProxyFromEnvironment
	transport.MaxIdleConns = 256
	transport.MaxIdleConnsPerHost = 128
	transport.IdleConnTimeout = 90 * time.Second
	transport.TLSHandshakeTimeout = 10 * time.Second
	transport.ExpectContinueTimeout = 1 * time.Second
	transport.ResponseHeaderTimeout = 10 * time.Second

	// Create reverse proxy
	proxy := &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			// Get target URL from context
			u, _ := r.Context().Value(targetKey).(*url.URL)
			if u == nil {
				log.Printf("[director] no target URL in context (skip) method=%s path=%q", r.Method, r.URL.Path)
				return
			}

			// Log original request details
			origHost := r.Host
			origPath := r.URL.Path
			origQuery := r.URL.RawQuery
			xffBefore := r.Header.Get("X-Forwarded-For")

			if os.Getenv("DEBUG") == "true" {
				log.Printf("[director][before] method=%s origHost=%s path=%q rawQuery=%q xff=%q target=%s",
					r.Method, origHost, origPath, origQuery, xffBefore, u.String())
			}

			// Set scheme and host
			r.URL.Scheme = u.Scheme
			r.URL.Host = u.Host

			// Handle path joining
			if u.Path != "" && u.Path != "/" {
				if !strings.HasPrefix(r.URL.Path, u.Path) {
					r.URL.Path = singleJoin(u.Path, r.URL.Path)
				}
			}

			// Set host header to upstream host
			r.Host = u.Host

			// Add X-Forwarded headers
			ip := clientIP(r)
			if xffBefore != "" {
				r.Header.Set("X-Forwarded-For", xffBefore+", "+ip)
			} else {
				r.Header.Set("X-Forwarded-For", ip)
			}
			r.Header.Set("X-Forwarded-Host", origHost)
			r.Header.Set("X-Forwarded-Proto", "http") // Adjust if using HTTPS

			if os.Getenv("DEBUG") == "true" {
				log.Printf("[director][after] forwardTo=%s path=%q xff=%q",
					u.String(), r.URL.Path, r.Header.Get("X-Forwarded-For"))
			}
		},

		Transport:     transport,
		FlushInterval: 50 * time.Millisecond,

		// Log response status
		ModifyResponse: func(resp *http.Response) error {
			if resp.StatusCode >= 400 || os.Getenv("DEBUG") == "true" {
				log.Printf("[proxy][resp] status=%d url=%s", resp.StatusCode, resp.Request.URL.String())
			}
			return nil
		},

		// Handle errors
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			u, _ := r.Context().Value(targetKey).(*url.URL)
			if u != nil {
				log.Printf("[proxy][error] upstream error: %v target=%s method=%s path=%q",
					err, u.String(), r.Method, r.URL.Path)
			} else {
				log.Printf("[proxy][error] upstream error: %v (no target) method=%s path=%q",
					err, r.Method, r.URL.Path)
			}

			// Return appropriate error based on the type
			if errors.Is(err, context.DeadlineExceeded) {
				http.Error(w, "gateway timeout", http.StatusGatewayTimeout)
			} else {
				http.Error(w, "bad gateway", http.StatusBadGateway)
			}
		},
	}

	// Create HTTP mux
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	// Readiness check endpoint
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()

		if err := rdb.Ping(ctx).Err(); err != nil {
			w.Header().Set("Content-Type", "text/plain")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("redis not ready"))
			return
		}

		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ready"))
	})

	// Main handler for proxying requests
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Get UUID from header
		uuid := strings.TrimSpace(r.Header.Get(config.SessionHeader))
		if uuid == "" {
			http.Error(w, "missing session header", http.StatusBadRequest)
			return
		}

		// Look up target with timeout
		ctx, cancel := context.WithTimeout(r.Context(), config.RedisLookupTimeout)
		defer cancel()

		u, err := lookupTarget(ctx, uuid)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				log.Printf("[gateway] UUID not found: %s", uuid)
				http.Error(w, "route not found", http.StatusNotFound)
				return
			}
			log.Printf("[redis] lookup error: %v", err)
			http.Error(w, "route lookup error", http.StatusBadGateway)
			return
		}

		// Add target URL to context and proxy the request
		ctx = context.WithValue(r.Context(), targetKey, u)
		if os.Getenv("DEBUG") == "true" {
			log.Printf("[gateway] routing request: method=%s path=%q target=%s", r.Method, r.URL.Path, u.String())
		}
		proxy.ServeHTTP(w, r.WithContext(ctx))
	})

	// Create HTTP server with timeouts
	srv := http.Server{
		Addr:              config.ListenAddr,
		Handler:           mux,
		ReadTimeout:       config.ReadTimeout,
		WriteTimeout:      config.WriteTimeout,
		IdleTimeout:       config.IdleTimeout,
		ReadHeaderTimeout: 5 * time.Second,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("[gateway] listening on %s", config.ListenAddr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Create shutdown context with timeout
	ctx, cancel = context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Shutdown the server
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited properly")
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
