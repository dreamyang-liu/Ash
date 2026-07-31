package tools

// ArtifactTool is the runtime primitive behind SDK-side custom tools:
// "ensure a file with the given sha256 exists locally, downloading it from
// a URL if needed". It knows nothing about specific tools — which URL,
// which hash, and how the binary is invoked all live in SDK-side manifests
// (swebench/agent/custom_tools.py), which route custom tool calls through
// artifact + shell. Registered in tool.go All().

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	// artifactCacheDir is where verified artifacts live, keyed by sha256.
	artifactCacheDir = "/tmp/ash-artifacts"
	// maxArtifactBytes caps downloads (anti-abuse).
	maxArtifactBytes = 500 << 20 // 500 MiB
	// artifactDownloadTimeout bounds a single download.
	artifactDownloadTimeout = 120 * time.Second
)

// artifactFlights deduplicates concurrent downloads of the same hash
// (singleflight): callers for the same sha256 share one download.
var artifactFlights = struct {
	mu  sync.Mutex
	m   map[string]*sync.Once
	res map[string]error // outcome of each flight, for followers
}{m: map[string]*sync.Once{}, res: map[string]error{}}

type ArtifactTool struct{}

func (a *ArtifactTool) Name() string { return "artifact" }

func (a *ArtifactTool) Description() string {
	return "Ensure a file with the given sha256 exists locally, downloading from a URL if not cached. Returns the local path."
}

func (a *ArtifactTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url":        map[string]any{"type": "string", "description": "HTTP(S) URL to download the artifact from if not cached"},
			"sha256":     map[string]any{"type": "string", "description": "Optional expected sha256 hex digest; when set, download fails if the content does not match"},
			"executable": map[string]any{"type": "boolean", "default": true, "description": "chmod +x the cached file"},
		},
		"required": []string{"url"},
	}
}

// artifactPath returns the cache location for a cache key.
func artifactPath(key string) string {
	return filepath.Join(artifactCacheDir, key[:16], "artifact")
}

// artifactCacheKey derives the cache key: the content hash when pinned,
// otherwise a URL-derived hash (unverified artifacts are cached per URL).
func artifactCacheKey(rawURL, sum string) string {
	if sum != "" {
		return sum
	}
	h := sha256.Sum256([]byte("url:" + rawURL))
	return hex.EncodeToString(h[:])
}

func (a *ArtifactTool) Execute(args map[string]any) Result {
	rawURL, _ := args["url"].(string)
	sum, _ := args["sha256"].(string)
	executable := true
	if e, ok := args["executable"].(bool); ok {
		executable = e
	}

	if rawURL == "" {
		return Err("url is required")
	}
	sum = strings.ToLower(strings.TrimSpace(sum))
	if sum != "" && (len(sum) != 64 || !isHex(sum)) {
		return Err("sha256 must be a 64-char hex digest")
	}
	if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
		return Err("url must be http(s)")
	}

	key := artifactCacheKey(rawURL, sum)
	dest := artifactPath(key)

	// Fast path: verified artifact already cached.
	if _, err := os.Stat(dest); err == nil {
		return Ok(dest)
	}

	// Singleflight: one download per cache key, concurrent callers wait.
	artifactFlights.mu.Lock()
	once, ok := artifactFlights.m[key]
	if !ok {
		once = &sync.Once{}
		artifactFlights.m[key] = once
	}
	artifactFlights.mu.Unlock()

	once.Do(func() {
		err := downloadAndVerify(rawURL, sum, dest, executable)
		artifactFlights.mu.Lock()
		artifactFlights.res[key] = err
		if err != nil {
			// Allow retry on a later call.
			delete(artifactFlights.m, key)
		}
		artifactFlights.mu.Unlock()
	})

	artifactFlights.mu.Lock()
	err := artifactFlights.res[key]
	artifactFlights.mu.Unlock()
	if err != nil {
		return Err(err.Error())
	}
	return Ok(dest)
}

// downloadAndVerify streams the URL to a temp file while hashing, verifies
// the digest when one is pinned (sum != ""), then atomically installs it
// at dest.
func downloadAndVerify(rawURL, sum, dest string, executable bool) error {
	client := &http.Client{Timeout: artifactDownloadTimeout}
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return fmt.Errorf("invalid url: %w", err)
	}
	req.Header.Set("User-Agent", defaultUA)

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("creating cache dir: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dest), ".download-*")
	if err != nil {
		return fmt.Errorf("creating temp file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op after successful rename

	hasher := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, hasher), io.LimitReader(resp.Body, maxArtifactBytes+1))
	closeErr := tmp.Close()
	if err != nil {
		return fmt.Errorf("download interrupted: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("writing artifact: %w", closeErr)
	}
	if n > maxArtifactBytes {
		return fmt.Errorf("artifact exceeds size limit (%d bytes)", maxArtifactBytes)
	}

	if sum != "" {
		got := hex.EncodeToString(hasher.Sum(nil))
		if got != sum {
			return fmt.Errorf("sha256 mismatch: got %s, want %s", got, sum)
		}
	}

	mode := os.FileMode(0o644)
	if executable {
		mode = 0o755
	}
	if err := os.Chmod(tmpName, mode); err != nil {
		return fmt.Errorf("chmod: %w", err)
	}
	if err := os.Rename(tmpName, dest); err != nil {
		return fmt.Errorf("installing artifact: %w", err)
	}
	return nil
}

func isHex(s string) bool {
	for _, r := range s {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return false
		}
	}
	return true
}
