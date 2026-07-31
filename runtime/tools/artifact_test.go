package tools

// Tests for artifact.go (run by `go test ./tools`, same pattern as
// web_test.go; no other file covers this): download+verify+cache, hash
// mismatch rejection, arg validation, concurrent singleflight dedup,
// HTTP error propagation. Writes only to artifactCacheDir under /tmp.
// User instruction: custom tools — "binary可以给一个url 然后运行时下载" / "ok".

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func TestArtifactDownloadVerifyCache(t *testing.T) {
	content := []byte("#!/bin/sh\necho custom-tool-ok\n")
	sum := sha256Hex(content)
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Write(content)
	}))
	defer server.Close()
	t.Cleanup(func() { os.RemoveAll(artifactPath(sum)) })

	tool := &ArtifactTool{}
	r := tool.Execute(map[string]any{"url": server.URL, "sha256": sum})
	if !r.Success {
		t.Fatalf("artifact failed: %s", r.Error)
	}
	path := r.Output
	got, err := os.ReadFile(path)
	if err != nil || string(got) != string(content) {
		t.Fatalf("cached file wrong: %v", err)
	}
	info, _ := os.Stat(path)
	if info.Mode()&0o111 == 0 {
		t.Fatal("artifact should be executable by default")
	}

	// Second call: cache hit, no new download.
	r2 := tool.Execute(map[string]any{"url": server.URL, "sha256": sum})
	if !r2.Success || r2.Output != path {
		t.Fatalf("cache hit failed: %#v", r2)
	}
	if hits.Load() != 1 {
		t.Fatalf("expected 1 download, got %d", hits.Load())
	}
}

func TestArtifactHashMismatchRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("malicious content"))
	}))
	defer server.Close()

	wrong := sha256Hex([]byte("expected content"))
	t.Cleanup(func() { os.RemoveAll(artifactPath(wrong)) })

	r := (&ArtifactTool{}).Execute(map[string]any{"url": server.URL, "sha256": wrong})
	if r.Success || !strings.Contains(r.Error, "sha256 mismatch") {
		t.Fatalf("expected hash mismatch error, got: %#v", r)
	}
	if _, err := os.Stat(artifactPath(wrong)); err == nil {
		t.Fatal("mismatched artifact must not be installed in cache")
	}
}

func TestArtifactValidation(t *testing.T) {
	tool := &ArtifactTool{}
	cases := []struct {
		args map[string]any
		want string
	}{
		{map[string]any{"sha256": strings.Repeat("a", 64)}, "url is required"},
		{map[string]any{"url": "https://x.com", "sha256": "short"}, "sha256 must be"},
		{map[string]any{"url": "https://x.com", "sha256": strings.Repeat("Z", 64)}, "sha256 must be"},
		{map[string]any{"url": "ftp://x.com/f", "sha256": strings.Repeat("a", 64)}, "must be http"},
	}
	for _, c := range cases {
		r := tool.Execute(c.args)
		if r.Success || !strings.Contains(r.Error, c.want) {
			t.Errorf("args %v: want error %q, got %#v", c.args, c.want, r)
		}
	}
}

func TestArtifactSingleflight(t *testing.T) {
	content := []byte("concurrent artifact body")
	sum := sha256Hex(content)
	var hits atomic.Int32
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		<-release // hold all requests until everyone has piled up
		w.Write(content)
	}))
	defer server.Close()
	t.Cleanup(func() { os.RemoveAll(artifactPath(sum)) })

	tool := &ArtifactTool{}
	var wg sync.WaitGroup
	errs := make([]string, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := tool.Execute(map[string]any{"url": server.URL, "sha256": sum})
			if !r.Success {
				errs[i] = r.Error
			}
		}(i)
	}
	close(release)
	wg.Wait()
	for i, e := range errs {
		if e != "" {
			t.Fatalf("caller %d failed: %s", i, e)
		}
	}
	if hits.Load() != 1 {
		t.Fatalf("singleflight: expected 1 download for 8 callers, got %d", hits.Load())
	}
}

func TestArtifactHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusForbidden)
	}))
	defer server.Close()
	sum := sha256Hex([]byte(fmt.Sprintf("unique-%s", server.URL)))
	r := (&ArtifactTool{}).Execute(map[string]any{"url": server.URL, "sha256": sum})
	if r.Success || !strings.Contains(r.Error, "HTTP 403") {
		t.Fatalf("expected HTTP error, got: %#v", r)
	}
}
