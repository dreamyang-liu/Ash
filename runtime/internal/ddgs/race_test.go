package ddgs

import (
	"context"
	"sync"
	"testing"
	"time"
)

// TestConcurrentSearchesNoRace hammers one shared DDGS + one shared engine
// from many goroutines under -race to verify goroutine safety of the
// engine cache, http client headers, and cookie jar.
func TestConcurrentSearchesNoRace(t *testing.T) {
	e := newTestEngine(t, "stub", "stubprov", ddgHTMLFixture)
	// Also exercise setCookies concurrently like brave/google/mojeek do.
	be := e.(*baseEngine)
	orig := be.buildPayload
	be.buildPayload = func(ctx context.Context, p searchParams) (map[string]string, error) {
		be.http.setCookies("https://example.com", map[string]string{"k": "v"})
		return orig(ctx, p)
	}

	d, _ := New(nil)
	saved := engineRegistry["text"]
	engineRegistry["text"] = map[string]engineFactory{
		"stub": func(string, time.Duration, Verify) (searchEngine, error) { return e, nil },
	}
	defer func() { engineRegistry["text"] = saved }()

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := d.Text(context.Background(), "golang", &SearchOptions{Backend: "stub"}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}
