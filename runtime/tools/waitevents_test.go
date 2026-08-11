package tools

// Tests for waitevents.go, the tool surface over the event log (run by
// `go test ./tools`; events/events_test.go covers the delivery model).
// Verifies: an existing event returns immediately, a wait wakes on a
// concurrent push, a timeout is reported rather than hanging, an unrequested
// event is left for other consumers, and events lost before delivery are
// reported as missed.
// User instruction: "我觉得可以搞一个 wait_for_events 这个东西" and the later
// opt-in model ("默认是谁都看不到...监听了kind 才能看到").

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/dreamyang-liu/ash/runtime/events"
)

type waitPayload struct {
	Events []struct {
		Kind    string `json:"kind"`
		Source  string `json:"source"`
		AgentID string `json:"agent_id"`
	} `json:"events"`
	TimedOut  bool `json:"timed_out"`
	WaitedFor int  `json:"waited_for"`
	Missed    int  `json:"missed"`
}

func runWait(t *testing.T, args map[string]any) waitPayload {
	t.Helper()
	r := (&WaitEventsTool{}).Execute(args)
	if !r.Success {
		t.Fatalf("wait_for_events failed: %s", r.Error)
	}
	var p waitPayload
	if err := json.Unmarshal([]byte(r.Output), &p); err != nil {
		t.Fatalf("bad payload %q: %v", r.Output, err)
	}
	return p
}

// drainAll clears all event state so tests don't leak into each other.
func drainAll(t *testing.T) {
	t.Helper()
	events.ResetForTest()
}

func TestWaitForEventsReturnsQueuedImmediately(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	events.Push("file_change", "/tmp/x", map[string]any{"path": "/tmp/x"})

	start := time.Now()
	p := runWait(t, map[string]any{"kinds": []any{"file_change"}, "timeout": float64(10)})
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("already-queued event should return immediately, took %v", elapsed)
	}
	if p.TimedOut || len(p.Events) != 1 || p.Events[0].Kind != "file_change" {
		t.Fatalf("unexpected payload: %+v", p)
	}
}

func TestWaitForEventsWakesOnPush(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	go func() {
		time.Sleep(150 * time.Millisecond)
		events.PushTo("agent-a", "process_exited", "pid-7", map[string]any{"exit_code": 0})
	}()

	start := time.Now()
	p := runWait(t, map[string]any{
		"kinds":    []any{"process_exited"},
		"timeout":  float64(10),
		"agent_id": "agent-a",
	})
	elapsed := time.Since(start)
	if p.TimedOut || len(p.Events) != 1 {
		t.Fatalf("expected the pushed event, got %+v", p)
	}
	if elapsed < 100*time.Millisecond {
		t.Errorf("returned before the event was pushed (%v)", elapsed)
	}
	if elapsed > 3*time.Second {
		t.Errorf("wakeup too slow: %v (should not wait out the timeout)", elapsed)
	}
}

func TestWaitForEventsTimesOut(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	start := time.Now()
	p := runWait(t, map[string]any{"kinds": []any{"never_happens"}, "timeout": float64(1)})
	if !p.TimedOut || len(p.Events) != 0 {
		t.Fatalf("expected timeout, got %+v", p)
	}
	if elapsed := time.Since(start); elapsed < 900*time.Millisecond {
		t.Errorf("returned before the timeout elapsed: %v", elapsed)
	}
}

func TestWaitForEventsLeavesNonMatchingQueued(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	events.Push("file_change", "/tmp/a", map[string]any{})
	events.Push("other_kind", "/tmp/b", map[string]any{})

	p := runWait(t, map[string]any{"kinds": []any{"file_change"}, "timeout": float64(2)})
	if len(p.Events) != 1 || p.Events[0].Kind != "file_change" {
		t.Fatalf("should take only the requested kind, got %+v", p)
	}
	// The unrequested event is untouched: a subscriber to it still gets it,
	// since a wait consumes only what it asked for.
	events.Subscribe("observer", events.Filter{Kinds: []string{"other_kind"}})
	remaining := events.DrainFor("observer")
	if len(remaining) != 1 || remaining[0].Kind != "other_kind" {
		t.Fatalf("the unrequested event should still be available: %+v", remaining)
	}
}

func TestWaitForEventsAnyKind(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	events.Push("whatever", "src", map[string]any{})
	p := runWait(t, map[string]any{"timeout": float64(2)}) // no kinds filter
	if p.TimedOut || len(p.Events) != 1 {
		t.Fatalf("empty kinds should match anything, got %+v", p)
	}
}

func TestWaitForEventsReportsMissed(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	// Shrink the log bound so overflow is cheap to trigger. `missed` is
	// per-subscriber -- an event lost before anyone wanted it is nobody's
	// loss -- so the agent subscribes first.
	const capEvents = 10
	restore := events.SetQueueBoundsForTest(1<<30, capEvents)
	t.Cleanup(restore)
	events.Subscribe("agent-a", events.Filter{Kinds: []string{"flood"}})

	for i := 0; i < capEvents+5; i++ {
		events.Push("flood", "src", map[string]any{"i": i})
	}
	p := runWait(t, map[string]any{
		"kinds":    []any{"flood"},
		"timeout":  float64(2),
		"agent_id": "agent-a",
	})
	if p.Missed != 5 {
		t.Errorf("expected 5 missed events reported, got %d", p.Missed)
	}
	if len(p.Events) != capEvents {
		t.Errorf("expected %d retained events, got %d", capEvents, len(p.Events))
	}
}

func TestSubscribeAndUnsubscribeActions(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	tool := &WaitEventsTool{}

	// subscribe requires kinds: subscribing to everything is rarely meant.
	if r := tool.Execute(map[string]any{"action": "subscribe", "agent_id": "a"}); r.Success {
		t.Error("subscribe without kinds should fail")
	}

	if r := tool.Execute(map[string]any{
		"action": "subscribe", "kinds": []any{"file_change"}, "agent_id": "a",
	}); !r.Success {
		t.Fatalf("subscribe failed: %s", r.Error)
	}
	if n := len(events.Subscriptions("a")); n != 1 {
		t.Fatalf("expected 1 active subscription, got %d", n)
	}

	// Subscribed events now arrive with ordinary tool responses.
	events.Push("file_change", "/x", nil)
	if got := events.DrainFor("a"); len(got) != 1 {
		t.Fatalf("subscribed event should be delivered, got %d", len(got))
	}

	if r := tool.Execute(map[string]any{
		"action": "unsubscribe", "kinds": []any{"file_change"}, "agent_id": "a",
	}); !r.Success {
		t.Fatalf("unsubscribe failed: %s", r.Error)
	}
	events.Push("file_change", "/y", nil)
	if got := events.DrainFor("a"); len(got) != 0 {
		t.Fatalf("nothing should arrive after unsubscribe, got %d", len(got))
	}
}

func TestZeroTimeoutPollsWithoutBlocking(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	// An explicit 0 means "whatever is already there". Treating it as unset
	// would fall back to the default and block for half a minute.
	start := time.Now()
	p := runWait(t, map[string]any{"kinds": []any{"never_happens"}, "timeout": float64(0)})
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("a zero timeout must not block, took %v", elapsed)
	}
	if !p.TimedOut || len(p.Events) != 0 {
		t.Fatalf("expected an empty result, got %+v", p)
	}
	if p.WaitedFor != 0 {
		t.Errorf("waited_for should report 0, got %d", p.WaitedFor)
	}
}

func TestZeroTimeoutStillReturnsQueuedEvents(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	events.Push("file_change", "/tmp/x", map[string]any{"path": "/tmp/x"})

	p := runWait(t, map[string]any{"kinds": []any{"file_change"}, "timeout": float64(0)})
	if len(p.Events) != 1 {
		t.Fatalf("polling should take what is already available, got %+v", p)
	}
}

func TestNegativeTimeoutFallsBackToTheDefault(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	// Nonsense input should not be honoured as a duration.
	events.Push("file_change", "/tmp/x", nil)
	p := runWait(t, map[string]any{"kinds": []any{"file_change"}, "timeout": float64(-5)})
	if p.WaitedFor != defaultWaitTimeoutSeconds {
		t.Errorf("negative timeout should use the default, got %d", p.WaitedFor)
	}
}

func TestWaitEventsRegistered(t *testing.T) {
	if Find("wait_for_events") == nil {
		t.Fatal("wait_for_events should be registered in All()")
	}
}
