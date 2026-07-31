package tools

// Tests for waitevents.go / events.WaitFor (run by `go test ./tools`; no
// other file covers blocking event delivery). Verifies: already-queued
// events return immediately, a wait wakes on a concurrent push, timeout is
// reported rather than hanging, non-matching kinds stay queued for the
// piggyback path, and queue overflow is surfaced as a dropped count.
// User instruction: "我觉得可以搞一个 wait_for_events 这个东西".

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/dreamyang-liu/ash/runtime/events"
)

// maxQueueLenForTest mirrors events.maxQueueLen (unexported there).
const maxQueueLenForTest = 100

type waitPayload struct {
	Events []struct {
		Kind    string `json:"kind"`
		Source  string `json:"source"`
		AgentID string `json:"agent_id"`
	} `json:"events"`
	TimedOut  bool `json:"timed_out"`
	WaitedFor int  `json:"waited_for"`
	Dropped   int  `json:"dropped"`
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

// drainAll clears queues so tests don't leak events into each other.
func drainAll(t *testing.T) {
	t.Helper()
	events.Drain()
	events.DrainFor("agent-a")
	events.DrainFor("agent-b")
	events.TakeDropped("")
	events.TakeDropped("agent-a")
	events.TakeDropped("agent-b")
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
	// The unrequested event must survive for the piggyback path.
	remaining := events.Drain()
	if len(remaining) != 1 || remaining[0].Kind != "other_kind" {
		t.Fatalf("non-matching event was consumed: %+v", remaining)
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

func TestWaitForEventsReportsDropped(t *testing.T) {
	drainAll(t)
	t.Cleanup(func() { drainAll(t) })

	// Overflow the queue: 5 more than the cap.
	for i := 0; i < maxQueueLenForTest+5; i++ {
		events.Push("flood", "src", map[string]any{"i": i})
	}
	p := runWait(t, map[string]any{"kinds": []any{"flood"}, "timeout": float64(2)})
	if p.Dropped != 5 {
		t.Errorf("expected 5 dropped events reported, got %d", p.Dropped)
	}
	if len(p.Events) != maxQueueLenForTest {
		t.Errorf("expected %d retained events, got %d", maxQueueLenForTest, len(p.Events))
	}
}

func TestWaitEventsRegistered(t *testing.T) {
	if Find("wait_for_events") == nil {
		t.Fatal("wait_for_events should be registered in All()")
	}
}
