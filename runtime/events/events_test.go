package events

// Tests for the reservation contract between the two consumption paths of the
// event queue (run by `go test ./events`; tools/waitevents_test.go covers the
// tool-level surface). Piggyback (DrainFor) must not consume events that a
// blocked WaitFor caller is waiting for; everything else it still delivers
// opportunistically.
// User instruction: "piggyback 每次拿的时候要看一下是否有wait for 这个event".

import (
	"sync"
	"testing"
	"time"
)

func reset() {
	mu.Lock()
	queues = make(map[string][]Event)
	dropped = make(map[string]int)
	waiters = make(map[uint64]*waiter)
	mu.Unlock()
}

func kindsOf(evts []Event) []string {
	out := make([]string, 0, len(evts))
	for _, e := range evts {
		out = append(out, e.Kind)
	}
	return out
}

// awaitWaiterRegistered blocks until at least one WaitFor call has registered.
func awaitWaiterRegistered(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		mu.Lock()
		n := len(waiters)
		mu.Unlock()
		if n > 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("waiter never registered")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestPiggybackTakesEverythingWhenNobodyWaits(t *testing.T) {
	reset()
	Push("file_change", "/a", nil)
	Push("process_exited", "pid1", nil)

	got := kindsOf(DrainFor(""))
	if len(got) != 2 {
		t.Fatalf("piggyback should deliver both events, got %v", got)
	}
	if len(DrainFor("")) != 0 {
		t.Fatal("queue should be empty after drain")
	}
}

func TestPiggybackLeavesReservedKindForWaiter(t *testing.T) {
	reset()

	var (
		wg     sync.WaitGroup
		waited []Event
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		waited = WaitFor("", []string{"process_exited"}, nil, 5*time.Second)
	}()
	awaitWaiterRegistered(t)

	// Both events arrive; a tool call piggybacks in between.
	Push("file_change", "/a", nil)
	Push("process_exited", "pid1", nil)

	got := kindsOf(DrainFor(""))
	if len(got) != 1 || got[0] != "file_change" {
		t.Fatalf("piggyback must take only the unreserved event, got %v", got)
	}

	wg.Wait()
	if len(waited) != 1 || waited[0].Kind != "process_exited" {
		t.Fatalf("waiter should have received its reserved event, got %v", kindsOf(waited))
	}
}

func TestPiggybackRespectsAnyKindWaiter(t *testing.T) {
	reset()

	done := make(chan []Event, 1)
	go func() { done <- WaitFor("", nil, nil, 5*time.Second) }() // nil kinds = any
	awaitWaiterRegistered(t)

	Push("anything", "src", nil)
	// An any-kind waiter reserves everything, so piggyback gets nothing.
	if got := DrainFor(""); len(got) != 0 {
		t.Fatalf("piggyback should be empty while an any-kind waiter blocks, got %v", kindsOf(got))
	}
	if got := <-done; len(got) != 1 {
		t.Fatalf("any-kind waiter should receive the event, got %v", kindsOf(got))
	}
}

func TestReservationEndsWhenWaiterReturns(t *testing.T) {
	reset()

	// Waiter times out quickly and unregisters.
	if got := WaitFor("", []string{"process_exited"}, nil, 50*time.Millisecond); len(got) != 0 {
		t.Fatalf("expected timeout, got %v", kindsOf(got))
	}
	mu.Lock()
	n := len(waiters)
	mu.Unlock()
	if n != 0 {
		t.Fatalf("waiter should be unregistered after returning, %d left", n)
	}

	// With no waiter, piggyback may take the event again.
	Push("process_exited", "pid1", nil)
	if got := kindsOf(DrainFor("")); len(got) != 1 || got[0] != "process_exited" {
		t.Fatalf("piggyback should take the event once nobody waits, got %v", got)
	}
}

func TestByteBudgetTrimsLargePayloads(t *testing.T) {
	reset()
	// Budget fits roughly two 1KiB payloads; the count backstop is generous
	// so only the byte budget can trigger trimming.
	restore := SetQueueBoundsForTest(3000, 1000)
	t.Cleanup(restore)

	big := string(make([]byte, 1024))
	for i := 0; i < 6; i++ {
		Push("file_change", "/f", map[string]any{"content": big})
	}

	mu.Lock()
	queued := len(queues[""])
	mu.Unlock()
	if queued >= 6 {
		t.Fatalf("byte budget should have trimmed large payloads, %d still queued", queued)
	}
	if n := TakeDropped(""); n == 0 {
		t.Error("trimming must be reported as dropped, got 0")
	}
}

func TestSmallEventsAreNotTrimmedByByteBudget(t *testing.T) {
	reset()
	restore := SetQueueBoundsForTest(1<<20, 1000) // 1MiB, plenty for tiny events
	t.Cleanup(restore)

	for i := 0; i < 200; i++ {
		Push("process_exited", "pid", map[string]any{"exit_code": 0})
	}
	mu.Lock()
	queued := len(queues[""])
	mu.Unlock()
	if queued != 200 {
		t.Errorf("small events should all fit the byte budget, got %d/200", queued)
	}
	if n := TakeDropped(""); n != 0 {
		t.Errorf("nothing should have been dropped, got %d", n)
	}
}

func TestNewestEventSurvivesEvenIfOversized(t *testing.T) {
	reset()
	restore := SetQueueBoundsForTest(100, 1000) // smaller than one payload
	t.Cleanup(restore)

	Push("file_change", "/f", map[string]any{"content": string(make([]byte, 4096))})
	mu.Lock()
	queued := len(queues[""])
	mu.Unlock()
	if queued != 1 {
		t.Errorf("the newest event must be kept even when oversized, got %d", queued)
	}
}

func TestEnvIntParsing(t *testing.T) {
	t.Setenv("ASH_TEST_EVENT_BOUND", "4096")
	if got := envInt("ASH_TEST_EVENT_BOUND", 7); got != 4096 {
		t.Errorf("env override = %d, want 4096", got)
	}
	t.Setenv("ASH_TEST_EVENT_BOUND", "not-a-number")
	if got := envInt("ASH_TEST_EVENT_BOUND", 7); got != 7 {
		t.Errorf("invalid value should fall back to default, got %d", got)
	}
	t.Setenv("ASH_TEST_EVENT_BOUND", "-5")
	if got := envInt("ASH_TEST_EVENT_BOUND", 7); got != 7 {
		t.Errorf("non-positive value should fall back to default, got %d", got)
	}
	if got := envInt("ASH_TEST_EVENT_BOUND_UNSET", 42); got != 42 {
		t.Errorf("unset should use default, got %d", got)
	}
}

func TestWaitForSpecificSource(t *testing.T) {
	reset()

	// Two background processes; we only care about pid-b.
	done := make(chan []Event, 1)
	go func() {
		done <- WaitFor("", []string{"process_exited"}, []string{"pid-b"}, 5*time.Second)
	}()
	awaitWaiterRegistered(t)

	// The uninteresting process exits first: must NOT wake the waiter, and
	// must stay available to piggyback (nobody reserved it).
	Push("process_exited", "pid-a", nil)
	select {
	case got := <-done:
		t.Fatalf("waiter woke on the wrong source: %+v", got)
	case <-time.After(150 * time.Millisecond):
	}
	if got := DrainFor(""); len(got) != 1 || got[0].Source != "pid-a" {
		t.Fatalf("unreserved source should piggyback, got %+v", got)
	}

	// The awaited process exits: waiter gets exactly that one.
	Push("process_exited", "pid-b", nil)
	got := <-done
	if len(got) != 1 || got[0].Source != "pid-b" {
		t.Fatalf("expected only pid-b, got %+v", got)
	}
}

func TestSourceReservationIsPreciseAgainstPiggyback(t *testing.T) {
	reset()

	done := make(chan []Event, 1)
	go func() {
		done <- WaitFor("", []string{"process_exited"}, []string{"pid-b"}, 5*time.Second)
	}()
	awaitWaiterRegistered(t)

	// Same kind, different sources, queued together.
	Push("process_exited", "pid-a", nil)
	Push("process_exited", "pid-b", nil)

	// Piggyback may take pid-a but must leave pid-b for its waiter.
	got := DrainFor("")
	if len(got) != 1 || got[0].Source != "pid-a" {
		t.Fatalf("piggyback should take only pid-a, got %+v", got)
	}
	if w := <-done; len(w) != 1 || w[0].Source != "pid-b" {
		t.Fatalf("waiter should still receive pid-b, got %+v", w)
	}
}

func TestReservationIsPerAgent(t *testing.T) {
	reset()

	done := make(chan []Event, 1)
	go func() { done <- WaitFor("agent-a", []string{"process_exited"}, nil, 5*time.Second) }()
	awaitWaiterRegistered(t)

	// An event for a *different* agent is not reserved by agent-a's waiter.
	PushTo("agent-b", "process_exited", "pid-b", nil)
	if got := kindsOf(DrainFor("agent-b")); len(got) != 1 {
		t.Fatalf("agent-b's event should not be reserved by agent-a's waiter, got %v", got)
	}

	// agent-a's own event is reserved and reaches the waiter.
	PushTo("agent-a", "process_exited", "pid-a", nil)
	if got := DrainFor("agent-a"); len(got) != 0 {
		t.Fatalf("agent-a's event should be reserved, got %v", kindsOf(got))
	}
	if got := <-done; len(got) != 1 || got[0].Source != "pid-a" {
		t.Fatalf("waiter should get its own agent's event, got %+v", got)
	}
}
