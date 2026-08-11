package events

// Tests for the opt-in delivery model (run by `go test ./events`;
// tools/waitevents_test.go covers the tool surface). Nothing is delivered
// without a subscription; a subscription may narrow to specific sources;
// reading does not consume, so several observers each get the event; events
// expire on their TTL and the loss is reported.
// User instruction: "默认是谁都看不到，至于监听了kind 才能看到，然后如果
// source 也制定了，只监听指定的event，然后这个event 有一个最长存活时间，
// 比如10分钟之后就会自动删除".

import (
	"testing"
	"time"
)

func kindsOf(evts []Event) []string {
	out := make([]string, 0, len(evts))
	for _, e := range evts {
		out = append(out, e.Kind)
	}
	return out
}

func sourcesOf(evts []Event) []string {
	out := make([]string, 0, len(evts))
	for _, e := range evts {
		out = append(out, e.Source)
	}
	return out
}

// awaitWaiterRegistered blocks until a WaitFor call has registered.
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

func TestNothingDeliveredWithoutSubscription(t *testing.T) {
	ResetForTest()

	Push("file_change", "/a", nil)
	Push("process_exited", "pid1", nil)

	if got := DrainFor("agent-a"); len(got) != 0 {
		t.Fatalf("delivery must be opt-in, got %v", kindsOf(got))
	}
}

func TestSubscriptionSelectsByKind(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"process_exited"}})

	Push("file_change", "/a", nil)
	Push("process_exited", "pid1", nil)

	got := DrainFor("agent-a")
	if len(got) != 1 || got[0].Kind != "process_exited" {
		t.Fatalf("only the subscribed kind should arrive, got %v", kindsOf(got))
	}
}

func TestSubscriptionCanNarrowToSource(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"process_exited"}, Sources: []string{"pid-b"}})

	Push("process_exited", "pid-a", nil)
	Push("process_exited", "pid-b", nil)

	got := DrainFor("agent-a")
	if len(got) != 1 || got[0].Source != "pid-b" {
		t.Fatalf("source filter should select one event, got %v", sourcesOf(got))
	}
}

func TestDeliveryIsPerAgentAndNotConsumed(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"file_change"}})
	Subscribe("agent-b", Filter{Kinds: []string{"file_change"}})

	Push("file_change", "/shared", nil)

	// Reading does not consume: both observers see it.
	if got := DrainFor("agent-a"); len(got) != 1 {
		t.Fatalf("agent-a should receive it, got %v", kindsOf(got))
	}
	if got := DrainFor("agent-b"); len(got) != 1 {
		t.Fatalf("agent-b should receive it too, got %v", kindsOf(got))
	}
	// But no agent gets the same event twice.
	if got := DrainFor("agent-a"); len(got) != 0 {
		t.Fatalf("second read should be empty, got %v", kindsOf(got))
	}
}

func TestTargetedEventsStayPrivate(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"process_exited"}})
	Subscribe("agent-b", Filter{Kinds: []string{"process_exited"}})

	PushTo("agent-a", "process_exited", "pid1", nil)

	if got := DrainFor("agent-b"); len(got) != 0 {
		t.Fatalf("a targeted event must not reach another agent, got %v", kindsOf(got))
	}
	if got := DrainFor("agent-a"); len(got) != 1 {
		t.Fatalf("the addressed agent should receive it, got %v", kindsOf(got))
	}
}

func TestOwnActionsAreNotEchoedButAreVisibleToOthers(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"tool:web_fetch"}})
	Subscribe("agent-b", Filter{Kinds: []string{"tool:web_fetch"}})

	PushAction("agent-a", "tool:web_fetch", "https://example.com", nil)

	if got := DrainFor("agent-a"); len(got) != 0 {
		t.Fatalf("an agent's own action should not be echoed back, got %v", kindsOf(got))
	}
	if got := DrainFor("agent-b"); len(got) != 1 {
		t.Fatalf("another observer should see it, got %v", kindsOf(got))
	}
}

func TestIncludeOwnOptsIntoSelfEvents(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"tool:shell"}, IncludeOwn: true})

	PushAction("agent-a", "tool:shell", "make", nil)

	if got := DrainFor("agent-a"); len(got) != 1 {
		t.Fatalf("include_own should deliver self-caused events, got %v", kindsOf(got))
	}
}

func TestExplicitWaitNeedsNoSubscription(t *testing.T) {
	ResetForTest()

	done := make(chan []Event, 1)
	go func() {
		done <- WaitFor("agent-a", Filter{Kinds: []string{"process_exited"}, Sources: []string{"pid-b"}}, 5*time.Second)
	}()
	awaitWaiterRegistered(t)

	Push("process_exited", "pid-a", nil) // wrong source: must not wake
	select {
	case got := <-done:
		t.Fatalf("woke on the wrong source: %v", sourcesOf(got))
	case <-time.After(150 * time.Millisecond):
	}

	Push("process_exited", "pid-b", nil)
	got := <-done
	if len(got) != 1 || got[0].Source != "pid-b" {
		t.Fatalf("expected pid-b, got %v", sourcesOf(got))
	}
}

func TestWaitReturnsOwnActionsToo(t *testing.T) {
	ResetForTest()

	// No subscription, and it is the agent's own action: asking explicitly
	// is still enough.
	PushAction("agent-a", "tool:web_search", "golang", nil)
	got := WaitFor("agent-a", Filter{Kinds: []string{"tool:web_search"}}, time.Second)
	if len(got) != 1 {
		t.Fatalf("an explicit wait should return own actions, got %v", kindsOf(got))
	}
}

func TestPiggybackLeavesEventsReservedByOwnWait(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"process_exited", "file_change"}})

	done := make(chan []Event, 1)
	go func() { done <- WaitFor("agent-a", Filter{Kinds: []string{"process_exited"}}, 5*time.Second) }()
	awaitWaiterRegistered(t)

	Push("file_change", "/a", nil)
	Push("process_exited", "pid1", nil)

	// The subscribed-but-unrequested event arrives; the awaited one is held.
	got := DrainFor("agent-a")
	if len(got) != 1 || got[0].Kind != "file_change" {
		t.Fatalf("piggyback should take only the unreserved event, got %v", kindsOf(got))
	}
	if w := <-done; len(w) != 1 || w[0].Kind != "process_exited" {
		t.Fatalf("the waiter should still receive its event, got %v", kindsOf(w))
	}
}

func TestAnotherAgentsWaitDoesNotBlockDelivery(t *testing.T) {
	ResetForTest()
	Subscribe("agent-b", Filter{Kinds: []string{"process_exited"}})

	done := make(chan []Event, 1)
	go func() { done <- WaitFor("agent-a", Filter{Kinds: []string{"process_exited"}}, 3*time.Second) }()
	awaitWaiterRegistered(t)

	Push("process_exited", "pid1", nil)

	// agent-a's wait must not withhold agent-b's subscription delivery.
	if got := DrainFor("agent-b"); len(got) != 1 {
		t.Fatalf("agent-b should still receive it, got %v", kindsOf(got))
	}
	if got := <-done; len(got) != 1 {
		t.Fatalf("agent-a's wait should also resolve, got %v", kindsOf(got))
	}
}

func TestEventsExpireAfterTTL(t *testing.T) {
	ResetForTest()
	restore := SetTTLForTest(80 * time.Millisecond)
	t.Cleanup(restore)

	Subscribe("agent-a", Filter{Kinds: []string{"file_change"}})
	Push("file_change", "/a", nil)

	time.Sleep(200 * time.Millisecond)

	if got := DrainFor("agent-a"); len(got) != 0 {
		t.Fatalf("expired events must not be delivered, got %v", kindsOf(got))
	}
	mu.Lock()
	remaining := len(log)
	mu.Unlock()
	if remaining != 0 {
		t.Errorf("expired event should be removed from the log, %d left", remaining)
	}
	// The loss is reported rather than hidden.
	if n := TakeMissed("agent-a"); n != 1 {
		t.Errorf("expected 1 missed event reported, got %d", n)
	}
}

func TestFreshEventsSurviveGC(t *testing.T) {
	ResetForTest()
	restore := SetTTLForTest(5 * time.Second)
	t.Cleanup(restore)

	Subscribe("agent-a", Filter{Kinds: []string{"file_change"}})
	Push("file_change", "/old", nil)
	time.Sleep(30 * time.Millisecond)
	Push("file_change", "/new", nil)

	if got := DrainFor("agent-a"); len(got) != 2 {
		t.Fatalf("both fresh events should be delivered, got %v", sourcesOf(got))
	}
	if n := TakeMissed("agent-a"); n != 0 {
		t.Errorf("nothing should be reported missed, got %d", n)
	}
}

func TestUnsubscribeStopsDelivery(t *testing.T) {
	ResetForTest()
	Subscribe("agent-a", Filter{Kinds: []string{"file_change"}})
	if n := len(Subscriptions("agent-a")); n != 1 {
		t.Fatalf("expected 1 subscription, got %d", n)
	}

	if removed := Unsubscribe("agent-a", []string{"file_change"}); removed != 1 {
		t.Errorf("expected 1 removed, got %d", removed)
	}
	Push("file_change", "/a", nil)
	if got := DrainFor("agent-a"); len(got) != 0 {
		t.Fatalf("no delivery after unsubscribe, got %v", kindsOf(got))
	}
}

func TestBudgetTrimReportsMissed(t *testing.T) {
	ResetForTest()
	restore := SetQueueBoundsForTest(3000, 1000) // ~2 KiB payloads fit twice
	t.Cleanup(restore)
	Subscribe("agent-a", Filter{Kinds: []string{"file_change"}})

	big := string(make([]byte, 1024))
	for i := 0; i < 6; i++ {
		Push("file_change", "/f", map[string]any{"content": big})
	}

	mu.Lock()
	remaining := len(log)
	mu.Unlock()
	if remaining >= 6 {
		t.Fatalf("byte budget should have trimmed the log, %d still held", remaining)
	}
	DrainFor("agent-a")
	if n := TakeMissed("agent-a"); n == 0 {
		t.Error("trimmed-away events must be reported as missed")
	}
}

func TestSmallEventsAreNotTrimmed(t *testing.T) {
	ResetForTest()
	restore := SetQueueBoundsForTest(1<<20, 1000)
	t.Cleanup(restore)
	Subscribe("agent-a", Filter{Kinds: []string{"process_exited"}})

	for i := 0; i < 200; i++ {
		Push("process_exited", "pid", map[string]any{"exit_code": 0})
	}
	if got := DrainFor("agent-a"); len(got) != 200 {
		t.Errorf("all small events should fit, got %d/200", len(got))
	}
	if n := TakeMissed("agent-a"); n != 0 {
		t.Errorf("nothing should be missed, got %d", n)
	}
}

func TestEnvIntParsing(t *testing.T) {
	t.Setenv("ASH_TEST_EVENT_BOUND", "4096")
	if got := envInt("ASH_TEST_EVENT_BOUND", 7); got != 4096 {
		t.Errorf("env override = %d, want 4096", got)
	}
	t.Setenv("ASH_TEST_EVENT_BOUND", "not-a-number")
	if got := envInt("ASH_TEST_EVENT_BOUND", 7); got != 7 {
		t.Errorf("invalid value should fall back, got %d", got)
	}
	t.Setenv("ASH_TEST_EVENT_BOUND", "-5")
	if got := envInt("ASH_TEST_EVENT_BOUND", 7); got != 7 {
		t.Errorf("non-positive should fall back, got %d", got)
	}
	if got := envInt("ASH_TEST_EVENT_BOUND_UNSET", 42); got != 42 {
		t.Errorf("unset should use default, got %d", got)
	}
}
