package events

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type Event struct {
	ID        string         `json:"id"`
	Kind      string         `json:"kind"`
	Source    string         `json:"source"`
	AgentID   string         `json:"agent_id,omitempty"`
	Data      map[string]any `json:"data"`
	Timestamp string         `json:"timestamp"`
}

// Queue bounds. Event payloads are delivered in full — truncating them is a
// consumer-side policy decision (an SDK or post-processing pipeline may want
// the complete data, while a harness renders a shortened form into the LLM
// context). The runtime only enforces what it must to stay alive: this queue
// lives inside the sandbox's runtime process, and letting it grow without
// limit would OOM-kill the runtime, taking the whole sandbox — and the
// rollout — with it. That loss is unrecoverable, unlike a truncated payload.
//
// The budget is byte-based rather than count-based because payload sizes vary
// by orders of magnitude (a pid versus a file's full contents); a fixed event
// count is a meaningless bound when one event can be megabytes.
//
// Override with ASH_EVENT_QUEUE_BYTES / ASH_EVENT_QUEUE_MAX_EVENTS.
const (
	defaultQueueBytes = 8 << 20 // 8 MiB per queue
	defaultQueueLen   = 1000    // backstop against many tiny events
)

var (
	maxQueueBytes = envInt("ASH_EVENT_QUEUE_BYTES", defaultQueueBytes)
	maxQueueLen   = envInt("ASH_EVENT_QUEUE_MAX_EVENTS", defaultQueueLen)
)

// envInt reads a positive integer from the environment, falling back to def.
func envInt(name string, def int) int {
	if raw := os.Getenv(name); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// approxSize estimates an event's memory footprint for the byte budget. It is
// deliberately cheap: exact accounting would mean serializing every event.
func approxSize(evt Event) int {
	n := len(evt.ID) + len(evt.Kind) + len(evt.Source) + len(evt.AgentID) + len(evt.Timestamp)
	for k, v := range evt.Data {
		n += len(k)
		switch tv := v.(type) {
		case string:
			n += len(tv)
		case []byte:
			n += len(tv)
		default:
			n += 8 // numbers, bools, pointers to nested values
		}
	}
	return n
}

// waiter records a blocked WaitFor call. It stays registered for the whole
// wait so that piggyback delivery (DrainFor) can tell which kinds are
// reserved and must be left queued.
type waiter struct {
	agentID string
	kinds   []string      // empty = any kind
	sources []string      // empty = any source (pid, path, ...)
	signal  chan struct{} // buffered(1): coalesces wakeups
}

var (
	mu        sync.Mutex
	queues    = make(map[string][]Event) // per-agent queues, "" = broadcast
	dropped   = make(map[string]int)     // per-queue overflow counts
	waiters   = make(map[uint64]*waiter) // blocked WaitFor calls
	waiterSeq atomic.Uint64
	counter   atomic.Uint64
)

// notifyWaitersLocked nudges every blocked WaitFor caller to re-check the
// queues. Sends are non-blocking: a pending signal already means "re-check".
// Must hold mu.
func notifyWaitersLocked() {
	for _, w := range waiters {
		select {
		case w.signal <- struct{}{}:
		default:
		}
	}
}

// registerWaiterLocked records a blocked waiter. Must hold mu.
func registerWaiterLocked(agentID string, kinds, sources []string) (uint64, chan struct{}) {
	id := waiterSeq.Add(1)
	w := &waiter{
		agentID: agentID,
		kinds:   kinds,
		sources: sources,
		signal:  make(chan struct{}, 1),
	}
	waiters[id] = w
	return id, w.signal
}

func unregisterWaiter(id uint64) {
	mu.Lock()
	delete(waiters, id)
	mu.Unlock()
}

// reservedLocked reports whether some blocked waiter is waiting for evt.
// Such events must survive piggyback delivery: consuming them there would
// strand the waiter until its timeout even though its event did occur.
// Must hold mu.
func reservedLocked(evt Event) bool {
	for _, w := range waiters {
		// A waiter receives broadcasts plus its own agent's events.
		if evt.AgentID != "" && evt.AgentID != w.agentID {
			continue
		}
		if matches(evt, w.kinds) && matchesSource(evt, w.sources) {
			return true
		}
	}
	return false
}

// Push adds an event. If agentID is empty, event goes to all agents.
func Push(kind, source string, data map[string]any) {
	PushTo("", kind, source, data)
}

// PushTo adds an event targeted at a specific agent. Empty agentID = broadcast.
func PushTo(agentID, kind, source string, data map[string]any) {
	mu.Lock()
	defer mu.Unlock()

	id := counter.Add(1)
	evt := Event{
		ID:        fmt.Sprintf("evt_%d", id),
		Kind:      kind,
		Source:    source,
		AgentID:   agentID,
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	queues[agentID] = append(queues[agentID], evt)
	trimQueueLocked(agentID)
	notifyWaitersLocked()
}

// trimQueueLocked drops oldest events until the queue fits both the byte
// budget and the event-count backstop, counting the loss so it can be
// reported instead of silently vanishing. Must hold mu.
func trimQueueLocked(agentID string) {
	q := queues[agentID]

	total := 0
	for _, evt := range q {
		total += approxSize(evt)
	}

	cut := 0
	for cut < len(q)-1 && (total > maxQueueBytes || len(q)-cut > maxQueueLen) {
		total -= approxSize(q[cut])
		cut++
	}
	// Keep at least the newest event even if it alone exceeds the budget:
	// dropping it would lose the most relevant information.
	if cut > 0 {
		queues[agentID] = q[cut:]
		dropped[agentID] += cut
	}
}

// Drain returns pending events for a specific agent (+ broadcasts), clears them.
func Drain() []Event {
	return DrainFor("")
}

// DrainFor returns events targeted at agentID plus any broadcast events,
// for piggyback delivery on a tool response.
//
// Events that a blocked WaitFor caller is waiting for are left queued: the
// waiter asked for them explicitly, and handing them to an unrelated tool
// response would strand that waiter until its timeout. Everything else is
// delivered opportunistically, so environment changes reach the agent
// without it having to ask.
func DrainFor(agentID string) []Event {
	mu.Lock()
	defer mu.Unlock()

	sources := []string{""} // broadcast queue
	if agentID != "" {
		sources = append(sources, agentID)
	}

	var result []Event
	for _, q := range sources {
		pending := queues[q]
		if len(pending) == 0 {
			continue
		}
		var reserved []Event
		for _, evt := range pending {
			if reservedLocked(evt) {
				reserved = append(reserved, evt)
			} else {
				result = append(result, evt)
			}
		}
		queues[q] = reserved
	}

	if result == nil {
		return []Event{}
	}
	return result
}

// TakeDropped returns and clears the overflow count for an agent's queues
// (its own plus broadcast), so loss can be surfaced instead of hidden.
func TakeDropped(agentID string) int {
	mu.Lock()
	defer mu.Unlock()
	n := dropped[""]
	delete(dropped, "")
	if agentID != "" {
		n += dropped[agentID]
		delete(dropped, agentID)
	}
	return n
}

// matches reports whether evt is of one of the requested kinds. An empty
// kinds slice matches everything.
func matches(evt Event, kinds []string) bool {
	if len(kinds) == 0 {
		return true
	}
	for _, k := range kinds {
		if evt.Kind == k {
			return true
		}
	}
	return false
}

// matchesSource reports whether evt came from one of the requested sources.
// An empty sources slice matches everything. Source is the handle the
// emitter used: a pid for process_exited, a path for file_change.
func matchesSource(evt Event, sources []string) bool {
	if len(sources) == 0 {
		return true
	}
	for _, s := range sources {
		if evt.Source == s {
			return true
		}
	}
	return false
}

// drainMatchingLocked removes and returns events for agentID (plus
// broadcasts) whose Kind is in kinds, leaving non-matching events queued so
// a later call still delivers them. Must hold mu.
func drainMatchingLocked(agentID string, kinds, sources []string) []Event {
	queueKeys := []string{""} // broadcast queue
	if agentID != "" {
		queueKeys = append(queueKeys, agentID)
	}

	var taken []Event
	for _, q := range queueKeys {
		pending := queues[q]
		if len(pending) == 0 {
			continue
		}
		var kept []Event
		for _, evt := range pending {
			if matches(evt, kinds) && matchesSource(evt, sources) {
				taken = append(taken, evt)
			} else {
				kept = append(kept, evt)
			}
		}
		queues[q] = kept
	}
	return taken
}

// WaitFor blocks until at least one matching event is available for agentID
// (broadcast events included), or until timeout elapses. It is a long-poll:
// the sandbox never initiates a connection, so the transport stays a plain
// one-way request/response protocol.
//
// Matching is the conjunction of two filters, each matching everything when
// empty:
//   - kinds:   event type, e.g. "process_exited"
//   - sources: the emitter's handle, e.g. a specific pid or file path
//
// So waiting on one particular background process is
// WaitFor(agent, []string{"process_exited"}, []string{pid}, timeout).
//
// Only matching events are consumed; anything else stays queued and rides
// along with a later tool response.
func WaitFor(agentID string, kinds, sources []string, timeout time.Duration) []Event {
	deadline := time.Now().Add(timeout)

	// Register before the first check and stay registered for the whole
	// wait: this both guarantees no wakeup is missed and reserves the
	// requested events against piggyback delivery (see reservedLocked).
	mu.Lock()
	id, signal := registerWaiterLocked(agentID, kinds, sources)
	mu.Unlock()
	defer unregisterWaiter(id)

	for {
		mu.Lock()
		taken := drainMatchingLocked(agentID, kinds, sources)
		mu.Unlock()
		if len(taken) > 0 {
			return taken
		}

		remaining := time.Until(deadline)
		if remaining <= 0 {
			return []Event{}
		}
		timer := time.NewTimer(remaining)
		select {
		case <-signal:
			timer.Stop() // something arrived: re-check the queues
		case <-timer.C:
			return []Event{}
		}
	}
}

// SetQueueBoundsForTest overrides the queue bounds and returns a function
// restoring the previous values. Exported for tests in sibling packages.
func SetQueueBoundsForTest(bytes, count int) func() {
	mu.Lock()
	prevBytes, prevCount := maxQueueBytes, maxQueueLen
	maxQueueBytes, maxQueueLen = bytes, count
	mu.Unlock()
	return func() {
		mu.Lock()
		maxQueueBytes, maxQueueLen = prevBytes, prevCount
		mu.Unlock()
	}
}
