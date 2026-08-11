// Package events is the sandbox's channel for asynchronous facts: a
// background process exited, a file changed, some agent ran a tool.
//
// Delivery is opt-in. Nothing reaches an agent unless that agent subscribed
// to a kind (and optionally narrowed it to specific sources), because in a
// sandbox shared by several actors most events are irrelevant to any given
// one, and unrequested notifications are pure context noise.
//
// Events live in one append-only log with a time-to-live, not in per-consumer
// queues. Reading does not consume: each agent tracks what it has already
// been given, so two observers subscribed to the same kind both see it. An
// event leaves the log when its TTL expires (or when the log has to be
// trimmed to stay within its memory budget), never because someone else read
// it first.
//
// Callers: runtime/toolevents.go (tool-call events + piggyback delivery),
// runtime/tools/shell.go (process_exited), runtime/tools/edit.go
// (file_change), runtime/tools/waitevents.go (the events tool).
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
	ID     string `json:"id"`
	Kind   string `json:"kind"`
	Source string `json:"source"`
	// AgentID restricts the audience: when set, only that agent may receive
	// the event (a background process's exit concerns whoever started it).
	AgentID string `json:"agent_id,omitempty"`
	// Origin names the agent whose action produced the event. Piggyback
	// delivery skips an event for its own originator -- its call result
	// already reported the outcome -- unless the subscription asks for own
	// actions. An explicit wait always returns it.
	Origin    string         `json:"origin,omitempty"`
	Data      map[string]any `json:"data"`
	Timestamp string         `json:"timestamp"`

	seq    uint64
	expiry time.Time
}

// Filter selects events by kind and, optionally, by source handle. An empty
// list matches everything, so {} matches all events and
// {Kinds: ["process_exited"], Sources: ["pid7"]} matches exactly one process.
type Filter struct {
	Kinds   []string `json:"kinds,omitempty"`
	Sources []string `json:"sources,omitempty"`
	// IncludeOwn also delivers events the subscribing agent itself caused.
	IncludeOwn bool `json:"include_own,omitempty"`
}

func (f Filter) matches(evt Event) bool {
	return matchesAny(evt.Kind, f.Kinds) && matchesAny(evt.Source, f.Sources)
}

func matchesAny(value string, allowed []string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, a := range allowed {
		if value == a {
			return true
		}
	}
	return false
}

// Bounds. The log lives inside the sandbox's runtime process: unbounded
// growth would OOM-kill it and take the whole sandbox -- and its rollout --
// with it, which nothing downstream can recover. Payloads themselves are
// delivered in full, since shortening them is a consumer-side decision.
//
// The byte budget is byte-based rather than count-based because payload sizes
// span orders of magnitude (a pid versus a file's contents). TTL is the
// primary reclaim mechanism; the budget is a backstop for bursts.
const (
	defaultQueueBytes = 8 << 20 // 8 MiB
	defaultQueueLen   = 1000    // backstop against many tiny events
	defaultTTLSeconds = 600     // 10 minutes
)

var (
	maxQueueBytes = envInt("ASH_EVENT_QUEUE_BYTES", defaultQueueBytes)
	maxQueueLen   = envInt("ASH_EVENT_QUEUE_MAX_EVENTS", defaultQueueLen)
	eventTTL      = time.Duration(envInt("ASH_EVENT_TTL_SECONDS", defaultTTLSeconds)) * time.Second
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

// approxSize estimates an event's memory footprint. Deliberately cheap:
// exact accounting would mean serializing every event.
func approxSize(evt Event) int {
	n := len(evt.ID) + len(evt.Kind) + len(evt.Source) + len(evt.AgentID) +
		len(evt.Origin) + len(evt.Timestamp)
	for k, v := range evt.Data {
		n += len(k)
		switch tv := v.(type) {
		case string:
			n += len(tv)
		case []byte:
			n += len(tv)
		default:
			n += 8
		}
	}
	return n
}

// waiter records a blocked WaitFor call. It stays registered for the whole
// wait so piggyback delivery can tell which events are reserved.
type waiter struct {
	agentID string
	filter  Filter
	signal  chan struct{} // buffered(1): coalesces wakeups
}

var (
	mu  sync.Mutex
	log []Event // append-only, TTL- and budget-bounded

	subs      = map[string][]Filter{}        // agent -> subscriptions
	delivered = map[string]map[string]bool{} // agent -> event id -> given
	missed    = map[string]int{}             // agent -> matching events lost before delivery
	waiters   = map[uint64]*waiter{}
	waiterSeq atomic.Uint64
	counter   atomic.Uint64
)

// --- subscriptions ---

// Subscribe adds a delivery filter for an agent. Without one, piggyback
// delivery gives that agent nothing.
func Subscribe(agentID string, f Filter) {
	mu.Lock()
	defer mu.Unlock()
	subs[agentID] = append(subs[agentID], f)
}

// Unsubscribe removes filters mentioning any of kinds, or all of the agent's
// filters when kinds is empty.
func Unsubscribe(agentID string, kinds []string) int {
	mu.Lock()
	defer mu.Unlock()
	if len(kinds) == 0 {
		n := len(subs[agentID])
		delete(subs, agentID)
		return n
	}
	var kept []Filter
	removed := 0
	for _, f := range subs[agentID] {
		if mentionsAny(f.Kinds, kinds) {
			removed++
			continue
		}
		kept = append(kept, f)
	}
	if len(kept) == 0 {
		delete(subs, agentID)
	} else {
		subs[agentID] = kept
	}
	return removed
}

func mentionsAny(have, wanted []string) bool {
	for _, h := range have {
		for _, w := range wanted {
			if h == w {
				return true
			}
		}
	}
	return false
}

// Subscriptions returns an agent's current filters.
func Subscriptions(agentID string) []Filter {
	mu.Lock()
	defer mu.Unlock()
	out := make([]Filter, len(subs[agentID]))
	copy(out, subs[agentID])
	return out
}

// --- producing ---

// Push records an external fact visible to any subscriber.
func Push(kind, source string, data map[string]any) {
	add(Event{Kind: kind, Source: source, Data: data})
}

// PushTo records a fact addressed to one agent; others never receive it.
func PushTo(agentID, kind, source string, data map[string]any) {
	add(Event{AgentID: agentID, Kind: kind, Source: source, Data: data})
}

// PushAction records an action performed by origin, visible to other
// subscribers but not echoed back to the actor itself.
func PushAction(origin, kind, source string, data map[string]any) {
	add(Event{Origin: origin, Kind: kind, Source: source, Data: data})
}

func add(evt Event) {
	mu.Lock()
	defer mu.Unlock()

	now := time.Now()
	evt.seq = counter.Add(1)
	evt.ID = fmt.Sprintf("evt_%d", evt.seq)
	evt.Timestamp = now.UTC().Format(time.RFC3339)
	evt.expiry = now.Add(eventTTL)

	log = append(log, evt)
	gcLocked(now)
	notifyWaitersLocked()
}

// --- consuming ---

// DrainFor returns the events an agent is subscribed to and has not been
// given yet, for piggyback delivery on a tool response.
//
// Events a blocked WaitFor of the same agent is waiting for are left alone:
// it asked for them explicitly, and handing them to an unrelated response
// would strand that wait until its timeout.
func DrainFor(agentID string) []Event {
	mu.Lock()
	defer mu.Unlock()
	gcLocked(time.Now())

	filters := subs[agentID]
	if len(filters) == 0 {
		return []Event{} // opt-in: no subscription, no delivery
	}

	var out []Event
	for _, evt := range log {
		if !audienceAllows(evt, agentID) || wasDelivered(agentID, evt.ID) {
			continue
		}
		f, ok := matchingFilter(filters, evt)
		if !ok {
			continue
		}
		if evt.Origin != "" && evt.Origin == agentID && !f.IncludeOwn {
			continue // the caller's own action: its result already said so
		}
		if reservedLocked(evt, agentID) {
			continue
		}
		markDelivered(agentID, evt.ID)
		out = append(out, evt)
	}
	if out == nil {
		return []Event{}
	}
	return out
}

// Drain is DrainFor with no agent identity.
func Drain() []Event { return DrainFor("") }

// WaitFor blocks until an event matching filter is available for agentID, or
// until timeout. It is a long poll: the sandbox never initiates a connection,
// so the transport stays a plain one-way request/response protocol.
//
// A wait is independent of subscriptions -- asking explicitly is enough --
// and returns own actions too, since the caller asked for them by name.
func WaitFor(agentID string, filter Filter, timeout time.Duration) []Event {
	deadline := time.Now().Add(timeout)

	// Register before the first check and stay registered for the whole wait:
	// no wakeup can be missed, and piggyback delivery leaves these events be.
	mu.Lock()
	id, signal := registerWaiterLocked(agentID, filter)
	mu.Unlock()
	defer unregisterWaiter(id)

	for {
		if got := takeMatching(agentID, filter); len(got) > 0 {
			return got
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return []Event{}
		}
		timer := time.NewTimer(remaining)
		select {
		case <-signal:
			timer.Stop()
		case <-timer.C:
			return []Event{}
		}
	}
}

func takeMatching(agentID string, filter Filter) []Event {
	mu.Lock()
	defer mu.Unlock()
	gcLocked(time.Now())

	var out []Event
	for _, evt := range log {
		if !audienceAllows(evt, agentID) || wasDelivered(agentID, evt.ID) {
			continue
		}
		if !filter.matches(evt) {
			continue
		}
		markDelivered(agentID, evt.ID)
		out = append(out, evt)
	}
	return out
}

// TakeMissed returns and clears the number of events that matched an agent's
// subscriptions but left the log before it received them (TTL expiry or a
// budget trim), so loss is reported instead of hidden.
func TakeMissed(agentID string) int {
	mu.Lock()
	defer mu.Unlock()
	n := missed[agentID]
	delete(missed, agentID)
	return n
}

// --- internals ---

func audienceAllows(evt Event, agentID string) bool {
	return evt.AgentID == "" || evt.AgentID == agentID
}

func matchingFilter(filters []Filter, evt Event) (Filter, bool) {
	for _, f := range filters {
		if f.matches(evt) {
			return f, true
		}
	}
	return Filter{}, false
}

func wasDelivered(agentID, eventID string) bool {
	return delivered[agentID][eventID]
}

func markDelivered(agentID, eventID string) {
	if delivered[agentID] == nil {
		delivered[agentID] = map[string]bool{}
	}
	delivered[agentID][eventID] = true
}

// gcLocked drops expired events, then trims the oldest until the log fits its
// budget. Anything a subscriber would have wanted but never got is counted as
// missed for that subscriber. Must hold mu.
func gcLocked(now time.Time) {
	cut := 0
	for cut < len(log) && now.After(log[cut].expiry) {
		cut++
	}

	total := 0
	for _, evt := range log[cut:] {
		total += approxSize(evt)
	}
	for cut < len(log)-1 && (total > maxQueueBytes || len(log)-cut > maxQueueLen) {
		total -= approxSize(log[cut])
		cut++
	}
	if cut == 0 {
		return
	}

	for _, evt := range log[:cut] {
		accountMissedLocked(evt)
		for agentID := range delivered {
			delete(delivered[agentID], evt.ID)
		}
	}
	log = append([]Event(nil), log[cut:]...)
}

// accountMissedLocked credits a lost event to every subscriber that matched
// it and never received it. Must hold mu.
func accountMissedLocked(evt Event) {
	for agentID, filters := range subs {
		if !audienceAllows(evt, agentID) || wasDelivered(agentID, evt.ID) {
			continue
		}
		f, ok := matchingFilter(filters, evt)
		if !ok {
			continue
		}
		if evt.Origin != "" && evt.Origin == agentID && !f.IncludeOwn {
			continue
		}
		missed[agentID]++
	}
}

func notifyWaitersLocked() {
	for _, w := range waiters {
		select {
		case w.signal <- struct{}{}:
		default:
		}
	}
}

func registerWaiterLocked(agentID string, filter Filter) (uint64, chan struct{}) {
	id := waiterSeq.Add(1)
	waiters[id] = &waiter{agentID: agentID, filter: filter, signal: make(chan struct{}, 1)}
	return id, waiters[id].signal
}

func unregisterWaiter(id uint64) {
	mu.Lock()
	delete(waiters, id)
	mu.Unlock()
}

// reservedLocked reports whether the same agent has a blocked wait for evt.
// Another agent's wait does not hold back this agent's delivery.
func reservedLocked(evt Event, agentID string) bool {
	for _, w := range waiters {
		if w.agentID == agentID && w.filter.matches(evt) {
			return true
		}
	}
	return false
}

// --- test hooks ---

// SetQueueBoundsForTest overrides the byte and count budgets, returning a
// function that restores the previous values.
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

// SetTTLForTest overrides the event TTL, returning a restore function.
func SetTTLForTest(ttl time.Duration) func() {
	mu.Lock()
	prev := eventTTL
	eventTTL = ttl
	mu.Unlock()
	return func() {
		mu.Lock()
		eventTTL = prev
		mu.Unlock()
	}
}

// ResetForTest clears all state.
func ResetForTest() {
	mu.Lock()
	defer mu.Unlock()
	log = nil
	subs = map[string][]Filter{}
	delivered = map[string]map[string]bool{}
	missed = map[string]int{}
	waiters = map[uint64]*waiter{}
}
