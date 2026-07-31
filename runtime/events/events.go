package events

import (
	"fmt"
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

// maxQueueLen bounds each per-agent queue; overflow increments a dropped
// counter that is reported to the agent rather than silently discarded.
const maxQueueLen = 100

var (
	mu      sync.Mutex
	queues  = make(map[string][]Event) // per-agent queues, "" = broadcast
	dropped = make(map[string]int)     // per-queue overflow counts
	waiters []chan struct{}            // signalled on every Push (see WaitFor)
	counter atomic.Uint64
)

// notifyWaitersLocked wakes every blocked WaitFor caller. Must hold mu.
func notifyWaitersLocked() {
	for _, ch := range waiters {
		close(ch)
	}
	waiters = nil
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

	// Cap per-agent queue, counting what fell off the front so the loss
	// can be reported instead of silently vanishing.
	if overflow := len(queues[agentID]) - maxQueueLen; overflow > 0 {
		queues[agentID] = queues[agentID][overflow:]
		dropped[agentID] += overflow
	}

	notifyWaitersLocked()
}

// Drain returns pending events for a specific agent (+ broadcasts), clears them.
func Drain() []Event {
	return DrainFor("")
}

// DrainFor returns events targeted at agentID plus any broadcast events.
func DrainFor(agentID string) []Event {
	mu.Lock()
	defer mu.Unlock()

	var result []Event

	// Broadcast events (agentID = "")
	if broadcasts, ok := queues[""]; ok && len(broadcasts) > 0 {
		result = append(result, broadcasts...)
		queues[""] = nil
	}

	// Agent-specific events
	if agentID != "" {
		if agentEvents, ok := queues[agentID]; ok && len(agentEvents) > 0 {
			result = append(result, agentEvents...)
			queues[agentID] = nil
		}
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

// drainMatchingLocked removes and returns events for agentID (plus
// broadcasts) whose Kind is in kinds, leaving non-matching events queued so
// a later call still delivers them. Must hold mu.
func drainMatchingLocked(agentID string, kinds []string) []Event {
	sources := []string{""} // broadcast queue
	if agentID != "" {
		sources = append(sources, agentID)
	}

	var taken []Event
	for _, q := range sources {
		pending := queues[q]
		if len(pending) == 0 {
			continue
		}
		var kept []Event
		for _, evt := range pending {
			if matches(evt, kinds) {
				taken = append(taken, evt)
			} else {
				kept = append(kept, evt)
			}
		}
		queues[q] = kept
	}
	return taken
}

// WaitFor blocks until at least one event matching kinds is available for
// agentID (broadcast events included), or until timeout elapses. It is a
// long-poll: the sandbox never initiates a connection, so the transport
// stays a plain one-way request/response protocol.
//
// Only matching events are consumed; anything else stays queued and rides
// along with a later tool response.
func WaitFor(agentID string, kinds []string, timeout time.Duration) []Event {
	deadline := time.Now().Add(timeout)
	for {
		mu.Lock()
		if taken := drainMatchingLocked(agentID, kinds); len(taken) > 0 {
			mu.Unlock()
			return taken
		}
		// Register for a wakeup before releasing the lock, so an event
		// pushed between the check and the wait cannot be missed.
		signal := make(chan struct{})
		waiters = append(waiters, signal)
		mu.Unlock()

		remaining := time.Until(deadline)
		if remaining <= 0 {
			return []Event{}
		}
		timer := time.NewTimer(remaining)
		select {
		case <-signal:
			timer.Stop() // an event arrived: re-check the queues
		case <-timer.C:
			return []Event{}
		}
	}
}
