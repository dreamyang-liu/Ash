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

var (
	mu      sync.Mutex
	queues  = make(map[string][]Event) // per-agent queues, "" = broadcast
	counter atomic.Uint64
)

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

	// Cap per-agent queue
	if len(queues[agentID]) > 100 {
		queues[agentID] = queues[agentID][len(queues[agentID])-100:]
	}
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
