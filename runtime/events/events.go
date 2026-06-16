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
	Data      map[string]any `json:"data"`
	Timestamp string         `json:"timestamp"`
}

var (
	mu      sync.Mutex
	queue   []Event
	counter atomic.Uint64
)

// Push adds an event to the queue. Called by tools when something happens
// (process exits, file changes, etc).
func Push(kind, source string, data map[string]any) {
	mu.Lock()
	defer mu.Unlock()

	id := counter.Add(1)
	queue = append(queue, Event{
		ID:        fmt.Sprintf("evt_%d", id),
		Kind:      kind,
		Source:    source,
		Data:      data,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	})

	// Keep max 100 events
	if len(queue) > 100 {
		queue = queue[len(queue)-100:]
	}
}

// Drain returns all pending events and clears the queue.
// Called by the HTTP handler after every tool execution.
func Drain() []Event {
	mu.Lock()
	defer mu.Unlock()

	if len(queue) == 0 {
		return []Event{}
	}

	result := queue
	queue = nil
	return result
}
