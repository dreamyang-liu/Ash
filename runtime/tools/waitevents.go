package tools

// WaitEventsTool is the runtime primitive for asynchronous sandbox facts:
// it blocks until the sandbox produces an event the agent cares about
// (background process exit, file change, ...) or until a timeout.
//
// This is a long poll, deliberately not a push: the sandbox never initiates
// a connection, so the transport stays a one-way request/response protocol
// and all three modes (http, stdio, mcp) keep identical semantics. Waiting
// also becomes an explicit agent action, visible in the trajectory.
//
// Events not matching `kinds` stay queued and ride along with a later tool
// response (the existing piggyback path in main.go). Registered in tool.go
// All(); invoked by agents through the normal tool interface.

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/dreamyang-liu/ash/runtime/events"
)

const (
	defaultWaitTimeoutSeconds = 30
	maxWaitTimeoutSeconds     = 300
)

type WaitEventsTool struct{}

func (w *WaitEventsTool) Name() string { return "wait_for_events" }

func (w *WaitEventsTool) Description() string {
	return "Block until the sandbox emits an event (e.g. a background process exits or a file changes), or until the timeout elapses. Returns the matching events as JSON."
}

func (w *WaitEventsTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"kinds": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Event kinds to wait for, e.g. [\"process_exited\", \"file_change\"]. Omit to wait for any event.",
			},
			"timeout": map[string]any{
				"type":        "integer",
				"default":     defaultWaitTimeoutSeconds,
				"description": "Seconds to wait before giving up. Clamped to the runtime maximum.",
			},
			"agent_id": map[string]any{
				"type":        "string",
				"description": "Only receive events targeted at this agent (plus broadcasts).",
			},
		},
	}
}

func (w *WaitEventsTool) Execute(args map[string]any) Result {
	var kinds []string
	if raw, ok := args["kinds"].([]any); ok {
		for _, k := range raw {
			if s, ok := k.(string); ok && s != "" {
				kinds = append(kinds, s)
			}
		}
	}
	timeoutSeconds := defaultWaitTimeoutSeconds
	if t, ok := args["timeout"].(float64); ok && int(t) > 0 {
		timeoutSeconds = clampInt(int(t), 1, maxWaitTimeoutSeconds)
	}
	agentID, _ := args["agent_id"].(string)

	matched := events.WaitFor(agentID, kinds, time.Duration(timeoutSeconds)*time.Second)

	payload := map[string]any{
		"events":     matched,
		"timed_out":  len(matched) == 0,
		"waited_for": timeoutSeconds,
	}
	// Surface queue overflow instead of losing events silently.
	if n := events.TakeDropped(agentID); n > 0 {
		payload["dropped"] = n
	}

	out, err := json.Marshal(payload)
	if err != nil {
		return Err(fmt.Sprintf("encoding events: %v", err))
	}
	return Ok(string(out))
}
