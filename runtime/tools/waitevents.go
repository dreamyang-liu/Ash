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
	return "Observe asynchronous sandbox facts. Delivery is opt-in: action=subscribe registers interest in event kinds (optionally narrowed to specific sources) so they arrive with later tool responses; action=wait (the default) blocks until a matching event occurs or the timeout elapses. Events expire automatically after their time-to-live."
}

func (w *WaitEventsTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action": map[string]any{
				"type":        "string",
				"enum":        []string{"wait", "subscribe", "unsubscribe", "subscriptions"},
				"default":     "wait",
				"description": "wait: block for a matching event. subscribe: receive these kinds with later tool responses (nothing is delivered without a subscription). unsubscribe: stop receiving them. subscriptions: list active ones.",
			},
			"kinds": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Event kinds, e.g. [\"process_exited\", \"tool:text_editor\", \"tool:web_fetch\"]. Any tool call is observable as \"tool:<name>\". Required for subscribe; omit when waiting to accept any kind.",
			},
			"include_own": map[string]any{
				"type":        "boolean",
				"default":     false,
				"description": "Also receive events caused by your own tool calls. Off by default: your call's result already reported them.",
			},
			"sources": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Wait only for events from these handles: a pid returned by a background shell call, or a file path. Omit to accept any source. Use this to wait on one specific background process.",
			},
			"timeout": map[string]any{
				"type":        "integer",
				"default":     defaultWaitTimeoutSeconds,
				"description": "Seconds to wait before giving up. Clamped to the runtime maximum.",
			},
			// No agent_id property: the caller's identity is supplied by the
			// transport and injected by executeTool. Advertising it here would
			// invite a model to name another agent and read events addressed
			// to it.
		},
	}
}

// stringList extracts a []string from a JSON array argument, ignoring
// non-string and empty entries.
func stringList(v any) []string {
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, item := range raw {
		if s, ok := item.(string); ok && s != "" {
			out = append(out, s)
		}
	}
	return out
}

func (w *WaitEventsTool) Execute(args map[string]any) Result {
	agentID, _ := args["agent_id"].(string)
	filter := events.Filter{
		Kinds:      stringList(args["kinds"]),
		Sources:    stringList(args["sources"]),
		IncludeOwn: args["include_own"] == true,
	}

	switch action, _ := args["action"].(string); action {
	case "subscribe":
		if len(filter.Kinds) == 0 {
			return Err("subscribe requires kinds (subscribing to everything is rarely intended)")
		}
		events.Subscribe(agentID, filter)
		return okJSON(map[string]any{
			"subscribed": filter,
			"active":     events.Subscriptions(agentID),
		})
	case "unsubscribe":
		removed := events.Unsubscribe(agentID, filter.Kinds)
		return okJSON(map[string]any{
			"unsubscribed": removed,
			"active":       events.Subscriptions(agentID),
		})
	case "subscriptions":
		return okJSON(map[string]any{"active": events.Subscriptions(agentID)})
	case "", "wait":
		// fall through to waiting
	default:
		return Err("unknown action: " + action)
	}

	// A timeout of 0 is meaningful: check what is already there and return.
	// Treating it as "unset" would silently block for the default instead,
	// turning a poll into a 30-second wait.
	timeoutSeconds := defaultWaitTimeoutSeconds
	if t, ok := args["timeout"].(float64); ok && int(t) >= 0 {
		timeoutSeconds = clampInt(int(t), 0, maxWaitTimeoutSeconds)
	}

	matched := events.WaitFor(agentID, filter, time.Duration(timeoutSeconds)*time.Second)

	payload := map[string]any{
		"events":     matched,
		"timed_out":  len(matched) == 0,
		"waited_for": timeoutSeconds,
	}
	// Report subscribed events that expired or were trimmed before delivery,
	// instead of losing them silently.
	if n := events.TakeMissed(agentID); n > 0 {
		payload["missed"] = n
	}
	return okJSON(payload)
}

func okJSON(payload map[string]any) Result {
	out, err := json.Marshal(payload)
	if err != nil {
		return Err(fmt.Sprintf("encoding events: %v", err))
	}
	return Ok(string(out))
}
