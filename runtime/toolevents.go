package main

// Tool-call events: every tool invocation emits one event, so an observer can
// see what happened in the sandbox without having made the call itself.
//
// Emitted from the shared dispatch path rather than inside each tool, so all
// tools -- including ones added later -- are covered uniformly and no tool has
// to know the event system exists. Callers: the three transport handlers in
// main.go (http, mcp, stdio) via executeTool.
//
// Kind is "tool:<name>" (e.g. "tool:web_search") so the existing kind filter
// works unchanged, and Source is the call's subject (a URL, a query, a path,
// a pid) so wait_for_events can target one specific call with its source
// filter.

import (
	"github.com/dreamyang-liu/ash/runtime/events"
	"github.com/dreamyang-liu/ash/runtime/tools"
)

// toolEventSubjectKeys lists, per tool, which argument identifies the call's
// subject. The first key present becomes the event Source. An empty list means
// the tool's calls are not announced.
var toolEventSubjectKeys = map[string][]string{
	"shell":           {"command"},
	"process":         {"pid"},
	"text_editor":     {"path"},
	"grep_files":      {"pattern"},
	"web_fetch":       {"url"},
	"web_search":      {"query"},
	"artifact":        {"url"},
	"wait_for_events": {}, // waiting is not itself a reportable action
}

// fallbackSubjectKeys is used for tools with no explicit entry (e.g. custom
// tools routed through shell would already be covered, but a future runtime
// tool should still get a sensible source).
var fallbackSubjectKeys = []string{"path", "url", "command", "query"}

// toolEventSource picks a stable, human-meaningful handle for the call.
func toolEventSource(name string, args map[string]any) string {
	keys, known := toolEventSubjectKeys[name]
	if !known {
		keys = fallbackSubjectKeys
	}
	for _, k := range keys {
		if v, ok := args[k].(string); ok && v != "" {
			return v
		}
	}
	return name
}

// emitsToolEvent reports whether a tool's calls are worth announcing.
func emitsToolEvent(name string) bool {
	keys, known := toolEventSubjectKeys[name]
	return !known || len(keys) > 0
}

// eventKindForTool is the kind an observer passes to wait_for_events to await
// a given tool's calls.
func eventKindForTool(name string) string { return "tool:" + name }

// executeTool runs a tool, announces the call as an event, and drains pending
// events for piggyback delivery. It is the single dispatch path shared by
// every transport.
func executeTool(target tools.Tool, args map[string]any, agentID string) (tools.Result, []events.Event) {
	if agentID != "" {
		args["agent_id"] = agentID
	}

	result := target.Execute(args)

	// Drain before announcing this call: the piggyback payload reports what
	// happened *before* it. Draining afterwards would let the caller consume
	// its own just-announced event off the broadcast queue (where it is
	// dropped as its own action), leaving no other observer able to see it.
	var notifications []events.Event
	if agentID != "" {
		notifications = events.DrainFor(agentID)
	} else {
		notifications = events.Drain()
	}

	if emitsToolEvent(target.Name()) {
		data := map[string]any{
			"tool": target.Name(),
			"ok":   result.Success,
		}
		if !result.Success && result.Error != "" {
			data["error"] = result.Error
		}
		if n := len(result.Output); n > 0 {
			data["output_bytes"] = n
		}
		// Carry the arguments verbatim: shortening them is a consumer-side
		// decision, and the queue's byte budget already bounds memory.
		data["args"] = redactArgs(args)
		// PushAction: broadcast so other observers (a subagent sharing the
		// sandbox, an explicit wait, an SDK pipeline) can see it, with the
		// caller recorded as origin so it is not echoed back to them.
		events.PushAction(agentID, eventKindForTool(target.Name()),
			toolEventSource(target.Name(), args), data)
	}

	return result, notifications
}

// redactArgs copies call arguments for the event payload, dropping internal
// plumbing that would only be noise to an observer.
func redactArgs(args map[string]any) map[string]any {
	out := make(map[string]any, len(args))
	for k, v := range args {
		if k == "agent_id" {
			continue
		}
		out[k] = v
	}
	return out
}
