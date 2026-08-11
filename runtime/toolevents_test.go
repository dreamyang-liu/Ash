package main

// Tests for executeTool, the dispatch path every transport shares (run by
// `go test .`). It is where a call's identity is established and where the
// call is announced to observers, so both are pinned here rather than in any
// one transport's tests.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/dreamyang-liu/ash/runtime/events"
	"github.com/dreamyang-liu/ash/runtime/tools"
)

// fakeTool records the arguments dispatch handed it.
type fakeTool struct {
	name string
	seen map[string]any
}

func (f *fakeTool) Name() string           { return f.name }
func (f *fakeTool) Description() string    { return "fake" }
func (f *fakeTool) Schema() map[string]any { return map[string]any{"type": "object"} }

func (f *fakeTool) Execute(a map[string]any) tools.Result {
	f.seen = a
	return tools.Ok("done")
}

func TestIdentityComesFromTheTransportNotTheArguments(t *testing.T) {
	events.ResetForTest()
	t.Cleanup(events.ResetForTest)

	// A model that puts agent_id in its arguments must not be able to choose
	// who it is: the transport's identity wins.
	f := &fakeTool{name: "shell"}
	executeTool(f, map[string]any{"command": "ls", "agent_id": "victim"}, "worker-3")

	if got := f.seen["agent_id"]; got != "worker-3" {
		t.Errorf("agent_id should be the transport's, got %q", got)
	}
}

func TestAnonymousTransportCannotBeUpgradedByAnArgument(t *testing.T) {
	events.ResetForTest()
	t.Cleanup(events.ResetForTest)

	// The dangerous case: no identity on the connection at all. A supplied
	// agent_id must still be discarded, or an anonymous caller could read
	// events addressed to a named agent by simply naming it.
	events.PushTo("victim", "process_exited", "pid9", map[string]any{"secret": 1})

	result, _ := executeTool(tools.Find("wait_for_events"), map[string]any{
		"agent_id": "victim",
		"kinds":    []any{"process_exited"},
		"timeout":  float64(0),
	}, "")

	var payload struct {
		Events []struct {
			Kind string `json:"kind"`
		} `json:"events"`
	}
	if err := json.Unmarshal([]byte(result.Output), &payload); err != nil {
		t.Fatalf("bad payload %q: %v", result.Output, err)
	}
	if len(payload.Events) != 0 {
		t.Errorf("impersonation: read %d event(s) addressed to another agent",
			len(payload.Events))
	}

	// The victim's event is still there for its actual owner.
	events.Subscribe("victim", events.Filter{Kinds: []string{"process_exited"}})
	if got := events.DrainFor("victim"); len(got) != 1 {
		t.Errorf("the targeted event should still reach its owner, got %d", len(got))
	}
}

func TestWaitForEventsDoesNotAdvertiseIdentity(t *testing.T) {
	// A property in the schema is an invitation to the model. Identity is not
	// the model's to choose, so it must not appear there.
	schema := tools.Find("wait_for_events").Schema()
	props, _ := schema["properties"].(map[string]any)
	if _, exposed := props["agent_id"]; exposed {
		t.Error("wait_for_events must not expose agent_id to the model")
	}
}

func TestCallIsAnnouncedToOtherObservers(t *testing.T) {
	events.ResetForTest()
	t.Cleanup(events.ResetForTest)

	events.Subscribe("observer", events.Filter{Kinds: []string{"tool:shell"}})
	executeTool(&fakeTool{name: "shell"}, map[string]any{"command": "make test"}, "worker-3")

	got := events.DrainFor("observer")
	if len(got) != 1 {
		t.Fatalf("an observer should see another agent's call, got %d", len(got))
	}
	if got[0].Origin != "worker-3" {
		t.Errorf("origin should identify the caller, got %q", got[0].Origin)
	}
	if got[0].Source != "make test" {
		t.Errorf("source should be the call's subject, got %q", got[0].Source)
	}
	// The identity is plumbing, not something an observer needs echoed back.
	if data, ok := got[0].Data["args"].(map[string]any); ok {
		if _, leaked := data["agent_id"]; leaked {
			t.Error("agent_id should be redacted from the announced args")
		}
	}
}

func TestCallerDoesNotSeeItsOwnCallEchoed(t *testing.T) {
	events.ResetForTest()
	t.Cleanup(events.ResetForTest)

	events.Subscribe("worker-3", events.Filter{Kinds: []string{"tool:shell"}})
	// Two calls: the second's piggyback must not carry the first, because both
	// were this caller's own actions and its results already reported them.
	executeTool(&fakeTool{name: "shell"}, map[string]any{"command": "a"}, "worker-3")
	_, notifications := executeTool(&fakeTool{name: "shell"},
		map[string]any{"command": "b"}, "worker-3")

	for _, e := range notifications {
		if strings.HasPrefix(e.Kind, "tool:") && e.Origin == "worker-3" {
			t.Errorf("own action echoed back: %s from %s", e.Kind, e.Origin)
		}
	}
}
