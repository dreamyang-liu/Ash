"""Tests for the SDK's typed event access and handle identity.

Run by pytest with the other swebench tests, against a fake Backend. Covers
what the SDK owns: parsing the runtime's event shape into values, and carrying
an identity on the handle so a consumer keeps its own cursor over the log.
User instruction: "wait for 就够了" + "我觉得可以" (identity on the handle).
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ash_sandbox import Event, Sandbox  # noqa: E402
from ash_sandbox.backends import Backend  # noqa: E402
from ash_sandbox.events import parse_batch, parse_events  # noqa: E402
from ash_sandbox.result import ToolResult  # noqa: E402


class EventBackend(Backend):
    """Serves a canned wait_for_events payload and records identities."""

    def __init__(self, payload=None, notifications=None):
        self.payload = payload if payload is not None else {"events": [], "timed_out": True}
        self.notifications = notifications or []
        self.calls = []

    async def call(self, tool_name, args, agent_id=""):
        self.calls.append((tool_name, dict(args), agent_id))
        if tool_name == "wait_for_events":
            return ToolResult(output=json.dumps(self.payload), is_error=False)
        return ToolResult(output="ok", is_error=False, notifications=self.notifications)

    async def list_tools(self):
        return []

    async def close(self):
        pass


PROCESS_EXITED = {
    "id": "evt_1", "kind": "process_exited", "source": "pid7",
    "data": {"exit_code": 3}, "timestamp": "2026-08-11T09:00:00Z",
    "agent_id": "main",
}
TOOL_CALL = {
    "id": "evt_2", "kind": "tool:web_search", "source": "golang",
    "data": {"tool": "web_search", "ok": True}, "origin": "main",
    "timestamp": "2026-08-11T09:00:01Z",
}


def test_event_parsing_keeps_every_field():
    e = Event.from_dict(PROCESS_EXITED)
    assert (e.id, e.kind, e.source) == ("evt_1", "process_exited", "pid7")
    assert e.data["exit_code"] == 3
    assert e.timestamp == "2026-08-11T09:00:00Z"
    assert e.agent_id == "main"


def test_tool_accessor_strips_the_kind_namespace():
    assert Event.from_dict(TOOL_CALL).tool == "web_search"
    # A non-tool event has no tool, rather than a misleading prefix-slice.
    assert Event.from_dict(PROCESS_EXITED).tool == ""


def test_batch_reports_what_it_could_not_return():
    batch = parse_batch(json.dumps({
        "events": [PROCESS_EXITED], "timed_out": False, "missed": 4,
    }))
    assert len(batch) == 1
    assert batch.missed == 4, "loss must stay visible in the result"
    assert not batch.timed_out


def test_batch_is_falsy_when_empty_and_iterable_when_not():
    empty = parse_batch(json.dumps({"events": [], "timed_out": True}))
    assert not empty and empty.timed_out

    full = parse_batch(json.dumps({"events": [PROCESS_EXITED, TOOL_CALL]}))
    assert full
    assert [e.kind for e in full] == ["process_exited", "tool:web_search"]
    assert [e.kind for e in full.of_kind("tool:web_search")] == ["tool:web_search"]


def test_malformed_payload_degrades_instead_of_raising():
    # A tool result is text; a truncated or non-JSON body should not crash a
    # harness mid-loop.
    assert parse_batch("not json").events == []
    assert parse_events(None) == []


def test_wait_for_events_passes_filters_and_timeout():
    backend = EventBackend({"events": [PROCESS_EXITED], "timed_out": False})
    sb = Sandbox(backend=backend, agent_id="main")

    batch = asyncio.run(sb.wait_for_events(kinds=["process_exited"], sources=["pid7"], timeout=5))
    assert batch.events[0].data["exit_code"] == 3

    tool, args, agent_id = backend.calls[-1]
    assert tool == "wait_for_events"
    assert args == {"timeout": 5, "kinds": ["process_exited"], "sources": ["pid7"]}
    assert agent_id == "main"


def test_poll_is_wait_with_no_timeout():
    backend = EventBackend()
    sb = Sandbox(backend=backend, agent_id="main")

    asyncio.run(sb.poll_events(kinds=["file_change"]))
    _, args, _ = backend.calls[-1]
    assert args["timeout"] == 0, "polling must not block"


def test_omitted_filters_are_not_sent():
    backend = EventBackend()
    sb = Sandbox(backend=backend)
    asyncio.run(sb.wait_for_events())
    _, args, _ = backend.calls[-1]
    assert args == {"timeout": 30}, "an unset filter should match everything"


def test_wait_failure_is_raised_not_returned_as_empty():
    class FailingBackend(EventBackend):
        async def call(self, tool_name, args, agent_id=""):
            return ToolResult(output="unknown action: nope", is_error=True)

    sb = Sandbox(backend=FailingBackend())
    with pytest.raises(RuntimeError, match="wait_for_events failed"):
        asyncio.run(sb.wait_for_events())


def test_handle_identity_is_used_without_repeating_it():
    backend = EventBackend()
    sb = Sandbox(backend=backend, agent_id="main")

    asyncio.run(sb.call("shell", command="ls"))
    asyncio.run(sb.wait_for_events())
    assert [c[2] for c in backend.calls] == ["main", "main"]


def test_explicit_identity_overrides_the_handle():
    backend = EventBackend()
    sb = Sandbox(backend=backend, agent_id="main")
    asyncio.run(sb.call("shell", agent_id="other", command="ls"))
    assert backend.calls[-1][2] == "other"


def test_as_agent_shares_the_sandbox_but_not_the_identity():
    backend = EventBackend()
    main = Sandbox(backend=backend, agent_id="main")
    reviewer = main.as_agent("reviewer")

    assert reviewer.backend is main.backend, "same sandbox"
    assert reviewer.tools is main.tools, "same tool panel"
    assert reviewer.agent_id == "reviewer"

    asyncio.run(reviewer.poll_events())
    assert backend.calls[-1][2] == "reviewer", "its own cursor over the log"


def test_piggybacked_events_are_available_typed():
    backend = EventBackend(notifications=[TOOL_CALL])
    sb = Sandbox(backend=backend, agent_id="observer")

    result = asyncio.run(sb.call("shell", command="true"))
    # The raw form stays visible; the typed view is derived from it.
    assert result.notifications == [TOOL_CALL]
    assert result.events[0].tool == "web_search"
    assert result.events[0].origin == "main"


def test_result_without_notifications_has_no_events():
    sb = Sandbox(backend=EventBackend())
    assert asyncio.run(sb.call("shell", command="true")).events == []
