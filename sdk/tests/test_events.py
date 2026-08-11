"""Tests for the SDK's typed event access and handle identity.

Run by pytest from the SDK (`pytest sdk/tests`), against a fake Backend. Covers
what the SDK owns: parsing the runtime's event shape into values, and carrying
an identity on the handle so a consumer keeps its own cursor over the log.
User instruction: "wait for 就够了" + "我觉得可以" (identity on the handle).
"""

import asyncio
import json
from pathlib import Path

import pytest

from ash_sandbox import Event, Sandbox
from ash_sandbox.backends import Backend
from ash_sandbox.events import parse_batch, parse_events
from ash_sandbox.result import ToolResult

class EventBackend(Backend):
    """Serves a canned wait_for_events payload and records identities."""

    def __init__(self, payload=None, notifications=None):
        self.payload = payload if payload is not None else {"events": [], "timed_out": True}
        self.notifications = notifications or []
        self.calls = []

    async def call(self, tool_name, args, agent_id=""):
        self.calls.append((tool_name, dict(args), agent_id))
        # Notifications ride on EVERY response, wait_for_events included: the
        # runtime drains an identity's owed events on every tool call, so a
        # fake that omits them here cannot exercise the two-channel case.
        if tool_name == "wait_for_events":
            return ToolResult(output=json.dumps(self.payload), is_error=False,
                              notifications=self.notifications)
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

def test_cli_backend_keeps_one_runtime_across_calls():
    """A sandbox's state lives in the runtime process, so it must persist.

    CLIBackend used to spawn a process per call. The filesystem survived
    (shared host fs) but the event log, background processes and artifact cache
    did not, so events silently never arrived -- through this backend only,
    while the README promised every transport behaves alike.
    """
    import shutil

    from ash_sandbox.backends import CLIBackend

    binary = shutil.which("ash-runtime") or "/tmp/ash-pg"
    if not Path(binary).exists():
        pytest.skip("no ash-runtime binary available")

    backend = CLIBackend(binary)

    async def scenario():
        first = await backend.call("shell", {"command": "echo $PPID"}, "cli")
        second = await backend.call("shell", {"command": "echo $PPID"}, "cli")
        proc = backend._proc
        await backend.close()
        return first.output.strip(), second.output.strip(), proc

    first, second, proc = asyncio.run(scenario())
    assert first and first == second, \
        f"one runtime should serve both calls, got {first!r} then {second!r}"
    assert proc is not None and proc.returncode is not None, \
        "close() must reap the runtime rather than leaking it"

def test_wait_returns_both_delivery_channels():
    """A wait_for_events response carries two sets of events, not one.

    The runtime drains an identity's owed events onto every tool response --
    including this one -- and marks them delivered. Returning only the matched
    batch dropped them permanently, and because they were gone from the log
    they were never counted as missed either: silent loss.
    """
    backend = EventBackend({"events": [PROCESS_EXITED], "timed_out": False},
                           notifications=[TOOL_CALL])
    sb = Sandbox(backend=backend, agent_id="main")

    batch = asyncio.run(sb.wait_for_events())
    kinds = sorted(e.kind for e in batch)
    assert kinds == ["process_exited", "tool:web_search"], \
        f"both channels must survive, got {kinds}"

def test_wait_does_not_report_an_event_twice():
    # The channels are disjoint today; if that ever changes, a caller must not
    # see one event on both.
    backend = EventBackend({"events": [PROCESS_EXITED], "timed_out": False},
                           notifications=[dict(PROCESS_EXITED)])
    sb = Sandbox(backend=backend)

    batch = asyncio.run(sb.wait_for_events())
    assert len(batch) == 1, f"deduplicate by id, got {[e.id for e in batch]}"

def test_per_call_identity_is_the_simple_path():
    # A component dispatching for several agents does not need a handle each:
    # the identity is a dispatch-time argument, invisible to the tool itself.
    backend = EventBackend()
    sb = Sandbox(backend=backend)

    asyncio.run(sb.call("shell", agent_id="parent", command="ls"))
    asyncio.run(sb.call("shell", agent_id="child", command="ls"))
    asyncio.run(sb.wait_for_events(agent_id="child"))

    assert [c[2] for c in backend.calls] == ["parent", "child", "child"]
    # None of it leaked into the arguments the tool sees.
    assert all("agent_id" not in args for _, args, _ in backend.calls)

def test_as_agent_shares_sandbox_state_not_just_the_connection():
    # Resolved binaries describe the sandbox, not the caller: a second
    # identity must not re-download what is already cached there.
    from ash_sandbox import ToolRegistry, parse_manifest

    registry = ToolRegistry()
    registry.register(parse_manifest({
        "name": "analyzer", "description": "d",
        "binary": {"url": "https://example.com/a", "sha256": "c" * 64},
        "parameters": {"file": {"type": "string", "required": True,
                                "map": {"positional": 0}}},
    }))

    class ArtifactBackend(EventBackend):
        async def call(self, tool_name, args, agent_id=""):
            self.calls.append((tool_name, dict(args), agent_id))
            output = "/cache/bin" if tool_name == "artifact" else "ok"
            return ToolResult(output=output, is_error=False)

    backend = ArtifactBackend()
    parent = Sandbox(backend=backend, tools=registry, agent_id="parent")
    asyncio.run(parent.call_agent_tool("analyzer", {"file": "x"}))
    assert [c[0] for c in backend.calls] == ["artifact", "shell"]

    child = parent.as_agent("child")
    backend.calls.clear()
    asyncio.run(child.call_agent_tool("analyzer", {"file": "x"}))

    assert [c[0] for c in backend.calls] == ["shell"], \
        "a second identity should reuse the sandbox's cached binary"
    assert backend.calls[0][2] == "child", "but act under its own identity"

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
