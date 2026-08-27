"""Normalizer tests against fixtures captured from real CLI runs.

Fixtures were recorded with codex-cli 0.145.0 / opencode 1.18.5. When an upstream
bump changes the stream, these fail first -- which is the point (see contracts/).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.core import events as E
from harness.normalize import claude_code as cc
from harness.normalize import codex as cx
from harness.normalize import opencode as oc

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return [json.loads(l) for l in (FIXTURES / name).read_text().splitlines() if l.strip()]


def run_normalizer(normalize, records):
    out = []
    for record in records:
        out.extend(normalize(record))
    return out


def types_of(events):
    return [t for t, _ in events]


# --- opencode --------------------------------------------------------------
def test_opencode_maps_tool_call_pair():
    events = run_normalizer(oc.normalize, load("opencode_read_tool.jsonl"))
    kinds = types_of(events)
    assert E.TOOL_STARTED in kinds and E.TOOL_FINISHED in kinds

    started = next(p for t, p in events if t == E.TOOL_STARTED)
    finished = next(p for t, p in events if t == E.TOOL_FINISHED)
    assert started["name"] == "read"
    assert started["args"]["filePath"].endswith("probe.txt")
    # opencode emits one terminal event; the start is derived and marked as such.
    assert started["synthetic"] is True
    assert started["call_id"] == finished["call_id"]
    assert finished["status"] == "ok"
    assert "hello" in finished["output"]
    assert finished["duration_ms"] == 19


def test_opencode_usage_splits_cache_dimensions():
    events = run_normalizer(oc.normalize, load("opencode_read_tool.jsonl"))
    usages = [p["usage"] for t, p in events if t == E.TURN_COMPLETED]
    assert len(usages) == 2
    first = usages[0]
    assert first["input_tokens"] == 10
    assert first["output_tokens"] == 20
    assert first["cached_input_tokens"] == 70      # cache.read
    assert first["cache_creation_tokens"] == 3900  # cache.write
    assert first["cost_usd"] == 0.01
    # tokens.total is deliberately ignored (it double counts cache writes).
    assert first["input_tokens"] + first["output_tokens"] != 4000


def test_opencode_session_ref_and_text():
    events = run_normalizer(oc.normalize, load("opencode_read_tool.jsonl"))
    session = next(p for t, p in events if t == E.SESSION_REF)
    assert session["native_session_id"].startswith("ses_")
    texts = [p["text"] for t, p in events if t == E.AGENT_MESSAGE]
    assert texts == ["The file contains: hello"]


# --- codex -----------------------------------------------------------------
def test_codex_maps_command_and_mcp_calls():
    events = run_normalizer(cx.normalize, load("codex_shell_tool.jsonl"))
    started = [p for t, p in events if t == E.TOOL_STARTED]
    finished = [p for t, p in events if t == E.TOOL_FINISHED]
    assert [s["name"] for s in started] == ["shell", "ash__shell"]
    assert started[0]["args"]["command"] == "cat probe.txt"
    # mcp arguments arrive JSON-encoded and must be parsed, not passed through.
    assert started[1]["args"] == {"command": "ls"}
    assert finished[0]["exit_code"] == 0
    assert finished[0]["status"] == "ok"
    assert "hello" in finished[0]["output"]


def test_codex_usage_and_reasoning():
    events = run_normalizer(cx.normalize, load("codex_shell_tool.jsonl"))
    usage = next(p["usage"] for t, p in events if t == E.TURN_COMPLETED)
    assert usage["input_tokens"] == 120
    assert usage["cached_input_tokens"] == 64
    assert usage["reasoning_output_tokens"] == 12
    thinking = [p["text"] for t, p in events if t == E.AGENT_THINKING]
    assert thinking == ["I should inspect the file."]


def test_codex_thread_id_becomes_session_ref():
    events = run_normalizer(cx.normalize, load("codex_shell_tool.jsonl"))
    session = next(p for t, p in events if t == E.SESSION_REF)
    assert session["native_session_id"].endswith("000000000001")


# --- claude-code (duck-typed SDK objects) ----------------------------------
class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TextBlock(_Block):
    pass


class ThinkingBlock(_Block):
    pass


class ToolUseBlock(_Block):
    pass


class ToolResultBlock(_Block):
    pass


class AssistantMessage(_Block):
    pass


class UserMessage(_Block):
    pass


class ResultMessage(_Block):
    pass


class SystemMessage(_Block):
    pass


def test_claude_code_assistant_blocks():
    message = AssistantMessage(
        content=[
            ThinkingBlock(text="planning"),
            TextBlock(text="running a command"),
            ToolUseBlock(id="tu_1", name="mcp__ash__shell", input={"command": "ls"}),
        ]
    )
    events = cc.normalize(message)
    assert types_of(events) == [E.AGENT_THINKING, E.AGENT_MESSAGE, E.TOOL_STARTED]
    assert events[2][1]["call_id"] == "tu_1"
    assert events[2][1]["args"] == {"command": "ls"}


def test_claude_code_tool_result_and_usage():
    events = cc.normalize(
        UserMessage(content=[ToolResultBlock(tool_use_id="tu_1", content="probe.txt", is_error=False)])
    )
    assert events[0][0] == E.TOOL_FINISHED
    assert events[0][1] == {"call_id": "tu_1", "status": "ok", "output": "probe.txt"}

    result = ResultMessage(
        subtype="success",
        result="done",
        session_id="sess_abc",
        is_error=False,
        num_turns=3,
        total_cost_usd=0.25,
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        },
    )
    events = cc.normalize(result)
    usage = next(p["usage"] for t, p in events if t == E.TURN_COMPLETED)
    assert usage["cached_input_tokens"] == 80
    assert usage["cache_creation_tokens"] == 20
    assert usage["cost_usd"] == 0.25
    assert next(p["native_session_id"] for t, p in events if t == E.SESSION_REF) == "sess_abc"
    assert next(p["text"] for t, p in events if t == E.RUN_RESULT) == "done"


def test_claude_code_init_captures_capabilities():
    events = cc.normalize(
        SystemMessage(
            subtype="init",
            data={"session_id": "s1", "capabilities": ["interrupt_receipt_v1"], "tools": ["Bash"]},
        )
    )
    assert events[0][0] == E.SESSION_REF
    assert events[0][1]["capabilities"] == ["interrupt_receipt_v1"]


# --- the shared invariant --------------------------------------------------
@pytest.mark.parametrize(
    "normalize,payload",
    [
        (oc.normalize, {"type": "brand_new_event", "part": {"type": "mystery"}}),
        (cx.normalize, {"type": "brand_new_event", "data": 1}),
        (cc.normalize, object()),
    ],
)
def test_unknown_input_is_preserved_never_dropped(normalize, payload):
    """Unmapped native payloads must survive as raw.* events.

    A silent drop is the one unacceptable failure mode: the trajectory would be
    wrong with nothing surfacing an error.
    """
    events = normalize(payload)
    assert events, "normalizer dropped an unknown event"
    assert all(t.startswith("raw.") for t, _ in events)


# --- turn boundaries (quiesce points for snapshotting) ---------------------
def test_turn_started_is_mapped_not_raw():
    """turn.started marks a quiesce boundary; it must not land in raw.*."""
    assert cx.normalize({"type": "turn.started"}) == [(E.TURN_STARTED, {})]
    events = oc.normalize(
        {"type": "step_start", "sessionID": "ses_1", "part": {"type": "step-start"}}
    )
    assert types_of(events) == [E.SESSION_REF, E.TURN_STARTED]


def test_real_streams_have_no_unmapped_events():
    """Fixtures captured from real CLIs must map cleanly end to end."""
    for name, normalize in (
        ("opencode_read_tool.jsonl", oc.normalize),
        ("codex_shell_tool.jsonl", cx.normalize),
    ):
        events = run_normalizer(normalize, load(name))
        unmapped = [t for t in types_of(events) if t.startswith("raw.")]
        assert not unmapped, "%s produced %s" % (name, unmapped)
