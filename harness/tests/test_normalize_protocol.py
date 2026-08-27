"""Normalizers for the protocol drivers (codex SDK, opencode server).

Fixtures are the shapes captured from live runs -- openai-codex 0.147.0 and
opencode 1.18.5 -- because all three bugs these tests pin down were invisible to
the published schemas and produced *plausible* journals rather than errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from harness.core import events as E
from harness.core.journal import JournalWriter, read_journal
from harness.normalize import codex_sdk as cx
from harness.normalize import opencode_server as oc


# --- fakes mirroring the SDK's pydantic surface -----------------------------
class Root:
    """Stands in for a pydantic RootModel wrapper."""

    def __init__(self, inner):
        self.root = inner


@dataclass
class Item:
    type: str
    id: str = "item-1"
    text: Optional[str] = None
    command: Any = None
    cwd: Any = None
    aggregated_output: Optional[str] = None
    exit_code: Optional[int] = None
    status: Optional[str] = None
    server: Optional[str] = None
    tool: Optional[str] = None
    arguments: Any = None

    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Payload:
    item: Any = None
    turn: Any = None
    token_usage: Any = None
    turn_id: Optional[str] = None
    message: Optional[str] = None

    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Note:
    method: str
    payload: Any


@dataclass
class Turn:
    id: str = "turn-1"
    status: str = "completed"


# --- codex: the three shapes that had to be read off live objects ----------
def test_root_model_items_are_unwrapped():
    """Reading the wrapper classified nothing: every item became raw.codex."""
    note = Note("item/completed", Payload(item=Root(Item(type="agentMessage", text="hi"))))
    assert cx.normalize(note) == [(E.AGENT_MESSAGE, {"text": "hi", "part_id": "item-1"})]


def test_discriminator_is_type_not_item_type():
    note = Note("item/started", Payload(item=Item(type="commandExecution",
                                                  command="ls", cwd="/w")))
    (event_type, payload), = cx.normalize(note)
    assert event_type == E.TOOL_STARTED
    assert payload["name"] == "shell"
    assert payload["args"] == {"command": "ls", "cwd": "/w"}


def test_root_models_inside_tool_args_are_unwrapped():
    """A wrapped cwd rendered as "root='/w'" in the journal."""
    note = Note("item/started", Payload(item=Item(type="commandExecution",
                                                  command=Root("ls -la"),
                                                  cwd=Root("/w"))))
    (_, payload), = cx.normalize(note)
    assert payload["args"] == {"command": "ls -la", "cwd": "/w"}


def test_turn_completed_is_a_boundary_and_carries_no_usage():
    """Taking usage from the turn yielded zeros for every run."""
    note = Note("turn/completed", Payload(turn=Turn()))
    (event_type, payload), = cx.normalize(note)
    assert event_type == E.TURN_COMPLETED
    assert payload["boundary"] is True
    assert "usage" not in payload


def test_token_usage_is_not_a_turn_boundary():
    """It must not be TURN_COMPLETED: the checkpoint bridge snapshots on those,
    so a usage report there took a snapshot per token update."""
    note = Note("thread/tokenUsage/updated", Payload(
        token_usage={"total": {"input_tokens": 100, "output_tokens": 7,
                               "cached_input_tokens": 60,
                               "cache_write_input_tokens": 3,
                               "reasoning_output_tokens": 2}},
        turn_id="turn-1"))
    (event_type, payload), = cx.normalize(note)
    assert event_type == E.USAGE_UPDATED
    assert payload["cumulative"] is True
    assert payload["usage"]["input_tokens"] == 100
    assert payload["usage"]["cache_creation_tokens"] == 3   # not the Anthropic name
    assert payload["usage"]["reasoning_output_tokens"] == 2


def test_cumulative_usage_replaces_rather_than_sums():
    """`total` is thread-cumulative; summing reports multiplies the real count.

    Uses ``stream_into`` rather than ``collect_and_journal``: the latter hands the
    stream to the SDK's own collector, which isinstance-checks real pydantic
    payloads, so driving it with fixtures would test the SDK and not this mapping.
    """
    handle = _FakeHandle([
        Note("thread/tokenUsage/updated", Payload(token_usage={"total": {"input_tokens": 100}})),
        Note("thread/tokenUsage/updated", Payload(token_usage={"total": {"input_tokens": 250}})),
        Note("turn/completed", Payload(turn=Turn())),
    ])
    journal = _MemJournal()
    usage = cx.stream_into(handle, journal)
    assert usage["input_tokens"] == 250          # not 350

    # And the boundary event is still exactly one, so one checkpoint fires.
    boundaries = [p for t, p in journal.events
                  if t == E.TURN_COMPLETED and p.get("boundary")]
    assert len(boundaries) == 1


def test_user_message_items_are_not_agent_output():
    note = Note("item/completed", Payload(item=Item(type="userMessage", text="the prompt")))
    assert cx.normalize(note) == []


def test_deltas_are_dropped():
    for method in ("item/agentMessage/delta", "item/reasoning/textDelta",
                   "item/commandExecution/outputDelta"):
        assert cx.normalize(Note(method, Payload())) == []


def test_failed_command_is_an_error_result():
    note = Note("item/completed", Payload(item=Item(
        type="commandExecution", aggregated_output="boom", exit_code=2)))
    (_, payload), = cx.normalize(note)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["output"] == "boom"


def test_mcp_tool_calls_are_named_by_server_and_tool():
    note = Note("item/started", Payload(item=Item(
        type="mcpToolCall", server="ash", tool="shell", arguments={"command": "ls"})))
    (_, payload), = cx.normalize(note)
    assert payload["name"] == "ash__shell"
    assert payload["args"] == {"command": "ls"}


def test_unknown_methods_are_preserved_never_dropped():
    (event_type, payload), = cx.normalize(Note("thread/somethingNew", Payload()))
    assert event_type == E.raw_event("codex")
    assert payload["method"] == "thread/somethingNew"


# --- opencode server -------------------------------------------------------
ASSISTANT = {
    "info": {
        "id": "msg_2", "role": "assistant", "finish": "stop",
        "modelID": "claude", "cost": 0.031,
        "tokens": {"total": 8335, "input": 3, "output": 9, "reasoning": 0,
                   "cache": {"read": 11, "write": 8323}},
    },
    "parts": [
        {"type": "step-start", "id": "prt_1", "messageID": "msg_2"},
        {"type": "text", "id": "prt_2", "messageID": "msg_2", "text": "done"},
        {"type": "step-finish", "id": "prt_3", "messageID": "msg_2"},
    ],
}
USER = {"info": {"id": "msg_1", "role": "user"},
        "parts": [{"type": "text", "id": "prt_0", "text": "do it"}]}


def test_cache_dimensions_are_split_and_total_ignored():
    """tokens.total counts cache writes, so summing it double counts."""
    usage = oc.map_usage(ASSISTANT["info"])
    assert usage["input_tokens"] == 3
    assert usage["output_tokens"] == 9
    assert usage["cached_input_tokens"] == 11
    assert usage["cache_creation_tokens"] == 8323
    assert usage["cost_usd"] == 0.031


def test_message_ids_are_recorded_as_branch_points():
    """POST /session/{id}/fork takes a messageID, so the journal must carry it."""
    events = oc.normalize_message(ASSISTANT)
    kinds = dict((t, p) for t, p in events)
    assert kinds[E.AGENT_MESSAGE]["message_id"] == "msg_2"
    assert kinds[E.AGENT_MESSAGE]["part_id"] == "prt_2"
    assert oc.message_ids([USER, ASSISTANT]) == ["msg_1", "msg_2"]


def test_tool_parts_produce_a_started_and_finished_pair():
    part = {
        "type": "tool", "id": "prt_9", "callID": "call_1", "tool": "read",
        "state": {"status": "completed", "input": {"filePath": "/x"},
                  "output": "contents"},
    }
    events = oc.normalize_part(part, "msg_2")
    assert [t for t, _ in events] == [E.TOOL_STARTED, E.TOOL_FINISHED]
    assert events[0][1]["name"] == "read"
    assert events[0][1]["args"] == {"filePath": "/x"}
    assert events[1][1]["status"] == "ok"
    assert events[1][1]["output"] == "contents"


def test_a_pending_tool_has_no_finished_event_yet():
    part = {"type": "tool", "id": "p", "callID": "c", "tool": "bash",
            "state": {"status": "running", "input": {}}}
    assert [t for t, _ in oc.normalize_part(part, "m")] == [E.TOOL_STARTED]


def test_errored_tool_is_reported_as_error():
    part = {"type": "tool", "id": "p", "callID": "c", "tool": "bash",
            "state": {"status": "error", "input": {}, "output": "nope"}}
    finished = [p for t, p in oc.normalize_part(part, "m") if t == E.TOOL_FINISHED][0]
    assert finished["status"] == "error"


def test_step_markers_add_nothing():
    assert oc.normalize_part({"type": "step-start", "id": "p"}, "m") == []
    assert oc.normalize_part({"type": "step-finish", "id": "p"}, "m") == []


def test_unknown_part_types_are_preserved():
    (event_type, payload), = oc.normalize_part({"type": "brand-new", "id": "p"}, "m")
    assert event_type == E.raw_event("opencode")
    assert payload["part"] == "brand-new"


def test_emit_history_accumulates_usage(tmp_path):
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r")
    usage = oc.emit_history([USER, ASSISTANT], journal)
    journal.close()
    assert usage["output_tokens"] == 9
    types = [r["type"] for r in read_journal(tmp_path / "j.jsonl")]
    assert E.TURN_COMPLETED in types
    assert types.count(E.TURN_COMPLETED) == 1      # only the assistant turn


def test_emit_history_can_skip_already_journalled_messages(tmp_path):
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r")
    oc.emit_history([USER, ASSISTANT], journal, since="msg_1")
    journal.close()
    records = read_journal(tmp_path / "j.jsonl")
    assert all(r.get("message_id") != "msg_1" for r in records)


def test_final_text_takes_the_last_assistant_message():
    assert oc.final_text([USER, ASSISTANT]) == "done"
    assert oc.final_text([USER]) == ""


# --- helpers ---------------------------------------------------------------
@dataclass
class _FakeHandle:
    notes: List[Any]
    id: str = "turn-1"

    def stream(self):
        for note in self.notes:
            yield note


class _MemJournal:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **payload):
        self.events.append((event_type, payload))
        return payload
