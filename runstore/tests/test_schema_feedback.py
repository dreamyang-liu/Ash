from copy import deepcopy
import json

import pytest

from harness.core.journal import read_journal
from harness.orchestrator.run import Orchestrator
from harness.rollback import turn_branch_checkpoints
from harness.slots.mini_history import read_entries
from harness.tests.test_mini_swe import FilesystemSession, model_server, owned_filesystem, pytestmark, reply, spec
from runstore.message_export import export_messages
from runstore.native import index_native, materialize, read_prefix
from runstore.tests.test_assistant_turn_branch import restore_files


def malformed_reply(kind="extra", commands=None):
    response = reply(*(commands or ["touch must-not-execute"]), ids=["bad-" + str(i) for i in range(len(commands or [1]))])
    function = response["choices"][0]["message"]["tool_calls"][-1]["function"]
    if kind == "extra":
        arguments = json.loads(function["arguments"])
        arguments["timeout_ms"] = 10000
        function["arguments"] = json.dumps(arguments)
    elif kind == "json":
        function["arguments"] = "{not-json"
    elif kind == "type":
        function["arguments"] = '{"command": 123}'
    elif kind == "unknown":
        function["name"] = "shell"
    return response


@pytest.mark.parametrize("kind", ["extra", "json", "type", "unknown"])
def test_schema_error_reaches_model_then_corrected_call_runs_and_remains_indexable(tmp_path, monkeypatch, kind):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    bad = malformed_reply(kind)
    original = deepcopy(bad)
    with model_server([bad, reply("printf fixed > answer", ids=["corrected"]),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ids=["submit"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert bad == original
    assert len(requests) == 3 and outcome.usage["input_tokens"] == 33
    assert not (memory.root / "must-not-execute").exists()
    assert (memory.root / "answer").read_text() == "fixed"
    # The original bad call is preserved, not cleaned or silently executed.
    previous = requests[1]["messages"][-2:]
    assert previous[0]["role"] == "assistant"
    assert previous[0]["tool_calls"][0]["function"] == original["choices"][0]["message"]["tool_calls"][0]["function"]
    assert previous[1]["role"] == "tool" and previous[1]["tool_call_id"] == "bad-0"
    feedback = json.loads(previous[1]["content"])
    assert feedback["executed"] is False and feedback["error"]["type"] == "tool_schema_error"
    assert feedback["available_tools"] == requests[1]["tools"]
    events = read_journal(outcome.journal_path)
    assert [e["call_id"] for e in events if e["type"] == "tool.started"] == ["corrected", "submit"]
    assert len([e for e in events if e["type"] == "tool.rejected"]) == 1
    assert set(turn_branch_checkpoints(outcome.journal_path)) == {1, 2}
    native = tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl"
    points = index_native(outcome.journal_path, native, "mini-swe-agent", outcome.native_session_id)
    assert [p.tool_depth for p in points] == [1, 2]
    assert points[0].message_step == 2  # rejected response + corrected response
    messages = export_messages(tmp_path, outcome.native_session_id, "mini-swe-agent", [])
    assert any(m.get("tool_call_id") == "bad-0" and json.loads(m["content"]).get("executed") is False
               for m in messages if m["role"] == "tool")


def test_entire_mixed_batch_is_rejected_before_any_command_executes(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([
        malformed_reply(commands=["touch valid-sibling-must-not-execute", "touch must-not-execute"]),
        reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
    ]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert not (memory.root / "valid-sibling-must-not-execute").exists()
    assert not (memory.root / "must-not-execute").exists()
    feedback = requests[1]["messages"][-2:]
    assert [m["tool_call_id"] for m in feedback] == ["bad-0", "bad-1"]
    assert [json.loads(m["content"])["error"]["type"] for m in feedback] == [
        "tool_batch_rejected", "tool_schema_error"]
    assert all(json.loads(m["content"])["executed"] is False for m in feedback)


def test_feedback_retries_use_existing_model_budget(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    first, second = malformed_reply(), malformed_reply()
    second["choices"][0]["message"]["tool_calls"][0]["id"] = "bad-second"
    with model_server([first, second]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["rollout_contract"]["max_turns"] = 2
        run_spec.extra["mini"] = {"agent": {"max_consecutive_format_errors": 0}}
        outcome = Orchestrator().run(run_spec)
    assert outcome.stop_reason == "max_turns_reached"
    assert len(requests) == 2 and outcome.usage["input_tokens"] == 22
    events = read_journal(outcome.journal_path)
    assert not any(e["type"] in {"tool.started", "checkpoint.captured"} for e in events)
    assert len([e for e in events if e["type"] == "mini.turn.rejected"]) == 2
    assert turn_branch_checkpoints(outcome.journal_path) == {}


def test_duplicate_call_ids_remain_a_protocol_error(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("touch never", "touch never2", ids=["duplicate", "duplicate"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "error" and "unique ids" in outcome.error
    assert len(requests) == 1
    assert not any(e["type"] in {"tool.started", "tool.rejected"} for e in read_journal(outcome.journal_path))


def test_consecutive_format_error_limit_still_applies(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    first, second = malformed_reply(), malformed_reply()
    second["choices"][0]["message"]["tool_calls"][0]["id"] = "second-error"
    with model_server([first, second]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["mini"] = {"agent": {"max_consecutive_format_errors": 2}}
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "error" and outcome.error == "RepeatedFormatError"
    assert len(requests) == 2
    assert not any(e["type"] == "tool.started" for e in read_journal(outcome.journal_path))


def test_provider_protocol_errors_are_not_mislabeled_as_tool_schema_errors(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([{"error": {"message": "provider failed"}}]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "error"
    assert len(requests) == 1
    assert not any(e["type"] == "tool.rejected" for e in read_journal(outcome.journal_path))


def test_correction_checkpoint_can_be_branched_with_rejected_history_intact(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent"
    memory = FilesystemSession(parent_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([malformed_reply(), reply("printf fixed > answer"),
                       reply("printf future > answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, _):
        parent = Orchestrator().run(spec(parent_dir, url))
    assert parent.status == "completed", parent.error
    native = parent_dir / "native-home" / f"{parent.native_session_id}.jsonl"
    points = index_native(parent.journal_path, native, "mini-swe-agent", parent.native_session_id)
    cut = points[0]
    preserved = read_prefix(cut.native)
    child_dir = tmp_path / "child"
    child = restore_files(memory, cut.snapshot_id, child_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    with model_server([reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        run_spec = spec(child_dir, url)
        run_spec.extra.update(materialize(cut.native, child_dir / "cwd",
                              child_dir / "restoration", child_dir / "native-home"), resume_without_hint=True)
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "completed", outcome.error
    assert requests[0]["messages"] == [
        {k: v for k, v in e["message"].items() if k != "extra"}
        for e in read_entries(preserved) if e["type"] == "mini.message"]
    assert "printf future" not in json.dumps(requests)
    assert json.loads(requests[1]["messages"][-1]["content"])["output"] == "fixed"
    child_native = child_dir / "native-home" / f"{outcome.native_session_id}.jsonl"
    assert len(index_native(outcome.journal_path, child_native, "mini-swe-agent",
                            outcome.native_session_id, inherited_native=cut.native)) == 2
    assert read_prefix(cut.native) == preserved
    # Reject a fabricated rejection marker with no corresponding journal proof.
    altered = [e for e in read_journal(parent.journal_path) if e["type"] != "mini.turn.rejected"]
    parent.journal_path.write_text("".join(json.dumps(e) + "\n" for e in altered))
    assert index_native(parent.journal_path, native, "mini-swe-agent", parent.native_session_id) == []
