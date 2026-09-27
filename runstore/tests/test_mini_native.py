from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from harness.core.journal import read_journal
from harness.orchestrator.run import Orchestrator
from harness.slots.mini_history import read_entries, training_messages
from harness.tests.test_mini_swe import FilesystemSession, model_server, owned_filesystem, reply, spec, pytestmark
from runstore.message_export import clean_messages, export_messages, mark_hint
from runstore.native import index_native, materialize, read_prefix


def parent_run(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("echo early > answer", "cat answer"),
                       reply("echo late > answer"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, _):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    path = tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl"
    points = index_native(outcome.journal_path, path, "mini-swe-agent", outcome.native_session_id)
    return memory, outcome, points


def test_exact_branch_restores_closed_prefix_without_parent_suffix(tmp_path, monkeypatch):
    memory, parent, points = parent_run(tmp_path / "parent", monkeypatch)
    assert [p.tool_depth for p in points] == [2, 3, 4]
    cut = points[0]
    original = read_prefix(cut.native)
    assert b"echo late" not in original
    child_dir = tmp_path / "child"
    extra = materialize(cut.native, child_dir / "cwd", child_dir / "restoration", child_dir / "native-home")
    child = FilesystemSession(child_dir / "sandbox")
    for name, data in memory.snapshots[cut.snapshot_id].items():
        (child.root / name).write_bytes(data)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    with model_server([reply("cat answer"), reply("echo child > answer"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        run_spec = spec(child_dir, url, prompt=mark_hint("private branch direction"))
        run_spec.extra.update(extra)
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "completed", outcome.error
    assert "private branch direction" in json.dumps(requests[0]["messages"])
    assert "echo late" not in json.dumps(requests)
    assert json.loads(requests[1]["messages"][-1]["content"])["output"] == "early\n"
    messages = export_messages(child_dir, outcome.native_session_id, "mini-swe-agent", [])
    assert "private branch direction" not in json.dumps(messages)
    assert "echo child" in json.dumps(messages) and "echo late" not in json.dumps(messages)
    child_path = child_dir / "native-home" / f"{outcome.native_session_id}.jsonl"
    parent_entries = read_entries(original)
    child_entries = read_entries(child_path.read_bytes())
    assert parent_entries[0]["workspace"] == child_entries[0]["workspace"]
    parent_messages = training_messages(parent_entries)
    assert requests[0]["messages"][0] == parent_messages[0]
    child_points = index_native(outcome.journal_path, child_path, "mini-swe-agent",
                                outcome.native_session_id, inherited_native=cut.native)
    assert [p.tool_depth for p in child_points] == [1, 2, 3]
    assert read_prefix(cut.native) == original
    assert (memory.root / "answer").read_text() == "late\n"
    assert (child.root / "answer").read_text() == "child\n"


def test_branch_cannot_change_inherited_workspace(tmp_path, monkeypatch):
    _, _, points = parent_run(tmp_path / "parent", monkeypatch)
    child_dir = tmp_path / "child"
    extra = materialize(points[0].native, child_dir / "cwd",
                        child_dir / "restoration", child_dir / "native-home")
    child = FilesystemSession(child_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    with model_server([]) as (url, requests):
        run_spec = spec(child_dir, url, prompt=mark_hint("continue"))
        run_spec.extra.update(extra)
        run_spec.extra["mini"] = {"environment": {"cwd": "/another-repository"}}
        outcome = Orchestrator().run(run_spec)
    assert "differs from its inherited workspace" in outcome.error
    assert requests == []


def test_modified_or_partial_prefix_cannot_be_restored(tmp_path, monkeypatch):
    _, parent, points = parent_run(tmp_path, monkeypatch)
    original = points[0].native
    changed = {**original, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="checksum changed"):
        materialize(changed, tmp_path / "cwd", tmp_path / "recovery", tmp_path / "home")
    path = Path(original["path"])
    entries = path.read_bytes().splitlines(keepends=True)
    # A partial file never publishes the final turn while it is being written.
    path.write_bytes(b"".join(entries[:-2]) + b'{"type":')
    indexed = index_native(parent.journal_path, path, "mini-swe-agent", parent.native_session_id)
    assert all(point.tool_depth < 4 for point in indexed)


def test_swebench_selector_consumes_mini_native_cuts(tmp_path, monkeypatch):
    from swebench.fork_eval import available_branch_points, conversation_restore, prepare_branches

    _, parent, points = parent_run(tmp_path, monkeypatch)
    eligible = available_branch_points(parent.journal_path)
    assert set(eligible) == {2, 3, 4}
    assert conversation_restore(parent.journal_path, 2, parent.native_session_id) == (points[0].native["cut"], None)
    from types import SimpleNamespace
    choices = prepare_branches(
        {"branches": [{"base": "parent", "branch_step": 2, "hint": "inspect", "why": "test"}]},
        limit=1, round_no=1, attempts={"parent": SimpleNamespace(outcome=parent)},
        checkpoints={"parent": eligible})
    assert choices[0].cut == points[0].native["cut"]


def test_timeout_between_actions_grades_the_matching_closed_prefix(tmp_path, monkeypatch):
    from dataclasses import asdict
    from harness.slots.mini_runtime import McpEnvironment
    from runstore.message_completion import complete_message_result, is_truncated_result

    original_execute = McpEnvironment.execute
    def stop_after_partial(env, action, cwd=""):
        output = original_execute(env, action, cwd)
        if "partial" in action["command"]:
            env.control.request_stop("test wall deadline", stop_reason="timeout")
        return output
    monkeypatch.setattr(McpEnvironment, "execute", stop_after_partial)
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("echo good > answer"),
                       reply("echo partial > answer", "echo never > answer")]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert len(requests) == 2 and outcome.stop_reason == "timeout"
    disk_at_stop = memory.snapshot().id
    result = complete_message_result(
        {**asdict(outcome), "final_snapshot_id": disk_at_stop, "failure_kind": "actor"},
        tmp_path, "mini-swe-agent")
    assert is_truncated_result(result) and not result.get("training_export_error"), result
    assert result["timeout_fallback"]["tool_depth"] == 1
    assert memory.snapshots[result["final_snapshot_id"]]["answer"] == b"good\n"
    assert memory.snapshots[disk_at_stop]["answer"] == b"partial\n"
    assert "echo partial" not in json.dumps(result["training_messages"])
    assert result["rollout_usage"] == {"model_calls": 2, "tool_calls": 2}


def test_infrastructure_error_is_never_relabelled_as_timeout(tmp_path, monkeypatch):
    from dataclasses import asdict
    from runstore.message_completion import complete_message_result

    memory, outcome, _ = parent_run(tmp_path, monkeypatch)
    result = complete_message_result(
        {**asdict(outcome), "status": "error", "stop_reason": "timeout",
         "final_snapshot_id": memory.snapshot().id, "failure_kind": "infrastructure",
         "error": "execution_uncertain"}, tmp_path, "mini-swe-agent")
    assert result["status"] == "error" and "timeout_fallback" not in result
