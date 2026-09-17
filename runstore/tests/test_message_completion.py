import json
import time
from types import SimpleNamespace

import pytest

from harness.core.control import RunControl
from harness.core.slot import SlotResult
from harness.execution.pipeline import ToolPipeline
from harness.orchestrator.run import Orchestrator, RunOutcome
from runstore.child import execute
from runstore.message_completion import complete_message_result, is_truncated_result


def write_history(directory):
    home = directory / "native-home"
    home.mkdir(exist_ok=True)
    path = home / "native.jsonl"
    items = [
        {"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": "fix the task"}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": "partial solution"}},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in items))
    return path


@pytest.mark.parametrize("reason", ["max_turns_reached", "timeout"])
def test_child_captures_final_state_and_exports_native_identity_on_cutoff(tmp_path, monkeypatch, reason):
    directory = tmp_path / "attempt"
    directory.mkdir()
    order = []
    session = SimpleNamespace(
        snapshot=lambda **kwargs: order.append("snapshot") or SimpleNamespace(id="final-state"),
    )
    owned = SimpleNamespace(
        server=SimpleNamespace(pipeline=ToolPipeline()), session=session, sandbox_id="vm",
        stop_server=lambda: order.append("drain"),
        destroy=lambda: order.append("destroy"),
    )
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: (owned, None))
    monkeypatch.setattr(Orchestrator, "_wire_gateway", lambda *args: None)
    monkeypatch.setattr(Orchestrator, "_wire_checkpoints", lambda *args: None)

    class Slot:
        def run(self, task, journal, mcp):
            task._rollout_controls.prepare_model_request({}, "responses")
            write_history(directory)
            journal.emit("session.ref", native_session_id="native")
            journal.emit("rollout.usage", model_calls=1, tool_calls=0)
            if reason == "max_turns_reached":
                task._rollout_controls.prepare_model_request({}, "responses")
            else:
                until = time.monotonic() + 2
                while task.control.reason is None and time.monotonic() < until:
                    time.sleep(0.01)
                assert task.control.stop_reason == "timeout"
            return SlotResult(status="error", error="stopped")

    monkeypatch.setattr("harness.slots.load_slot", lambda name: Slot)
    request = {
        "kind": "rollout", "attempt_id": "attempt", "profile_config": {}, "effective_spec": {
            "prompt": "fix", "slot": "codex", "sandbox_image": "image", "extra": {
                "rollout_contract": {
                    "message_export": True, "max_turns": 1,
                    "deadline_at": time.time() + (0.5 if reason == "timeout" else 30),
                    "model_endpoint": "http://model", "sampling_params": {},
                }
            }
        },
    }
    result = execute(request, directory)
    assert result["status"] == "truncated", result
    assert result["stop_reason"] == reason
    assert result["native_session_id"] == "native"
    assert result["final_snapshot_id"] == "final-state"
    assert result["training_messages"][-1]["content"] == "partial solution"
    assert order.index("drain") < order.index("snapshot") < order.index("destroy")
    assert is_truncated_result(result)


def test_cutoff_does_not_reclassify_uncertain_execution(tmp_path):
    write_history(tmp_path)
    (tmp_path / "trajectory.jsonl").write_text(json.dumps({
        "type": "checkpoint.captured", "reason": "execution_uncertain",
    }) + "\n")
    result = complete_message_result({
        "status": "error", "stop_reason": "timeout", "error": "execution_uncertain",
        "failure_kind": "infrastructure", "final_snapshot_id": "not-eligible",
        "native_session_id": "native",
    }, tmp_path, "codex")
    assert result["status"] == "error"
    assert not is_truncated_result(result)


@pytest.mark.parametrize("stop_reason", ["max_turns_reached", "timeout", None])
def test_real_claude_slot_distinguishes_limit_from_uncertain_execution(tmp_path, monkeypatch, stop_reason):
    sdk = pytest.importorskip("claude_agent_sdk")
    from harness.slots.claude_code import ClaudeCodeSlot

    directory = tmp_path / "attempt"
    directory.mkdir()
    order = []
    session = SimpleNamespace(snapshot=lambda **kwargs: order.append("snapshot") or SimpleNamespace(id="final"))
    owned = SimpleNamespace(server=SimpleNamespace(pipeline=ToolPipeline()), session=session,
                            sandbox_id="vm", stop_server=lambda: order.append("drain"),
                            destroy=lambda: order.append("destroy"))
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: (owned, None))
    monkeypatch.setattr(Orchestrator, "_wire_gateway", lambda *args: None)
    monkeypatch.setattr(Orchestrator, "_wire_checkpoints", lambda *args: None)
    monkeypatch.setattr(ClaudeCodeSlot, "version", lambda self: "test")
    original = ClaudeCodeSlot.run_async

    async def run_async(self, task, journal, mcp=None):
        async def query(**kwargs):
            yield sdk.SystemMessage(subtype="init", data={"session_id": "native"})
            task.control.request_stop("rollout budget exhausted" if stop_reason else "execution_uncertain",
                                      stop_reason=stop_reason)
            await asyncio.Event().wait()

        monkeypatch.setattr(sdk, "query", query)
        return await original(self, task, journal, mcp)

    import asyncio
    monkeypatch.setattr(ClaudeCodeSlot, "run_async", run_async)
    result = execute({"kind": "rollout", "attempt_id": "attempt", "profile_config": {},
                      "effective_spec": {"prompt": "task", "slot": "claude-code", "extra": {
                          "rollout_contract": {"message_export": True, "max_turns": 1,
                                               "deadline_at": time.time() + 10,
                                               "model_endpoint": "http://model", "sampling_params": {}}
                      }}}, directory)
    if stop_reason:
        assert result["status"] == "truncated", result
        assert result["stop_reason"] == stop_reason
        assert result["final_snapshot_id"] == "final"
        assert order.index("drain") < order.index("snapshot") < order.index("destroy")
    else:
        assert result["status"] == "error"
        assert "final_snapshot_id" not in result
        assert "snapshot" not in order
        assert "Uncertain tool execution" in result["training_snapshot_error"]


def test_cutoff_exports_complete_records_without_rewriting_partial_native_tail(tmp_path):
    path = write_history(tmp_path)
    path.write_bytes(path.read_bytes() + b'{"unfinished":')
    original = path.read_bytes()
    (tmp_path / "trajectory.jsonl").write_text("")
    result = complete_message_result({
        "status": "error", "stop_reason": "timeout", "error": "rollout wall-time budget exhausted",
        "final_snapshot_id": "final", "native_session_id": "native",
    }, tmp_path, "codex")
    assert result["status"] == "truncated"
    assert result["training_messages"][-1]["role"] == "assistant"
    assert path.read_bytes() == original


def test_limit_reason_does_not_replace_an_earlier_infrastructure_stop():
    control = RunControl()
    control.request_stop("execution_uncertain")
    control.request_stop("deadline", stop_reason="timeout")
    assert control.stop_reason is None


def test_timeout_outcome_records_status_without_claiming_natural_completion():
    result = RunOutcome(run_id="r", journal_path="unused", status="error", stop_reason="timeout")
    assert not result.ok


@pytest.mark.parametrize("cleanup_ok", [True, False])
def test_worker_publishes_scored_limit_episodes_only_after_cleanup(tmp_path, cleanup_ok):
    from runstore.worker import Worker

    finished = []
    worker = object.__new__(Worker)
    worker.store = SimpleNamespace(finish=lambda *args, **kwargs: finished.append((args, kwargs)))
    worker._ingest = lambda *args: []
    worker._cleanup = lambda *args: cleanup_ok
    worker._publish({"id": "job", "lease_token": "lease"}, tmp_path, {
        "status": "truncated", "stop_reason": "timeout", "final_snapshot_id": "final",
        "error": "rollout wall-time budget exhausted",
    }, {"effective_spec": {}})
    (args, kwargs), = finished
    assert args[2]["status"] == "truncated"
    if cleanup_ok:
        assert kwargs.get("state", "succeeded") == "succeeded"
        assert args[2]["failure_kind"] is None
    else:
        assert kwargs["state"] == "quarantined"
        assert args[2]["failure_kind"] == "infrastructure"
