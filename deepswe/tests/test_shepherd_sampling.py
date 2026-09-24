"""Shepherd sampling, exact restore, bridge and budget invariants."""

import json
from types import SimpleNamespace

import pytest

from deepswe.branching.bridge import from_chat, message_events, to_chat
from deepswe.branching.policies import shepherd_select
from deepswe.branching.project_scope import (
    ProjectBindingSnapshot, capture_command, checked_workdir, restore_commands,
    verify_snapshot,
)
from deepswe.branching.runner import BenchmarkRunner, Config, result_resolved
from deepswe.branching.storage import rows, save


def test_jsonl_preserves_unicode_line_separator_in_model_text(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"content": "a\u2028b"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert rows(path) == [{"content": "a\u2028b"}]


@pytest.mark.parametrize("step", [True, 1.5, "2", -1, 3])
def test_shepherd_rejects_wrong_checkpoint_without_clamping(step):
    with pytest.raises(ValueError):
        shepherd_select({"checkpoint_step": step, "reason": "earliest mistake"}, [2, 4], 7)


def test_shepherd_one_state_and_no_hint():
    selected = shepherd_select({"checkpoint_step": 4, "reason": "wrong file"}, [2, 4], 7)
    assert len(selected) == 7
    assert {r["step"] for r in selected} == {4}
    assert all("hint" not in r for r in selected)


@pytest.mark.parametrize("path", ["/", "/etc", "/root", "relative", "/app/../etc"])
def test_project_binding_rejects_system_or_non_normalized_roots(path):
    with pytest.raises(ValueError):
        checked_workdir(path)


def test_project_binding_commands_touch_only_declared_root():
    assert capture_command("/app/project", "/tmp/state.tgz") == (
        "tar -czf /tmp/state.tgz -C /app/project .")
    prepare, restore = restore_commands("/app/project", "/tmp/state.tgz")
    assert "find /app/project" in prepare
    assert " -C /app/project" in restore
    assert "/etc" not in prepare + restore


def test_project_binding_manifest_is_content_addressed(tmp_path):
    archive = tmp_path / "state.tgz"
    archive.write_bytes(b"project-state")
    import hashlib
    manifest = ProjectBindingSnapshot(
        workdir="/app", checkpoint_step=3, archive=str(archive),
        archive_sha256=hashlib.sha256(b"project-state").hexdigest(),
        archive_bytes=len(b"project-state"), conversation_session_id="session",
        conversation_cut="cut",
    ).to_dict()
    assert verify_snapshot(manifest).checkpoint_step == 3
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size|hash"):
        verify_snapshot(manifest)


def test_reasoning_and_tool_history_survive_protocol_round_trip():
    body = {"system": [{"type": "text", "text": "system"}], "messages": [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "retained thought"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "shell", "input": {"command": "pwd"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]}]}
    native = to_chat(body, "model", "high")
    assert native["reasoning_effort"] == "high"
    assert native["messages"][2]["reasoning_content"] == "retained thought"
    assert native["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert len(native["messages"]) == 4


def test_resumed_sdk_system_messages_are_preserved():
    body = {"messages": [{"role": "system", "content": "restored system"},
                         {"role": "user", "content": "continue"}]}
    assert to_chat(body, "m", "high")["messages"][0] == {
        "role": "system", "content": "restored system"}


def test_bridge_usage_and_sse_are_not_duplicated():
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "done", "reasoning_content": "thought"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 80}}}
    message = from_chat(response, "model")
    assert message["usage"]["input_tokens"] == 20
    events = "".join(message_events(message))
    assert events.count("event: message_start") == 1
    assert events.count("event: message_stop") == 1
    assert '"thinking_delta"' in events


def test_incomplete_reasoning_is_budget_outcome_but_empty_success_is_invalid():
    data = {"choices": [{"finish_reason": "length", "message": {"reasoning_content": "thinking"}}]}
    assert from_chat(data, "m")["stop_reason"] == "max_tokens"
    data["choices"][0]["finish_reason"] = "stop"
    with pytest.raises(ValueError):
        from_chat(data, "m")


def fake_runner(tmp_path, monkeypatch, *, resolved=False):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    config = Config("m", str(tmp_path), str(tmp_path / "out"), "/runtime", methods=("baseline",))
    runner = BenchmarkRunner(config)
    journal = tmp_path / "parent.jsonl"
    journal.write_text('{"type":"run.finished"}\n')
    monkeypatch.setattr(runner, "initial", lambda task: ({"grade": {"resolved": resolved}}, journal))
    return runner


def test_shared_initial_success_skips_all_continuations(tmp_path, monkeypatch):
    runner = fake_runner(tmp_path, monkeypatch, resolved=True)
    monkeypatch.setattr(runner, "run_attempt", lambda *a, **k: pytest.fail("unexpected rollout"))
    summary = runner.run_task(SimpleNamespace(task_id="t"))
    assert summary["methods"]["baseline"]["extra_rollouts"] == 0


@pytest.mark.parametrize("success_at,expected", [(1, 1), (3, 3), (None, 7)])
def test_budget_and_stop_at_first_success(tmp_path, monkeypatch, success_at, expected):
    runner = fake_runner(tmp_path, monkeypatch)
    calls = []
    def attempt(task, method, name):
        calls.append(name)
        return {"name": name, "grade": {"resolved": len(calls) == success_at}}
    monkeypatch.setattr(runner, "run_attempt", attempt)
    summary = runner.run_task(SimpleNamespace(task_id="t"))
    assert len(calls) == expected
    assert summary["methods"]["baseline"]["extra_rollouts"] == expected


def test_grading_error_is_not_a_failure():
    with pytest.raises(RuntimeError, match="infrastructure"):
        result_resolved({"grade": {"resolved": False, "error": "verifier unavailable"}})


def test_settings_isolation_reaches_orchestrator(tmp_path):
    from swebench.fork_eval import run_attempt
    specs = []
    args = SimpleNamespace(slot="claude-code", model="m", runtime_bin="runtime/ash-runtime",
                           timeout=60, setting_sources=[])
    run_attempt(SimpleNamespace(run=specs.append), args, {}, name="parent", prompt="task",
                image="image", out_dir=tmp_path)
    assert specs[0].extra["setting_sources"] == []


def test_branch_wiring_uses_project_binding_and_independent_prefix(tmp_path, monkeypatch):
    from deepswe.branching.runner import CONTINUE
    runner = fake_runner(tmp_path, monkeypatch)
    source = SimpleNamespace(sha256="original-sha")
    prepared = {"resume_session_id": "independent-session", "cwd": str(tmp_path / "actor"),
                "manifest_path": "receipt.json"}
    runner.ev = SimpleNamespace(
        conversation_restore=lambda *args: ("cut-uuid", source),
        prepare_prefix=lambda *args: prepared, CLAUDE_PROJECTS_DIR=tmp_path)
    calls = []
    binding = {"workdir": "/app", "archive_sha256": "project-sha"}
    monkeypatch.setattr(runner, "materialize_project_binding",
                        lambda *args: dict(binding))
    monkeypatch.setattr(runner, "run_project_branch_attempt",
                        lambda *args, **kwargs: calls.append(kwargs))
    checkpoint = SimpleNamespace(step=4, snapshot_id="exact-snapshot", session_ckpt="parent-session")
    runner.branch(SimpleNamespace(task_id="task"), "shepherd", 1, {"step": 4}, checkpoint,
                  tmp_path / "parent.jsonl")
    assert calls[0]["binding_data"]["archive_sha256"] == "project-sha"
    assert calls[0]["prepared"]["resume_session_id"] == "independent-session"
    assert calls[0]["origin"]["conversation_cut"] == "cut-uuid"
    assert calls[0]["origin"]["whole_sandbox_snapshot_restored"] is False
    assert "snapshot_id" not in calls[0]["origin"]


def test_report_does_not_score_blocked_as_failure(tmp_path):
    from deepswe.branching.report import report
    save(tmp_path / "benchmark-manifest.json", {
        "tasks": ["pass", "blocked"], "config": {"methods": ["shepherd"], "max_rollouts": 8}})
    save(tmp_path / "task-summary" / "pass.json", {
        "methods": {"shepherd": {"status": "done", "resolved": True, "extra_rollouts": 1}}})
    save(tmp_path / "task-summary" / "blocked.json", {"methods": {"shepherd": {"status": "blocked"}}})
    result = report(tmp_path)
    assert result["methods"]["shepherd"]["observed_successes"] == 1
    assert result["methods"]["shepherd"]["success_rate"] is None
    assert result["methods"]["shepherd"]["pending_or_blocked_tasks"] == 1


def test_contract_checker_handles_supported_and_missing_sdk_fields(monkeypatch):
    from contracts.ci_check import Report, check_claude_sdk
    pytest.importorskip("claude_agent_sdk")
    report = Report()
    check_claude_sdk({"options_fields": ["cwd", "nonexistent_branchbench_field"]}, report)
    assert report.checks == 2
    assert len(report.failures) == 1


def test_bridge_health_and_rejects_non_experiment_keys(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from deepswe.branching.bridge import create_app
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "never-persist-me")
    with TestClient(create_app("m", tmp_path)) as client:
        assert client.get("/health").json()["effort"] == "high"
        assert client.post("/v1/messages", json={}).status_code == 401
        assert client.post("/v1/messages/count_tokens", json={"messages": []}).status_code == 200


@pytest.mark.parametrize("value", [0, -1, True, 2147483648, "600000"])
def test_invalid_api_timeout_is_rejected(value):
    with pytest.raises(ValueError, match="API timeout"):
        Config("model", "tasks", "output", "runtime", api_timeout_ms=value)


def test_buffered_bridge_sets_both_client_and_stream_watchdog_deadlines():
    from deepswe.branching.runner import actor_environment
    env = actor_environment(Config("model", "tasks", "output", "runtime"), "branchbench:t/shepherd/b01")
    assert env["API_TIMEOUT_MS"] == "1860000"
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "1860000"
    assert env["CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS"] == "1800000"
    assert env["ANTHROPIC_API_KEY"] == "branchbench:t/shepherd/b01"


def test_bridge_cancellation_retains_request_but_not_credentials(tmp_path, monkeypatch):
    import asyncio
    import httpx
    pytest.importorskip("fastapi")
    from deepswe.branching.bridge import create_app
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "never-persist-me")

    class CancelledClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): raise asyncio.CancelledError()

    class Request:
        headers = {"x-api-key": "branchbench:task/shepherd/branch"}
        async def json(self):
            return {"messages": [{"role": "user", "content": "task"}], "stream": False}

    monkeypatch.setattr(httpx, "AsyncClient", CancelledClient)
    app = create_app("model", tmp_path)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/messages")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(endpoint(Request()))
    audit = next((tmp_path / "provider-responses").glob("*.json")).read_text()
    assert "never-persist-me" not in audit
    assert json.loads(audit)["validation_error"] == "CancelledError"
    assert json.loads(audit)["request"]["messages"][0]["content"] == "task"
    assert json.loads((tmp_path / "actor-usage.jsonl").read_text())["validation_error"] == "CancelledError"
