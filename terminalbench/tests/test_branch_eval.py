from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.models.task.task import Task
from swebench import structured_review
from swebench.assistant_branch import ASSISTANT_REVIEW_PROMPT
from swebench.review_transport import ReviewTransport
from terminalbench import branch_eval


def _result(path: Path, reward, *, error=None):
    path.write_text(json.dumps({"finished_at": "2026-09-23T00:00:00Z",
                                "exception_info": error,
                                "verifier_result": {"rewards": {"reward": reward}}}))


def test_only_a_valid_official_harbor_grade_can_resolve_a_branch(tmp_path):
    result = tmp_path / "result.json"
    _result(result, 1)
    grade = branch_eval._read_grade(result)
    assert grade.resolved and grade.reward == 1
    assert "official Harbor reward=1" in grade.summary()
    _result(result, 0)
    assert not branch_eval._read_grade(result).resolved
    _result(result, None)
    with pytest.raises(RuntimeError, match="missing or invalid"):
        branch_eval._read_grade(result)
    _result(result, 1, error="verifier failed")
    with pytest.raises(RuntimeError, match="verifier failed"):
        branch_eval._read_grade(result)


def test_root_and_branch_trials_use_the_same_official_task(tmp_path):
    task_dir = Path(__file__).parent / "agentenv_fixtures" / "shell-task"
    task = Task(task_dir)
    runtime = tmp_path / "runtime"
    runtime.touch()
    args = SimpleNamespace(
        task_dir=task_dir, dataset_digest="sha256:pinned", output=tmp_path,
        model="fixture", model_endpoint="http://127.0.0.1:18252",
        model_key_env="TEST_KEY", max_output_tokens=64000, max_turns=300,
        runtime_bin=runtime, sandbox_ttl=36000, server_url="http://127.0.0.1:18000",
        api_key_file=None, image_registry="localhost:5000",
    )
    root = branch_eval._trial_config(args, task, name="parent", branch_context=None)
    context = tmp_path / "selected.json"
    branch = branch_eval._trial_config(args, task, name="r1b1", branch_context=context)
    assert root.task.path == branch.task.path == task_dir.resolve()
    assert root.agent.kwargs["actor_timeout_s"] == branch.agent.kwargs["actor_timeout_s"] == task.config.agent.timeout_sec
    assert root.agent.import_path == "terminalbench.agentenv_mini:AgentENVMini"
    assert branch.agent.import_path == "terminalbench.branching:BranchMini"
    assert branch.environment.import_path == "terminalbench.branching:SnapshotEnvironment"
    assert branch.agent.kwargs["branch_context"] == branch.environment.kwargs["branch_context"] == str(context)


def test_preparation_membership_requires_the_exact_task_and_dataset(tmp_path):
    task_dir = Path(__file__).parent / "agentenv_fixtures" / "shell-task"
    task = Task(task_dir)
    digest = hashlib.sha256((task_dir / "task.toml").read_bytes()).hexdigest()
    path = tmp_path / "manifest.json"
    row = {"name": task.name, "path": str(task_dir.resolve()), "task_toml_sha256": digest}
    path.write_text(json.dumps({"dataset_digest": "sha256:pinned", "tasks": [row]}))
    args = SimpleNamespace(preparation_manifest=path, dataset_digest="sha256:pinned", task_dir=task_dir)
    branch_eval._verify_prepared_task(args, task, digest)
    args.dataset_digest = "sha256:different"
    with pytest.raises(ValueError, match="digest"):
        branch_eval._verify_prepared_task(args, task, digest)
    args.dataset_digest = "sha256:pinned"
    row["task_toml_sha256"] = "changed"
    path.write_text(json.dumps({"dataset_digest": args.dataset_digest, "tasks": [row]}))
    with pytest.raises(ValueError, match="definition changed"):
        branch_eval._verify_prepared_task(args, task, digest)


def test_review_transport_keeps_keys_out_of_receipts_and_rejects_model_switch(tmp_path, monkeypatch):
    transport = ReviewTransport(tmp_path, model="fixture",
                                model_endpoint="http://127.0.0.1:18252",
                                api_key_env="TEST_KEY", max_output_tokens=1000)
    monkeypatch.setenv("TEST_KEY", "super-secret")
    with pytest.raises(ValueError, match="differs"):
        transport("other-model", "prompt")
    monkeypatch.setattr("swebench.review_transport.structured_review.request_format",
                        lambda prompt: ("analyst", prompt, {}))
    monkeypatch.setattr("swebench.review_transport.structured_review.response_text",
                        lambda kind, value: value["choices"][0]["message"]["content"])
    requests = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "valid"}}]}

    def post(url, **kwargs):
        requests.append((url, kwargs))
        return Response()

    monkeypatch.setattr("swebench.review_transport.httpx.post", post)
    assert transport("fixture", "prompt") == "valid"
    assert requests[0][0].endswith("/v1/chat/completions")
    assert requests[0][1]["headers"]["Authorization"] == "Bearer super-secret"
    assert "super-secret" not in (tmp_path / "review-001.json").read_text()


def test_review_transport_preserves_fenced_object_arguments(tmp_path, monkeypatch):
    transport = ReviewTransport(tmp_path, model="fixture",
                                model_endpoint="http://127.0.0.1:18252",
                                api_key_env="TEST_KEY", max_output_tokens=1000)
    monkeypatch.setenv("TEST_KEY", "local-test")
    plan = {"synthesis": "Continue from the checked point.",
            "branches": [{"name": "repair", "base": "parent", "branch_step": 1,
                          "why": "Test this one command", "assistant_turn": {
                              "role": "assistant", "content": "", "tool_calls": [{
                                  "id": "reviewer-call", "type": "function", "function": {
                                      "name": "bash", "arguments": {"command": "printf answer"}}}]}}]}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"role": "assistant",
                                               "content": "```branch-plan\n" + json.dumps(plan) + "\n```"}}]}

    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("swebench.review_transport.httpx.post", post)
    prompt = ASSISTANT_REVIEW_PROMPT.format(problem="task", reports="[]", count_rule="At most 1.")
    parsed = structured_review.extract_branch_plan(transport("fixture", prompt))
    assert json.loads(parsed["branches"][0]["assistant_turn"]["tool_calls"][0]["function"]["arguments"]) == {"command": "printf answer"}
    assert '"arguments": {"command"' in requests[0]["messages"][0]["content"]
