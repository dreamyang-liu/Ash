from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from harness.tests.test_assistant_turn import assistant_turn
from runstore.api import create_app
from runstore.branch_guidance import branch_spec, execution_guidance
from runstore.specs import JobSpec, validate_continuation


def point(tmp_path):
    old = assistant_turn("pwd", identifier="existing")
    rows = [
        {"type": "mini.session", "format": "ash-mini-v1", "mini_version": "2.4.6", "session_id": "native"},
        {"type": "mini.message", "message": {"role": "user", "content": "original task"}},
        {"type": "mini.message", "message": old},
        {"type": "mini.message", "message": {"role": "tool", "tool_call_id": "existing", "content": "/app"}},
        {"type": "mini.turn", "turn_id": "turn", "call_ids": ["existing"]},
    ]
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path = tmp_path / "native.jsonl"
    path.write_bytes(data)
    return {"id": "point", "job_id": "parent", "snapshot_id": "snapshot",
            "tool_depth": 1, "message_step": 1, "native": {
                "slot": "mini-swe-agent", "path": str(path), "byte_length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "session_id": "native"}}


def api(tmp_path, parent=None):
    parent = parent or JobSpec("rollout", {
        "prompt": "original task", "slot": "mini-swe-agent", "sandbox_image": "image",
        "extra": {"mini": {"environment": {"timeout": 60}}},
    }, profile="mini", max_infra_retries=0)
    selected = point(tmp_path)

    class Store:
        submissions = []

        def get(self, job_id):
            assert job_id == "parent"
            return {"request": parent.validate()}

        def submit(self, job, key):
            self.submissions.append(job.validate())
            return {"id": "child", "request": job.validate()}

    store = Store()
    index = SimpleNamespace(get_point=lambda _: selected, valid=lambda _: True)
    return TestClient(create_app(store, "fixture", index=index)), store, selected


def post(client, overrides):
    return client.post("/v1/jobs/parent/branch",
        headers={"Authorization": "Bearer fixture", "Idempotency-Key": "branch"},
        json={"point_id": "point", "overrides": overrides})


def test_mini_branch_defaults_to_assistant_turn_and_keeps_original_task(tmp_path):
    client, store, _ = api(tmp_path)
    turn = assistant_turn()
    with client:
        response = post(client, {"assistant_turn": turn})
    assert response.status_code == 202, response.text
    child = store.submissions[0]
    assert child["parent_point"] == "point"
    assert child["spec"]["prompt"] == "original task"
    assert child["spec"]["extra"]["assistant_turn"] == turn
    assert child["spec"]["extra"]["branch_guidance"] == "assistant-turn"


@pytest.mark.parametrize("kind", ["missing", "non-bash", "duplicate-id", "prompt", "extra"])
def test_invalid_guidance_rejected_before_queueing(tmp_path, kind):
    client, store, _ = api(tmp_path)
    overrides = {"assistant_turn": assistant_turn()}
    if kind == "missing":
        overrides = {}
    elif kind == "non-bash":
        overrides["assistant_turn"]["tool_calls"][0]["function"]["name"] = "apply_patch"
    elif kind == "duplicate-id":
        overrides["assistant_turn"]["tool_calls"][0]["id"] = "existing"
    elif kind == "prompt":
        overrides["prompt"] = "private user hint"
    else:
        overrides["extra"] = {"native_prefix": {"path": "/other"}}
    with client:
        response = post(client, overrides)
    assert response.status_code == 422, response.text
    assert not store.submissions
    if kind == "non-bash":
        assert "only bash is allowed" in response.text


@pytest.mark.parametrize("mode", ["user-hint", "none"])
def test_branch_of_branch_does_not_reinject_previous_seed(tmp_path, mode):
    original = assistant_turn(identifier="old-seed")
    parent = JobSpec("rollout", {
        "prompt": "original task", "slot": "mini-swe-agent",
        "extra": {"branch_guidance": "assistant-turn", "assistant_turn": original},
    }, profile="mini", parent_point="ancestor")
    client, store, _ = api(tmp_path, parent)
    overrides = {"branch_guidance": mode}
    if mode == "user-hint":
        overrides["prompt"] = "new explicit hint"
    with client:
        response = post(client, overrides)
    assert response.status_code == 202, response.text
    extra = store.submissions[0]["spec"]["extra"]
    assert "assistant_turn" not in extra
    assert extra.get("resume_without_hint", False) is (mode == "none")
    assert parent.spec["extra"]["assistant_turn"] == original


def test_guidance_does_not_allow_changing_inherited_execution_configuration():
    parent = JobSpec("rollout", {"prompt": "task", "slot": "mini-swe-agent",
                     "extra": {"mini": {"environment": {"timeout": 60}}}}).validate()
    spec = branch_spec(parent["spec"], {"assistant_turn": assistant_turn()}, "mini-swe-agent")
    child = JobSpec("rollout", spec, parent_point="point").validate()
    validate_continuation(parent, child)
    child["spec"]["extra"]["mini"]["environment"]["timeout"] = 999
    with pytest.raises(ValueError, match="retain its source"):
        validate_continuation(parent, child)


def test_seed_is_rejected_on_root_and_removed_on_same_job_retry(tmp_path):
    spec = {"prompt": "task", "slot": "mini-swe-agent",
            "extra": {"branch_guidance": "assistant-turn", "assistant_turn": assistant_turn()}}
    with pytest.raises(ValueError, match="parent_point"):
        JobSpec("rollout", spec).validate()
    recovered = point(tmp_path)
    extra, mode, origin = execution_guidance(spec, recovered, "parent")
    assert mode == "none" and extra["resume_without_hint"] is True
    assert "assistant_turn" not in extra and origin["recovery_kind"] == "retry"


def test_unmarked_stored_mini_continuation_keeps_legacy_hint_delivery(tmp_path):
    spec = {"prompt": "legacy hint", "slot": "mini-swe-agent"}
    JobSpec("rollout", spec, parent_point="point").validate()
    extra, mode, origin = execution_guidance(spec, point(tmp_path), "child")
    assert mode == "user-hint" and extra["branch_guidance"] == "user-hint"
    assert "assistant_turn" not in extra and not extra.get("resume_without_hint")


@pytest.mark.skipif(__import__("importlib.util").util.find_spec("minisweagent") is None,
                    reason="mini dependency required")
def test_worker_executes_seed_and_exports_real_observation_without_user_hint(tmp_path, monkeypatch):
    from harness.orchestrator.run import Orchestrator
    from harness.tests.test_mini_swe import model_server, owned_filesystem, reply, spec
    from runstore.child import execute
    from runstore.native import index_native
    from runstore.tests.test_assistant_turn_branch import restore_files
    from runstore.tests.test_mini_native import parent_run

    memory, parent, points = parent_run(tmp_path / "parent", monkeypatch)
    cut = points[0]
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = restore_files(memory, cut.snapshot_id, child_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    recovered = {"id": "selected", "job_id": "parent-job", "snapshot_id": cut.snapshot_id,
                 "tool_depth": cut.tool_depth, "message_step": cut.message_step, "native": cut.native}
    turn = assistant_turn("printf once >> seed-count; printf seeded > answer; cat answer")
    with model_server([reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
                       reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        parent_spec = asdict(spec(tmp_path / "ignored", url, prompt="USER_HINT_MUST_NOT_APPEAR"))
        parent_spec["extra"].pop("native_home")
        effective = branch_spec(parent_spec, {"assistant_turn": turn}, "mini-swe-agent")
        result = execute({"kind": "rollout", "job_id": "child-job", "attempt_id": "child-attempt",
                          "effective_spec": effective, "profile_config": {}, "recovery": recovered}, child_dir)
        assert result["status"] == "completed", result
        assert len(requests) == 2
        native = child_dir / "native-home" / (result["native_session_id"] + ".jsonl")
        point_after_seed = index_native(child_dir / "trajectory.jsonl", native, "mini-swe-agent",
                                       result["native_session_id"], inherited_native=cut.native)[0]
        retry_dir = tmp_path / "retry"
        retry_dir.mkdir()
        retry_memory = restore_files(child, point_after_seed.snapshot_id, retry_dir / "sandbox")
        monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(retry_memory))
        retry_point = {"id": "same-job-recovery", "job_id": "child-job",
                       "snapshot_id": point_after_seed.snapshot_id,
                       "tool_depth": point_after_seed.tool_depth, "message_step": point_after_seed.message_step,
                       "native": point_after_seed.native}
        retried = execute({"kind": "rollout", "job_id": "child-job", "attempt_id": "retry-attempt",
                           "effective_spec": effective, "profile_config": {}, "recovery": retry_point}, retry_dir)
        assert retried["status"] == "completed", retried
        assert (retry_memory.root / "seed-count").read_text() == "once"
        assert retried["training_origin"]["recovery_kind"] == "retry"
        assert retried["training_origin"]["branch_guidance"] == "none"
    assert result["status"] == "completed", result
    assert not result.get("training_export_error"), result
    assert len(requests) == 4  # Two model calls per execution; no extra call for the reviewer seed.
    assert requests[0]["messages"][-2]["content"] == turn["content"]
    assert json.loads(requests[0]["messages"][-1]["content"])["output"] == "seeded"
    assert "USER_HINT_MUST_NOT_APPEAR" not in json.dumps(requests)
    assert "echo late" not in json.dumps(requests)
    assert any(m.get("content") == turn["content"] for m in result["training_messages"])
    assert result["training_origin"]["branch_guidance"] == "assistant-turn"
    assert result["training_origin"]["assistant_turn_call_ids"] == [turn["tool_calls"][0]["id"]]
    assert result["rollout_usage"]["model_calls"] == 2
