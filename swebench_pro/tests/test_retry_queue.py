import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swebench_pro import retry_queue
from swebench_pro.recovery import durable_json, read_json


def manifest(count: int = 2) -> dict:
    return {"tasks": [{"index": index, "id": f"task-{index}"} for index in range(count)],
            "failure_policy": "isolated", "max_workers": 2, "max_infra_retries": 2,
            "retry_backoff_s": [30, 60], "actor_timeout_s": 3600,
            "total_tool_seconds": 1800, "consecutive_timeouts": 3}


def complete(queue, index, request, **state):
    return queue.finish(index, read_json(request)["attempt_id"], state, safe=True,
                        now=100, planner=lambda *args: {"recovery_action": "resume"})


def test_bounded_retry_does_not_retry_unresolved_or_accept_stale_result(tmp_path):
    queue = retry_queue.RetryQueue(tmp_path, manifest())
    first = queue.claim(0)
    healthy = queue.claim(1)
    assert complete(queue, 0, first, retryable=True, usage={"cost_usd": 1})
    assert queue.jobs[0]["phase"] == "retry_wait"
    assert queue.ready(129) == [] and queue.ready(130) == [0]
    with pytest.raises(ValueError, match="backoff"):
        queue.claim(0, now=129)
    assert queue.jobs[1]["phase"] == "running"
    assert complete(queue, 1, healthy, evidence_valid=True, resolved=False, retryable=True)
    assert queue.jobs[1]["phase"] == "completed"
    for number in (1, 2):
        request = queue.claim(0)
        assert request.parent.name == f"attempt-{number:03d}"
        assert not complete(queue, 0, first, evidence_valid=True, resolved=True)
        assert complete(queue, 0, request, retryable=True, usage={"cost_usd": 2})
    assert queue.jobs[0]["phase"] == "infra_failed"
    assert queue.ready(1000) == []
    result = read_json(tmp_path / "shard-000/worker.json")
    assert result["usage"]["cost_usd"] == 5
    assert result["retry_count"] == 2
    assert len(list((tmp_path / "shard-000/attempts").iterdir())) == 3
    assert not (tmp_path / "STOP_REQUEST.json").exists()


@pytest.mark.parametrize("safe,retryable,phase", [
    (False, True, "quarantined"), (True, False, "infra_failed")])
def test_unsafe_cleanup_or_nonretryable_error_never_requeues(tmp_path, safe, retryable, phase):
    queue = retry_queue.RetryQueue(tmp_path, manifest(1))
    request = queue.claim(0)

    def forbidden(*args):
        pytest.fail("Recovery planner called for unsafe or nonretryable job")

    queue.finish(0, read_json(request)["attempt_id"], {"retryable": retryable},
                 safe=safe, now=100, planner=forbidden)
    assert queue.jobs[0]["phase"] == phase
    assert retry_queue.RetryQueue(tmp_path, manifest(1)).ready(1000) == []


def test_grading_retry_does_not_duplicate_actor_usage(tmp_path):
    queue = retry_queue.RetryQueue(tmp_path, manifest(1))
    request = queue.claim(0)
    queue.finish(0, read_json(request)["attempt_id"], {"retryable": True, "usage": {"cost_usd": 3}},
                 safe=True, now=0, planner=lambda *args: {"recovery_action": "regrade", "regrade": {"saved": True}})
    request = queue.claim(0)
    complete(queue, 0, request, evidence_valid=True, resolved=False, usage={"cost_usd": 3})
    assert read_json(tmp_path / "shard-000/worker.json")["usage"] == {"cost_usd": 3}


def test_retry_cap_cannot_be_unbounded(tmp_path):
    with pytest.raises(ValueError, match="between zero and two"):
        retry_queue.RetryQueue(tmp_path, {**manifest(), "max_infra_retries": 3})


def write_attempt(tmp_path, events, status="error"):
    directory = tmp_path / "task-0"
    directory.mkdir()
    (directory / "parent.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    durable_json(directory / "execution.json", {"status": status})
    return directory


@pytest.mark.parametrize("status", ["completed", "timeout"])
def test_grader_failure_reuses_final_snapshot_even_after_actor_budget_exhausted(tmp_path, monkeypatch, status):
    from swebench_pro import resume

    monkeypatch.setattr(resume, "disk_manifest_valid", lambda identifier: identifier == "final")
    directory = write_attempt(tmp_path, [{"type": "checkpoint.captured", "snapshot_id": "final"}], status)
    result = retry_queue.retry_payload(tmp_path, manifest(1), {"id": "task-0"}, tmp_path,
                                      {"retry_kind": "grading", "budget_exhausted": True})
    assert result["continuation"] is None and result["recovery_action"] == "regrade"
    assert result["regrade"]["snapshot_id"] == "final"
    assert result["regrade"]["execution_sha256"] == retry_queue.digest(directory / "execution.json")


def test_actor_retry_selects_paired_prefix_and_carries_all_spent_budget(tmp_path, monkeypatch):
    from harness import rollback
    from swebench import fork_eval
    from swebench_pro import resume

    events = [{"type": "run.started", "ts": "2026-09-11T00:00:00Z"},
              {"type": "pro.tool_budget", "tool_seconds": 250, "consecutive_timeouts": 1},
              {"type": "pro.tool_budget", "effective_timeout": 450, "ts": "2026-09-11T00:01:00Z"}]
    write_attempt(tmp_path, events)
    monkeypatch.setattr(resume, "disk_manifest_valid", lambda identifier: True)
    monkeypatch.setattr(rollback, "turn_branch_checkpoints", lambda path: {
        1: SimpleNamespace(snapshot_id="paired", session_ckpt="native"),
        2: SimpleNamespace(snapshot_id="unpaired", session_ckpt="missing")})
    monkeypatch.setattr(fork_eval, "conversation_restore", lambda path, step, session:
                        ("cut", SimpleNamespace(sha256="prefix-hash", transcript=Path("native.jsonl"))) if step == 1 else None)
    item = {"id": "task-0", "continuation": {"actor_seconds_charged": 300, "tool_seconds_charged": 100}}
    result = retry_queue.retry_payload(tmp_path, manifest(1), item, tmp_path,
                                      {"retry_kind": "actor", "finished_at": "2026-09-11T00:02:00Z"})
    continuation = result["continuation"]
    assert continuation["snapshot_id"] == "paired" and continuation["step"] == 1
    assert continuation["actor_seconds_charged"] == 420
    assert continuation["actor_seconds_remaining"] == 3180
    assert continuation["tool_seconds_charged"] == 310
    assert continuation["consecutive_timeouts"] == 1


def test_actor_retry_without_safe_prefix_never_starts_fresh(tmp_path, monkeypatch):
    from harness import rollback

    write_attempt(tmp_path, [{"type": "run.started", "ts": "2026-09-11T00:00:00Z"}])
    monkeypatch.setattr(rollback, "turn_branch_checkpoints", lambda path: {})
    with pytest.raises(ValueError, match="refusing a fresh parent retry"):
        retry_queue.retry_payload(tmp_path, manifest(1), {"id": "task-0"}, tmp_path,
                                 {"finished_at": "2026-09-11T00:01:00Z"})


def test_actor_retry_rejects_incomplete_checkpoint_using_real_ledger(tmp_path, monkeypatch):
    from swebench import fork_eval
    from swebench_pro import resume

    events = [{"type": "run.started", "ts": "2026-09-11T00:00:00Z"},
              {"type": "checkpoint.policy", "pairing": "call-id-v1"}]
    for step in (1, 2):
        events.extend([{"type": "tool.started", "step": step, "call_id": f"call-{step}"},
                       {"type": "checkpoint.captured", "step": step, "call_id": f"call-{step}",
                        "pairing": "call-id-v1", "prefix_complete": step == 1,
                        "snapshot_id": f"disk-{step}", "session_ckpt": "native", "reason": "captured"}])
    write_attempt(tmp_path, events)
    monkeypatch.setattr(resume, "disk_manifest_valid", lambda identifier: True)

    def restore(path, step, session):
        assert step == 1
        return "cut", SimpleNamespace(sha256="prefix-hash", transcript=Path("native.jsonl"))

    monkeypatch.setattr(fork_eval, "conversation_restore", restore)
    result = retry_queue.retry_payload(tmp_path, manifest(1), {"id": "task-0"}, tmp_path,
                                      {"finished_at": "2026-09-11T00:01:00Z"})
    assert result["continuation"]["snapshot_id"] == "disk-1"


@pytest.mark.parametrize("events,state", [
    ([], {"budget_exhausted": True}),
    ([{"type": "pro.tool_budget", "tool_seconds": 1800}], {}),
    ([{"type": "pro.tool_budget", "tool_seconds": 100, "consecutive_timeouts": 3}], {}),
    ([], {"finished_at": "2026-09-11T01:01:00Z"})])
def test_actor_recovery_never_resets_exhausted_budget(tmp_path, events, state):
    write_attempt(tmp_path, [{"type": "run.started", "ts": "2026-09-11T00:00:00Z"}, *events])
    with pytest.raises(ValueError, match="budget exhausted"):
        retry_queue.retry_payload(tmp_path, manifest(1), {"id": "task-0"}, tmp_path,
                                 {"finished_at": "2026-09-11T00:01:00Z", **state})


@pytest.mark.parametrize("uncertain", [False, True])
def test_controller_isolates_fault_cleans_only_owned_and_keeps_other_worker(tmp_path, monkeypatch, uncertain):
    config = manifest()
    config["retry_backoff_s"] = [0, 0]
    durable_json(tmp_path / "manifest.json", config)
    monkeypatch.setattr(retry_queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(retry_queue.shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(retry_queue, "process_identity", lambda pid: str(pid))
    monkeypatch.setattr(retry_queue, "retry_payload", lambda *args: {"recovery_action": "resume"})
    inventory = [{"sandboxID": "retained"}, {"sandboxID": "owned"}]
    launched = []
    deleted = []
    polls = {0: 0, 1: 0}

    def spawn(arguments, **kwargs):
        request_path = Path(arguments[-1])
        request = read_json(request_path)
        index = request["index"]
        launched.append(index)
        assert kwargs["start_new_session"] is True
        durable_json(request_path.parent / "worker.json", {
            "task": f"task-{index}", "attempt_id": request["attempt_id"], "finished_at": "now",
            "retryable": index == 0, "evidence_valid": index == 1, "resolved": False,
            "creation_inflight": 0, "creation_uncertain": uncertain and index == 0})
        durable_json(request_path.parent / "sandboxes.json", {"owned": ["owned"] if index == 0 else []})

        def poll():
            polls[index] += 1
            return None if index == 1 and polls[index] < 4 else 0

        return SimpleNamespace(pid=100 + index, poll=poll)

    def delete(identifier):
        deleted.append(identifier)
        inventory[:] = [entry for entry in inventory if entry["sandboxID"] != identifier]

    assert retry_queue.controller(tmp_path, api=lambda route: inventory, delete=delete, popen=spawn,
                                  sleep=lambda seconds: None, process_group_alive=lambda pid: False) == 2
    assert launched.count(0) == (1 if uncertain else 3) and launched.count(1) == 1
    assert deleted == ([] if uncertain else ["owned"])
    assert inventory[0]["sandboxID"] == "retained"
    assert read_json(tmp_path / "shard-001/job.json")["phase"] == "completed"
    assert read_json(tmp_path / "shard-000/job.json")["phase"] == ("quarantined" if uncertain else "infra_failed")
    assert not (tmp_path / "STOP_REQUEST.json").exists()


def test_batch_routes_isolated_policy_to_retry_controller(tmp_path, monkeypatch):
    from swebench_pro import batch

    durable_json(tmp_path / "manifest.json", manifest())
    monkeypatch.setattr(retry_queue, "controller", lambda root: 17)
    assert batch.controller(tmp_path) == 17


def test_committed_result_is_republished_after_controller_crash(tmp_path, monkeypatch):
    queue = retry_queue.RetryQueue(tmp_path, manifest(1))
    request = queue.claim(0)

    def crash(*args):
        raise OSError("publication interrupted")

    monkeypatch.setattr(queue, "publish", crash)
    with pytest.raises(OSError, match="publication interrupted"):
        complete(queue, 0, request, evidence_valid=True, resolved=False)
    restored = retry_queue.RetryQueue(tmp_path, manifest(1))
    assert restored.jobs[0]["phase"] == "completed"
    assert read_json(tmp_path / "shard-000/worker.json")["evidence_valid"] is True
    assert restored.ready(1000) == []


@pytest.mark.parametrize("problem", ["api", "disk"])
def test_shared_fault_pauses_admission_without_stopping_existing_jobs(tmp_path, monkeypatch, problem):
    durable_json(tmp_path / "manifest.json", manifest(1))
    monkeypatch.setattr(retry_queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(retry_queue.shutil, "disk_usage", lambda root:
                        SimpleNamespace(free=0 if problem == "disk" else 1024**4))

    def api(route):
        if problem == "api":
            raise OSError("shared backend unavailable")
        return []

    def sleep(seconds):
        assert read_json(tmp_path / "controller.json")["shared_pause"]
        durable_json(tmp_path / "STOP_REQUEST.json", {"reason": "test drain"})

    assert retry_queue.controller(tmp_path, api=api, sleep=sleep,
                                  popen=lambda *args, **kwargs: pytest.fail("admitted during shared failure")) == 2
    assert read_json(tmp_path / "shard-000/job.json")["phase"] == "queued"


def test_controller_does_not_redispatch_unrecorded_process(tmp_path, monkeypatch):
    config = manifest(1)
    durable_json(tmp_path / "manifest.json", config)
    retry_queue.RetryQueue(tmp_path, config).claim(0)
    monkeypatch.setattr(retry_queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(retry_queue.shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**4))
    assert retry_queue.controller(tmp_path, api=lambda route: [],
                                  popen=lambda *args, **kwargs: pytest.fail("unknown old process was retried")) == 2
    assert read_json(tmp_path / "shard-000/job.json")["phase"] == "quarantined"


def test_uncommitted_attempt_directory_quarantines_only_its_job(tmp_path):
    config = manifest()
    retry_queue.RetryQueue(tmp_path, config)
    (tmp_path / "shard-000/attempts/attempt-000").mkdir(parents=True)
    restored = retry_queue.RetryQueue(tmp_path, config)
    assert restored.jobs[0]["phase"] == "quarantined"
    assert restored.ready(0) == [1]
    assert read_json(tmp_path / "shard-000/worker.json")["status"] == "quarantined"


def test_parent_lookup_uses_published_attempt_not_first_old_journal(tmp_path):
    from swebench.fork_eval import existing_parent

    config = manifest(1)
    durable_json(tmp_path / "manifest.json", config)
    queue = retry_queue.RetryQueue(tmp_path, config)
    request = queue.claim(0)
    old = request.parent / "task-0/parent.jsonl"
    old.parent.mkdir()
    old.write_text("old")
    complete(queue, 0, request, retryable=True)
    request = queue.claim(0)
    assert existing_parent(str(tmp_path), "task-0") is None
    latest = request.parent / "task-0/parent.jsonl"
    latest.parent.mkdir()
    latest.write_text("new")
    complete(queue, 0, request, evidence_valid=True, resolved=False)
    assert existing_parent(str(tmp_path), "task-0") == latest


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_creation_tracking_includes_unadopted_sandbox_and_unknown_response(asynchronous, fails):
    import asyncio
    import threading

    import httpx

    from swebench_pro.worker import track_creations

    state = {"creation_inflight": 0, "creation_uncertain": False}
    owned = []

    def handler(request):
        if request.url.path not in ("/sandboxes", "/sandboxes-cold"):
            assert state["creation_inflight"] == 0
            return httpx.Response(200, json={})
        assert state["creation_inflight"] == 1
        if fails:
            raise httpx.ReadTimeout("unknown create outcome")
        return httpx.Response(201, json={"sandboxID": "unadopted"})

    transport = httpx.MockTransport(handler)

    async def exercise_async():
        async with httpx.AsyncClient(transport=transport, base_url="http://agentenv") as client:
            await client.post("/snapshot")
            await client.post("/sandboxes")

    def exercise_sync():
        with httpx.Client(transport=transport, base_url="http://agentenv") as client:
            client.post("/snapshot")
            client.post("/sandboxes-cold")

    original = (httpx.Client.post, httpx.AsyncClient.post)
    with track_creations(state, threading.RLock(), state.update, owned.append):
        if fails:
            with pytest.raises(httpx.ReadTimeout):
                asyncio.run(exercise_async()) if asynchronous else exercise_sync()
        else:
            asyncio.run(exercise_async()) if asynchronous else exercise_sync()
    assert (httpx.Client.post, httpx.AsyncClient.post) == original
    assert owned == ([] if fails else ["unadopted"])
    assert state == {"creation_inflight": 0, "creation_uncertain": fails}
