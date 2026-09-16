from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from rl_driver.driver import Driver
from rl_driver.ledger import Ledger
from rl_driver.miles import MilesAdapter, internal_id
from rl_driver.policy import first_available_recovery, load_branch_policy
from rl_driver.protocol import RolloutGroupRequest, Trajectory
from rl_driver.server import create_app
from rl_driver.tests.test_driver import peer


def miles_request():
    # Verbatim request returned by the pinned haixin branch's HTTP e2e fixture.
    return json.loads((Path(__file__).parent / "fixtures/miles-request.json").read_text())


def config(body, *, slot="codex"):
    return {"environment_catalog": {"environments": [{**body["environment_ref"], "spawn_ref": "prepared-final"}]},
            "profile": slot, "run_defaults": {"slot": slot, "model": "served-model"},
            "resources": {"standard": {"cpu": 2, "memory_mb": 8192}}}


def recorded_state(body, output=90, version="42"):
    inputs = body["prompt_token_ids"]
    return {"metadata": {"accumulated_token_ids": [*inputs, output]}, "records": [{
        "request": {"input_ids": inputs, "messages": [{"role": "user", "content": body["prompt"]}]},
        "response": {"id": f"response-{output}", "choices": [{"message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop", "meta_info": {"weight_version": version, "output_token_logprobs": [[-0.3, output]]}}]},
    }]}


def branched_state(body, parent_output=90, child_output=91, version="42"):
    prompt = body["prompt_token_ids"]
    parent_record = {
        "request": {"input_ids": prompt, "messages": [{"role": "user", "content": body["prompt"]}]},
        "response": {"id": "response-parent", "choices": [{
            "message": {"role": "assistant", "content": "parent"},
            "finish_reason": "tool_calls", "meta_info": {
                "weight_version": version,
                "output_token_logprobs": [[-0.2, parent_output]],
            },
        }]},
    }
    child_input = [*prompt, parent_output, 77]
    child_record = {
        "request": {"input_ids": child_input, "messages": [
            {"role": "user", "content": body["prompt"]},
            {"role": "assistant", "content": "parent"},
            {"role": "tool", "content": "tool result"},
        ]},
        "response": {"id": "response-child", "choices": [{
            "message": {"role": "assistant", "content": "child"},
            "finish_reason": "stop", "meta_info": {
                "weight_version": version,
                "output_token_logprobs": [[-0.3, child_output]],
            },
        }]},
    }
    return {
        "records": [parent_record, child_record],
        "metadata": {
            "accumulated_token_ids": [*child_input, child_output],
            "tree": {"nodes": [
                {"id": 0, "parent": None, "response_id": "response-parent",
                 "completion_span": [len(prompt), len(prompt) + 1]},
                {"id": 1, "parent": 0, "response_id": "response-child",
                 "completion_span": [len(child_input), len(child_input) + 1]},
            ]},
            "tree_records": {
                "0": {"record": parent_record, "token_ids": [*prompt, parent_output]},
                "1": {"record": child_record, "token_ids": [*child_input, child_output]},
            },
        },
    }


def finish(queue, body, *, missing=False):
    for index, job_id in enumerate(queue.jobs):
        queue.finish(job_id)
        queue.events[job_id] = [{"seq": 1, "type": "rollout.usage", "model_calls": 1, "tool_calls": 0}]
        if not missing:
            queue.events[job_id].append({"seq": 2, "type": "rollout.session_state", "state": recorded_state(body, index + 90)})


def test_original_wire_request_to_runs_and_original_wire_response(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    with TestClient(create_app(driver, None, miles=adapter, background=False)) as http:
        environments = http.get("/rollout-environments").json()
        assert environments == {"protocol_version": "ash-rollout-v2", "environments": [body["environment_ref"]]}
        response = http.post("/rollout-groups", json=body)
        assert response.status_code == 202
        assert response.json() == {"protocol_version": "ash-rollout-v2", "rollout_job_id": body["rollout_job_id"], "status": "queued"}
        driver.tick()
        assert len(queue.jobs) == 2
        submissions = [r[2] for r in queue.requests if r[0] == "POST"]
        assert [r["context"]["sample_slot_id"] for r in submissions] == [s["sample_slot_id"] for s in body["sample_slots"]]
        assert all(r["spec"]["sandbox_image"] == "prepared-final" for r in submissions)
        assert all(r["spec"]["sandbox_resources"] == {"cpu": 2, "memory_mb": 8192} for r in submissions)
        assert sum(r["spec"]["extra"]["rollout_contract"]["max_model_calls"] for r in submissions) == body["budgets"]["max_model_calls"]
        assert sum(r["spec"]["extra"]["rollout_contract"]["max_tool_calls"] for r in submissions) == body["budgets"]["max_tool_calls"]
        assert all(r["spec"]["extra"]["rollout_contract"]["capture_recovery_points"] is False
                   for r in submissions)
        assert all(r["spec"]["extra"]["rollout_contract"]["capture_final_snapshot"] is False
                   for r in submissions)
        assert all(r["max_infra_retries"] == 0 for r in submissions)
        adapter.config["resources"] = {}  # retry must retain the originally planned execution
        assert http.post("/rollout-groups", json=body).json()["status"] == "running"
        assert len(queue.jobs) == 2
        finish(queue, body)
        driver.tick()
        result = http.get("/rollout-groups/" + body["rollout_job_id"]).json()
        assert result["protocol_version"] == "ash-rollout-v2"
        assert result["status"] == "completed"
        assert result["actual_samples"] == 2
        assert result["consumed_budget"] == {"model_calls": 2, "tool_calls": 0}
        assert [t["sample_slot_id"] for t in result["trajectories"]] == [s["sample_slot_id"] for s in body["sample_slots"]]
        for trajectory in result["trajectories"]:
            parsed = Trajectory.from_dict(trajectory)
            assert parsed.token_ids[:parsed.prompt_length] == body["prompt_token_ids"]
        queue.events.clear()  # result is persisted before a caller consumes it
        assert http.get("/rollout-groups/" + body["rollout_job_id"]).json() == result
        persisted = driver.ledger.get(internal_id(body["rollout_job_id"]))["document"]
        assert persisted["miles_result"] == result
        assert len(persisted["profiling_records"]) == result["actual_samples"]
        assert all(record["protocol_version"] == "ash-rollout-v2"
                   for record in persisted["profiling_records"])
        assert http.delete("/rollout-groups/" + body["rollout_job_id"]).json()["status"] == "completed"
    restored = MilesAdapter(Driver(backend, Ledger(tmp_path / "ledger")), config(body))
    assert restored.get(body["rollout_job_id"]) == result


def test_v2_grading_requests_one_final_snapshot_without_recovery_capture(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    cfg = config(body)
    cfg["tasks"] = {body["task_id"]: {
        "environment_ref": body["environment_ref"],
        "grade": {"profile": "grade", "spec": {
            "benchmark": "swebench-verified",
            "instance_id": body["task_id"],
            "dataset_path": "/worker/tasks.json",
            "dataset_sha256": "fixture",
            "grader_revision": "fixture",
        }},
    }}
    adapter = MilesAdapter(Driver(backend, Ledger(tmp_path / "ledger")), cfg)
    adapter.submit(body)
    adapter.driver.tick()

    submissions = [row[2] for row in queue.requests if row[:2] == ("POST", "/v1/jobs")]
    assert submissions
    for submission in submissions:
        contract = submission["spec"]["extra"]["rollout_contract"]
        assert contract["capture_recovery_points"] is False
        assert contract["capture_final_snapshot"] is True


def test_concurrent_terminal_reads_export_once(tmp_path, peer, monkeypatch):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    finish(queue, body)
    driver.tick()
    original_export = adapter._export
    calls = []

    def export(request, document):
        calls.append(request.rollout_job_id)
        time.sleep(0.05)
        return original_export(request, document)

    monkeypatch.setattr(adapter, "_export", export)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: adapter.get(body["rollout_job_id"]), range(2)))

    assert results[0] == results[1]
    assert calls == [body["rollout_job_id"]]
    typed = [
        request[3]
        for request in queue.requests
        if request[1].endswith("/events") and request[3].get("event_type")
    ]
    assert {params["event_type"] for params in typed} == {
        "rollout.usage", "rollout.session_state", "rollout.model_response"
    }
    assert all(
        params.get("newest") == "true"
        for params in typed
        if params["event_type"] in {"rollout.usage", "rollout.session_state"}
    )


def test_claude_structured_prompt_uses_native_system_append(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    body["prompt"] = [
        {"role": "system", "content": "system policy"},
        {"role": "developer", "content": "developer policy"},
        {"role": "user", "content": "run the task"},
    ]
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body, slot="claude-code"))
    adapter.submit(body)
    driver.tick()

    submissions = [request for method, _, request, _ in queue.requests if method == "POST"]
    assert len(submissions) == 2
    for submission in submissions:
        assert submission["spec"]["prompt"] == "run the task"
        assert submission["spec"]["extra"]["system_prompt"] == {
            "type": "preset",
            "preset": "claude_code",
            "append": "system policy\n\ndeveloper policy",
        }
        assert submission["context"]["prompt_token_alignment"] == "harness_rendered"

    finish(queue, body)
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed"
    assert {
        trajectory["prompt_token_alignment"] for trajectory in result["trajectories"]
    } == {"harness_rendered"}


@pytest.mark.parametrize(
    "prompt",
    [
        [{"role": "assistant", "content": "old answer"},
         {"role": "user", "content": "continue"}],
        [{"role": "user", "content": "first"},
         {"role": "user", "content": "second"}],
        [{"role": "system", "content": "policy"},
         {"role": "tool", "content": "old result"},
         {"role": "user", "content": "continue"}],
    ],
)
def test_fresh_native_rollout_rejects_unvalidated_history(tmp_path, peer, prompt):
    queue, backend = peer
    body = miles_request()
    body["prompt"] = prompt
    adapter = MilesAdapter(
        Driver(backend, Ledger(tmp_path / "ledger")),
        config(body, slot="claude-code"),
    )

    with pytest.raises(ValueError, match="validated native prefix"):
        adapter.submit(body)
    assert not queue.jobs


def test_codex_text_prompt_remains_request_exact(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()

    submissions = [request for method, _, request, _ in queue.requests if method == "POST"]
    assert submissions
    assert all(
        submission["context"]["prompt_token_alignment"] == "request_exact"
        for submission in submissions
    )


def test_trajectory_rejects_unknown_prompt_token_alignment():
    body = miles_request()
    trajectory = {
        "sample_slot_id": "slot",
        "branch_id": "branch",
        "messages": [{"role": "assistant", "content": "done"}],
        "token_ids": [11, 12, 90],
        "prompt_length": 2,
        "generated_spans": [{
            "response_id": "response",
            "start": 2,
            "end": 3,
            "input_token_ids": [11, 12],
            "output_token_ids": [90],
            "weight_version": body["expected_weight_version"],
            "finish_reason": "stop",
        }],
        "response_text": "done",
        "prompt_token_alignment": "guessed",
    }

    with pytest.raises(ValueError, match="request_exact or harness_rendered"):
        Trajectory.from_dict(trajectory)


def test_missing_training_tokens_is_explicit_failure_not_fake_completion(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    finish(queue, body, missing=True)
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "failed" and result["actual_samples"] == 0
    assert "no recorded training tokens" in result["stop_reason"]
    assert adapter.execution(body["rollout_job_id"])["status"] == "completed"


@pytest.mark.parametrize("version", [None, "wrong", "unknown"])
def test_export_rejects_unobserved_or_wrong_weight_versions(tmp_path, peer, version):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body); driver.tick(); finish(queue, body)
    for events in queue.events.values():
        events[-1]["state"]["records"][0]["response"]["choices"][0]["meta_info"]["weight_version"] = version
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "failed" and not result["trajectories"]
    assert "weight_version" in result["stop_reason"]


@pytest.mark.parametrize("update", [
    lambda body: body["environment_ref"].update(revision="unlisted"),
    lambda body: body.update(prompt=[{"role": "system", "content": "prefix"}, {"role": "user", "content": "task"}]),
    lambda body: body.update(sampling_params={"top_k": 8}),
    lambda body: body.update(model_endpoint="http://user:secret@example.com"),
    lambda body: body["budgets"].update(max_model_calls=1),
])
def test_unsupported_semantics_rejected_before_queuing(tmp_path, peer, update):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    update(body)
    with TestClient(create_app(driver, None, miles=adapter, background=False)) as http:
        assert http.post("/rollout-groups", json=body).status_code == 400
    assert driver.ledger.active() == []
    assert not queue.jobs


def test_deadline_and_cancel_ack_keep_backend_cleanup_tracked(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    document = driver.ledger.get(internal_id(body["rollout_job_id"]))["document"]
    document["deadline_at"] = time.time() - 1
    driver.ledger.save(document["rollout_job_id"], document)
    driver.tick()
    assert not queue.jobs
    assert adapter.get(body["rollout_job_id"])["status"] == "cancelled"


def test_running_cancellation_is_not_reported_terminal_before_cleanup(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    first, second = list(queue.jobs)
    queue.jobs[first].update(
        state="running",
        active_attempt="attempt-first",
        progress={"phase": "model_generation", "model_calls": 1},
    )

    acknowledgement = adapter.release(body["rollout_job_id"])
    assert acknowledgement["status"] == "cancelled"
    driver.tick()

    pending = adapter.get(body["rollout_job_id"])
    assert pending["status"] == "running"
    assert pending["progress"]["phase"] == "cancelling"
    assert "cleanup" in pending["stop_reason"]
    assert queue.jobs[first]["phase"] == "cancelling"
    assert queue.jobs[second]["state"] == "cancelled"

    queue.cancel(first)
    driver.tick()
    terminal = adapter.get(body["rollout_job_id"])
    assert terminal["status"] == "cancelled"
    assert "progress" not in terminal


def test_quarantined_actor_fails_group_instead_of_polling_forever(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    body["max_samples"] = body["minimum_returned_samples"] = 1
    body["sample_slots"] = body["sample_slots"][:1]
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))

    adapter.submit(body)
    driver.tick()
    job_id = next(iter(queue.jobs))
    queue.jobs[job_id].update(
        state="quarantined",
        phase="quarantined",
        active_attempt="attempt-quarantined",
        error="Owned allocation or cleanup needs reconciliation",
        result={
            "status": "error",
            "failure_kind": "infrastructure",
            "error": "Owned allocation or cleanup needs reconciliation",
        },
    )

    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "failed"
    assert result["actual_samples"] == 0
    assert "recorded training tokens" in result["stop_reason"]
    assert driver.ledger.get(internal_id(body["rollout_job_id"]))["terminal"]


def test_shared_session_survives_active_cancel_and_releases_once_after_cleanup(
    tmp_path, peer, monkeypatch
):
    queue, backend = peer
    body = miles_request()
    body["session_server_endpoint"] = "http://session-server:30000"
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    deleted = []

    def delete(url, *, timeout):
        deleted.append((url, timeout))
        return httpx.Response(204)

    monkeypatch.setattr(httpx, "delete", delete)
    adapter.submit(body)
    driver.tick()
    first, second = list(queue.jobs)
    queue.jobs[first].update(state="running", active_attempt="attempt-first")

    adapter.release(body["rollout_job_id"])
    assert deleted == []
    driver.tick()
    assert deleted == []
    assert queue.jobs[second]["state"] == "cancelled"

    queue.cancel(first)
    driver.tick()
    adapter.reconcile_sessions()
    assert deleted == [(
        "http://session-server:30000/sessions/" + internal_id(body["rollout_job_id"]),
        30,
    )]
    adapter.reconcile_sessions()
    adapter.release(body["rollout_job_id"])
    assert len(deleted) == 1


def test_completed_shared_session_is_retained_until_result_consumption(
    tmp_path, peer, monkeypatch
):
    queue, backend = peer
    body = miles_request()
    body["session_server_endpoint"] = "http://session-server:30000"
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    deleted = []
    monkeypatch.setattr(
        httpx,
        "delete",
        lambda url, *, timeout: deleted.append((url, timeout)) or httpx.Response(404),
    )

    adapter.submit(body)
    driver.tick()
    finish(queue, body)
    driver.tick()
    assert adapter.get(body["rollout_job_id"])["status"] == "completed"
    adapter.reconcile_sessions()
    assert deleted == []

    adapter.release(body["rollout_job_id"])
    assert len(deleted) == 1
    adapter.release(body["rollout_job_id"])
    assert len(deleted) == 1


def test_running_group_aggregates_durable_child_progress(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    first, second = list(queue.jobs)
    queue.jobs[first].update(
        state="running",
        active_attempt="attempt-first",
        progress={
            "phase": "model_response",
            "model_calls": 3,
            "tool_calls": 2,
            "trajectory_tokens": 101,
            "assistant_generated_tokens": 37,
            "current_context_tokens": 64,
            "peak_context_tokens": 80,
            "last_model_output_tokens": 11,
            "updated_at_unix_seconds": time.time(),
        },
    )
    queue.jobs[second].update(
        state="running",
        active_attempt="attempt-second",
        progress={
            "phase": "tool_execution",
            "model_calls": 1,
            "tool_calls": 1,
            "updated_at_unix_seconds": time.time() - 1,
        },
    )
    driver.tick()

    progress = adapter.get(body["rollout_job_id"])["progress"]
    assert progress["phase"] == "model_response"
    assert progress["model_calls"] == 4
    assert progress["tool_calls"] == 3
    assert progress["active_sample_slot_id"] == body["sample_slots"][0]["sample_slot_id"]
    assert progress["trajectory_tokens"] == 101
    assert progress["last_model_output_tokens"] == 11
    assert progress["remaining_wall_time_seconds"] > 0


def test_wire_fields_reach_execution_contract_and_no_legacy_request_is_accepted(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    body.update(model="checkpoint-7", session_server_endpoint="http://session-server:30000",
                sampling_params={"temperature": 0.75, "max_new_tokens": 27})
    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    with TestClient(create_app(driver, None, miles=adapter, background=False)) as http:
        assert http.post("/rollout-groups", json=body).status_code == 202
        assert http.post("/rollout-groups", json={"samples": []}).status_code == 400
    driver.stopping.clear()
    driver.tick()
    for method, path, request, _ in queue.requests:
        if method == "POST":
            contract = request["spec"]["extra"]["rollout_contract"]
            assert contract["session_server_endpoint"] == body["session_server_endpoint"]
            assert contract["model_endpoint"] == body["model_endpoint"]
            assert contract["sampling_params"] == body["sampling_params"]
            assert request["spec"]["model"] == "checkpoint-7"


def test_internal_branch_policy_consumes_only_allocated_slots_and_preserves_training_context(
    tmp_path, peer
):
    queue, backend = peer
    body = miles_request()
    body["session_server_endpoint"] = "http://session-server:30000"
    cfg = config(body)
    child_slot_id = body["sample_slots"][1]["sample_slot_id"]

    def policy(state):
        parent = state["samples"][0]
        if not parent["recovery_points"]:
            return None
        return [{
            "sample_slot_id": child_slot_id,
            "decision": {
                "kind": "branch",
                "source_sample_slot_id": state["samples"][0]["sample_slot_id"],
                "point_id": parent["recovery_points"][0]["id"],
                "overrides": {},
            },
        }]

    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, cfg, branch_policy=policy)
    adapter.submit(body)
    driver.tick()
    root_request = next(
        row[2] for row in queue.requests
        if row[0] == "POST" and row[1] == "/v1/jobs"
    )
    assert root_request["spec"]["extra"]["rollout_contract"][
        "capture_recovery_points"
    ] is True
    assert len(queue.jobs) == 1

    parent_slot, child_slot = body["sample_slots"]
    parent_job = next(iter(queue.jobs))
    queue.jobs[parent_job].update(
        state="succeeded", active_attempt="attempt-parent",
        result={"status": "completed"},
    )
    queue.events[parent_job] = [
        {"seq": 1, "type": "rollout.usage", "model_calls": 1, "tool_calls": 1},
        {"seq": 2, "type": "rollout.model_response", "response_id": "response-parent"},
        {"seq": 3, "type": "rollout.session_state", "state": recorded_state(body, 90)},
    ]
    queue.points[parent_job] = [{
        "id": "joint-point", "available": True, "tool_depth": 1,
        "message_step": 1, "model_position": {
            "session_id": internal_id(body["rollout_job_id"]),
            "response_id": "response-parent",
        },
    }]
    driver.tick()
    adapter.reconcile_policy()
    driver.tick()
    child_job = next(job for job in queue.jobs if job != parent_job)
    branch_request = next(
        request for method, path, request, _ in queue.requests
        if method == "POST" and path.endswith("/branch")
    )
    context = branch_request["context"]
    assert context["sample_slot_id"] == child_slot["sample_slot_id"]
    assert context["prompt_group_id"] == body["prompt_group_id"]
    assert context["miles_session_id"] == internal_id(body["rollout_job_id"])
    assert "rollout_contract" not in branch_request
    assert queue.jobs[child_job]


def test_branch_policy_cannot_consume_an_unallocated_or_eager_slot(tmp_path, peer):
    queue, backend = peer
    body = miles_request()
    attempted = ["unallocated-slot", body["sample_slots"][0]["sample_slot_id"]]

    def policy(_state):
        return [{"sample_slot_id": attempted.pop(0), "decision": {"kind": "skip"}}]

    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body), branch_policy=policy)
    adapter.submit(body)
    driver.tick()
    with pytest.raises(KeyError):
        adapter.reconcile_policy()
    with pytest.raises(ValueError, match="not deferred"):
        adapter.reconcile_policy()
    assert len(queue.jobs) == 1


def test_branch_policy_reference_validation():
    assert load_branch_policy(None) is None
    assert load_branch_policy("json:loads") is json.loads
    with pytest.raises(ValueError, match="module:callable"):
        load_branch_policy("missing-separator")
    with pytest.raises(ValueError, match="not callable"):
        load_branch_policy("json:decoder")


def test_reference_policy_is_explicitly_deterministic_and_quota_bounded():
    state = {
        "request": {},
        "samples": [{
            "sample_slot_id": "parent", "recovery_points": [
                {"id": "missing", "available": False},
                {"id": "point", "available": True},
            ],
        }],
        "deferred_sample_slot_ids": ["child-1", "child-2"],
    }

    assert first_available_recovery(state) == [
        {"sample_slot_id": child, "decision": {
            "kind": "branch", "source_sample_slot_id": "parent",
            "point_id": "point", "overrides": {
                "prompt": (
                    "Continue from the restored tool result and complete "
                    "the task without repeating completed actions."
                )
            },
        }}
        for child in ("child-1", "child-2")
    ]


def test_reference_policy_waits_for_joint_model_position():
    state = {
        "request": {"session_server_endpoint": "http://miles"},
        "samples": [{
            "sample_slot_id": "parent",
            "state": "succeeded",
            "recovery_points": [{"id": "environment-only", "available": True}],
        }],
        "deferred_sample_slot_ids": ["child"],
    }

    assert first_available_recovery(state) == [
        {"sample_slot_id": "child", "decision": {"kind": "skip"}}
    ]


def test_reference_policy_releases_deferred_slots_after_source_failure():
    state = {
        "samples": [
            {"sample_slot_id": "parent", "state": "failed", "recovery_points": []},
            {"sample_slot_id": "child", "state": "deferred", "recovery_points": []},
        ],
        "deferred_sample_slot_ids": ["child"],
    }

    assert first_available_recovery(state) == [
        {"sample_slot_id": "child", "decision": {"kind": "skip"}}
    ]


def test_reference_policy_releases_deferred_slots_after_source_quarantine():
    state = {
        "samples": [
            {"sample_slot_id": "parent", "state": "quarantined", "recovery_points": []},
            {"sample_slot_id": "child", "state": "deferred", "recovery_points": []},
        ],
        "deferred_sample_slot_ids": ["child"],
    }

    assert first_available_recovery(state) == [
        {"sample_slot_id": "child", "decision": {"kind": "skip"}}
    ]


def test_reference_policy_waits_while_source_can_still_publish_recovery():
    state = {
        "samples": [
            {"sample_slot_id": "parent", "state": "running", "recovery_points": []},
            {"sample_slot_id": "child", "state": "deferred", "recovery_points": []},
        ],
        "deferred_sample_slot_ids": ["child"],
    }

    assert first_available_recovery(state) is None


def test_internal_branch_policy_exports_parent_child_lineage_from_one_session_tree(
    tmp_path, peer
):
    queue, backend = peer
    body = miles_request()
    body["session_server_endpoint"] = "http://session-server:30000"
    cfg = config(body)
    child_slot_id = body["sample_slots"][1]["sample_slot_id"]

    def policy(state):
        parent = state["samples"][0]
        if not parent["recovery_points"]:
            return None
        return [{
            "sample_slot_id": child_slot_id,
            "decision": {
                "kind": "branch",
                "source_sample_slot_id": state["samples"][0]["sample_slot_id"],
                "point_id": parent["recovery_points"][0]["id"],
                "overrides": {},
            },
        }]

    driver = Driver(backend, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, cfg, branch_policy=policy)
    adapter.submit(body)
    driver.tick()
    parent_slot, child_slot = body["sample_slots"]
    parent_job = next(iter(queue.jobs))
    queue.jobs[parent_job].update(
        state="succeeded", active_attempt="attempt-parent", result={"status": "completed"}
    )
    parent_state = branched_state(body)
    queue.events[parent_job] = [
        {"seq": 1, "type": "rollout.usage", "model_calls": 1, "tool_calls": 1},
        {"seq": 2, "type": "rollout.model_response", "response_id": "response-parent"},
        {"seq": 3, "type": "rollout.session_state", "state": parent_state},
    ]
    queue.points[parent_job] = [{
        "id": "joint-point", "available": True, "tool_depth": 1,
        "message_step": 1, "model_position": {
            "session_id": internal_id(body["rollout_job_id"]),
            "response_id": "response-parent",
        },
    }]
    driver.tick()
    adapter.reconcile_policy()
    driver.tick()
    child_job = next(job for job in queue.jobs if job != parent_job)
    queue.jobs[child_job].update(
        state="succeeded", active_attempt="attempt-child", result={"status": "completed"}
    )
    queue.events[child_job] = [
        {"seq": 1, "type": "rollout.usage", "model_calls": 1, "tool_calls": 0},
        {"seq": 2, "type": "rollout.model_response", "response_id": "response-child"},
        {"seq": 3, "type": "rollout.session_state", "state": parent_state},
    ]
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed"
    assert result["search_branches"] == 1
    parent, child = result["trajectories"]
    assert parent["sample_slot_id"] == parent_slot["sample_slot_id"]
    assert parent["parent_branch_id"] is None
    assert parent["generated_spans"][0]["response_id"] == "response-parent"
    assert child["sample_slot_id"] == child_slot["sample_slot_id"]
    assert child["parent_branch_id"] == parent_job
    assert child["branch_point_token_count"] == len(body["prompt_token_ids"]) + 2
    assert [span["response_id"] for span in child["generated_spans"]] == [
        "response-parent", "response-child"
    ]


def test_main_example_uses_original_request_type():
    example = json.loads((Path(__file__).parents[1] / "group.example.json").read_text())
    parsed = RolloutGroupRequest.from_dict(example)
    assert parsed.protocol_version == "ash-rollout-v2"
    assert len(parsed.sample_slots) == 2
