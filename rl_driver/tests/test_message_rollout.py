from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from runstore.message_sampling import parameters
from rl_driver.driver import Driver
from rl_driver.ledger import Ledger
from rl_driver.message_protocol import MessageRequest
from rl_driver.miles import MilesAdapter, internal_id
from rl_driver.server import create_app
from rl_driver.tests.test_driver import peer as peer
from runstore.message_export import clean_messages, codex_messages, export_tools, mark_hint
from runstore.swerebench import resolved


def request():
    return json.loads((Path(__file__).parent / "fixtures/miles-message-request.json").read_text())


def config(body):
    return {
        "profile": "codex", "run_defaults": {"slot": "codex", "tools": "shell_only"},
        "resources": {"standard": {"cpu": 2, "memory_mb": 12288}},
        "allowed_oci_registries": ["docker.io"],
        "tasks": {body["task_id"]: {
            "environment_ref": deepcopy(body["environment_ref"]),
            "repository": {
                "workdir": "/task-repository",
                "base_commit": "1" * 40,
            },
            "grade": {"profile": "grade", "spec": {
                "benchmark": "swe-rebench-v2", "instance_id": body["task_id"],
                "dataset_path": "/ash/tasks.jsonl", "dataset_sha256": "fixture",
                "grader_revision": "sha256:fixture", "parser_path": "/ash/log_parsers.py",
            }},
        }},
    }


def messages():
    return [
        {"role": "user", "content": "fix task"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
        }]},
        {"role": "tool", "tool_call_id": "call", "content": "/workspace"},
        {"role": "user", "content": mark_hint("check the parser")},
        {"role": "assistant", "content": "fixed"},
    ]


def test_real_miles_request_to_queue_grade_and_message_response(tmp_path, peer, monkeypatch):
    queue, client = peer
    body = request()
    class Released:
        status_code = 204

        def raise_for_status(self):
            return None

    monkeypatch.setattr("rl_driver.miles.httpx.delete", lambda *args, **kwargs: Released())
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    with TestClient(create_app(driver, None, miles=adapter, background=False)) as http:
        assert http.post("/rollout-groups", json=body).status_code == 202
        driver.tick()
        actors = list(queue.jobs)
        for job_id in actors:
            queue.finish(job_id)
            queue.jobs[job_id]["result"]["training_messages"] = messages()
            queue.final_point(job_id)
        driver.tick()
        for job_id, job in queue.jobs.items():
            if job["kind"] == "grade":
                queue.finish(job_id, resolved=True)
        driver.tick()
        result = http.get("/rollout-groups/" + body["rollout_job_id"]).json()
        assert result["protocol_version"] == "ash-rollout-v3"
        assert result["status"] == "completed", result
        assert result["actual_samples"] == body["max_samples"]
        (tmp_path / "wire-result.json").write_text(json.dumps(result, indent=2) + "\n")
        trajectory = result["trajectories"][0]
        assert trajectory["reward"] == 1.0 and trajectory["hints_removed"]
        assert "check the parser" not in json.dumps(trajectory["messages"])
        assert "/workspace" in json.dumps(trajectory["messages"])
        assert "token_ids" not in trajectory
        persisted = driver.ledger.get(internal_id(body["rollout_job_id"]))["document"]
        assert persisted["message_result"] == result
        assert len(persisted["profiling_records"]) == result["actual_samples"]
        assert all(record["protocol_version"] == "ash-rollout-v3"
                   and record["final_transcript_tokens"] is None
                   for record in persisted["profiling_records"])
        assert http.post("/rollout-groups", json=body).json()["status"] == "completed"
        deleted = http.delete("/rollout-groups/" + body["rollout_job_id"])
        assert deleted.json()["protocol_version"] == "ash-rollout-v3"
        assert len(queue.jobs) == len(actors) * 2
    submissions = [row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs"]
    actor = next(row for row in submissions if row["kind"] == "rollout")
    grade = next(row for row in submissions if row["kind"] == "grade")
    assert actor["spec"]["sandbox_image"] == (
        "docker.io/swerebenchv2/task@sha256:" + "a" * 64
    )
    assert actor["spec"]["extra"]["rollout_contract"]["sampling_params"]["top_k"] == 20
    assert actor["spec"]["extra"]["rollout_contract"]["message_export"] is True
    assert actor["spec"]["extra"]["rollout_contract"]["max_turns"] == body["max_turns"]
    assert actor["spec"]["extra"]["rollout_contract"]["capture_recovery_points"] is False
    assert actor["spec"]["extra"]["rollout_contract"]["capture_final_snapshot"] is True
    assert actor["spec"]["extra"]["repository_preflight"] == {
        "workdir": "/task-repository",
        "base_commit": "1" * 40,
    }
    assert grade["spec"]["baseline_untracked"] == ["image-cache.txt"]
    assert grade["spec"]["repository_workdir"] == "/task-repository"
    assert grade["spec"]["repository_base_commit"] == "1" * 40


def test_concurrent_v3_terminal_reads_export_once(tmp_path, peer, monkeypatch):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    actors = list(queue.jobs)
    for job_id in actors:
        queue.finish(job_id)
        queue.jobs[job_id]["result"]["training_messages"] = messages()
        queue.final_point(job_id)
    driver.tick()
    for job_id, job in queue.jobs.items():
        if job["kind"] == "grade":
            queue.finish(job_id, resolved=True)
    driver.tick()

    from rl_driver.messages import MessageAdapter

    original_export = MessageAdapter._export
    calls = []

    def export(self, request, document, result):
        calls.append(request.rollout_job_id)
        return original_export(self, request, document, result)

    monkeypatch.setattr(MessageAdapter, "_export", export)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _index: adapter.get(body["rollout_job_id"]), range(2))
        )

    assert results[0] == results[1]
    assert calls == [body["rollout_job_id"]]


def test_health_reports_nonterminal_groups_for_storage_drain(tmp_path, peer):
    _queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    with TestClient(create_app(driver, None, miles=adapter, background=False)) as http:
        health = http.get("/health").json()
        assert health["nonterminal_groups"] == 0
        assert health["active_workers"] == 0
        assert http.post("/rollout-groups", json=body).status_code == 202
        health = http.get("/health").json()
        assert health["nonterminal_groups"] == 1
        assert health["active_workers"] == 1


def test_v3_without_deployment_grader_defers_reward_to_miles(tmp_path, peer):
    queue, client = peer
    body = request()
    cfg = config(body)
    cfg["tasks"][body["task_id"]].pop("grade")
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, cfg)

    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.finish(actor_id)
    queue.jobs[actor_id]["result"]["training_messages"] = messages()
    queue.final_point(actor_id)
    driver.tick()

    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed", result
    assert result["trajectories"][0]["reward"] is None
    assert all(job["kind"] == "rollout" for job in queue.jobs.values())


def test_v3_claude_preserves_system_preamble_as_sdk_system_prompt(tmp_path, peer):
    queue, client = peer
    body = request()
    body["prompt"] = [
        {"role": "system", "content": "benchmark policy"},
        {"role": "user", "content": "fix task"},
    ]
    cfg = config(body)
    cfg["run_defaults"]["slot"] = "claude-code"
    adapter = MilesAdapter(Driver(client, Ledger(tmp_path / "ledger")), cfg)
    adapter.submit(body)
    adapter.driver.tick()
    actor = next(
        body
        for method, path, body, _params in queue.requests
        if method == "POST" and path == "/v1/jobs"
    )
    assert actor["spec"]["prompt"] == "fix task"
    assert actor["spec"]["extra"]["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
        "append": "benchmark policy",
    }
    assert actor["context"]["prompt_token_alignment"] == "harness_rendered"


def test_v3_claude_plain_prompt_still_records_harness_rendering(tmp_path, peer):
    queue, client = peer
    body = request()
    cfg = config(body)
    cfg["run_defaults"]["slot"] = "claude-code"
    adapter = MilesAdapter(Driver(client, Ledger(tmp_path / "ledger")), cfg)
    adapter.submit(body)
    adapter.driver.tick()
    actor = next(
        request_body
        for method, path, request_body, _params in queue.requests
        if method == "POST" and path == "/v1/jobs"
    )
    assert actor["spec"]["prompt"] == "fix task"
    assert "system_prompt" not in actor["spec"]["extra"]
    assert actor["context"]["prompt_token_alignment"] == "harness_rendered"


def test_v3_claude_branch_policy_consumes_deferred_slot_and_exports_lineage(
    tmp_path, peer
):
    queue, client = peer
    body = request()
    body.update(max_samples=2, minimum_returned_samples=2)
    body["sample_slots"].append({"sample_slot_id": "child-slot", "sample_index": 12})
    child_slot_id = body["sample_slots"][1]["sample_slot_id"]

    def policy(state):
        parent = state["samples"][0]
        if not parent["recovery_points"]:
            return None
        return [{
            "sample_slot_id": child_slot_id,
            "decision": {
                "kind": "branch",
                "source_sample_slot_id": parent["sample_slot_id"],
                "point_id": parent["recovery_points"][0]["id"],
                "overrides": {},
            },
        }]

    cfg = config(body)
    cfg["profile"] = "claude-code"
    cfg["run_defaults"]["slot"] = "claude-code"
    cfg["branch_policy"] = "fixture:policy"
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, cfg, branch_policy=policy)
    adapter.submit(body)
    driver.tick()
    assert len(queue.jobs) == 1
    parent_job = next(iter(queue.jobs))
    root_request = next(
        row[2] for row in queue.requests
        if row[0] == "POST" and row[1] == "/v1/jobs"
    )
    assert root_request["spec"]["extra"]["rollout_contract"][
        "capture_recovery_points"
    ] is True

    queue.finish(parent_job)
    queue.jobs[parent_job]["result"]["training_messages"] = messages()
    queue.final_point(parent_job)
    queue.points[parent_job][0]["model_position"] = {
        "session_id": internal_id(body["rollout_job_id"]),
        "response_id": "parent-response",
    }
    driver.tick()
    adapter.reconcile_policy()
    driver.tick()

    branch_requests = [
        row[2] for row in queue.requests
        if row[0] == "POST" and row[1].endswith("/branch")
    ]
    assert len(branch_requests) == 1
    assert branch_requests[0]["point_id"] == "old"
    assert branch_requests[0]["context"]["sample_slot_id"] == child_slot_id
    assert branch_requests[0]["context"]["environment_ref"] == body["environment_ref"]
    child_job = next(
        job for job, value in queue.jobs.items()
        if job != parent_job and value["kind"] == "rollout"
    )

    queue.finish(child_job)
    queue.jobs[child_job]["result"]["training_messages"] = [
        *messages()[:3],
        {"role": "user", "content": mark_hint("try the branch")},
        {"role": "assistant", "content": "branched fix"},
    ]
    queue.final_point(child_job)
    driver.tick()
    driver.tick()
    grades = [job_id for job_id, job in queue.jobs.items() if job["kind"] == "grade"]
    assert len(grades) == 2
    for job_id in grades:
        queue.finish(job_id, resolved=True)
    driver.tick()

    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed", result
    assert result["actual_samples"] == 2
    assert result["search_branches"] == 1
    parent, child = result["trajectories"]
    assert parent["sample_slot_id"] == body["sample_slots"][0]["sample_slot_id"]
    assert parent["parent_branch_id"] is None
    assert child["sample_slot_id"] == child_slot_id
    assert child["parent_branch_id"] == parent_job
    assert child["metadata"]["origin"]["point_id"] == "old"
    assert "try the branch" not in json.dumps(child["messages"])
    assert child["messages"][-1]["content"] == "branched fix"


@pytest.mark.parametrize("group_size", [1, 2, 8])
def test_every_queued_trajectory_receives_the_full_turn_limit(tmp_path, peer, group_size):
    queue, client = peer
    body = request()
    body.update(max_samples=group_size, minimum_returned_samples=group_size, max_turns=2)
    body["sample_slots"] = [
        {"sample_slot_id": f"slot-{i}", "sample_index": i} for i in range(group_size)
    ]
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    actors = [row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs"]
    assert len(actors) == group_size
    controls = [actor["spec"]["extra"]["rollout_contract"] for actor in actors]
    assert all(control["max_turns"] == 2 for control in controls)
    assert all(not ({"max_model_calls", "max_tool_calls"} & control.keys()) for control in controls)
    assert len({control["deadline_at"] for control in controls}) == 1
    assert all(actor["spec"]["timeout_s"] == body["budgets"]["max_wall_time_seconds"] for actor in actors)


@pytest.mark.parametrize("reason", ["timeout", "max_turns_reached"])
@pytest.mark.parametrize("resolved", [False, True])
def test_limit_cutoff_is_graded_after_deadline_and_returned_as_truncated(tmp_path, peer, monkeypatch, reason, resolved):
    from rl_driver.miles import internal_id

    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.finish(actor_id)
    queue.jobs[actor_id]["result"].update(
        status="truncated", stop_reason=reason, final_snapshot_id="actual-final-state",
        training_messages=messages(),
    )
    queue.events[actor_id] = [{
        "seq": 1,
        "type": "environment.prepared",
        "workdir": "/task-repository",
        "base_commit": "1" * 40,
        "agent_workdir": "/testbed",
        "baseline_untracked": ["image-cache.txt"],
    }]
    row = driver.ledger.get(internal_id(body["rollout_job_id"]))
    document = row["document"]
    # Reaching the execution deadline must not cancel grading.
    execution_deadline = document["execution_deadline_at"]
    assert document["deadline_at"] == execution_deadline + 1800.0
    monkeypatch.setattr("rl_driver.driver.time.time", lambda: execution_deadline + 1)
    driver.tick()
    grades = [jid for jid, job in queue.jobs.items() if job["kind"] == "grade"]
    assert len(grades) == 1
    submission = next(r[2] for r in queue.requests if r[0] == "POST" and r[1] == "/v1/jobs"
                      and r[2]["kind"] == "grade")
    assert submission["spec"]["snapshot_id"] == "actual-final-state"
    queue.finish(grades[0], resolved=resolved)
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed", result
    trajectory = result["trajectories"][0]
    assert trajectory["status"] == "truncated" and trajectory["stop_reason"] == reason
    assert trajectory["reward"] == float(resolved)
    assert trajectory["metadata"]["graded_snapshot_id"] == "actual-final-state"


def test_infrastructure_failure_is_not_graded_as_a_timeout(tmp_path, peer):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.jobs[actor_id].update(
        state="failed", active_attempt="a",
        result={"status": "error", "failure_kind": "infrastructure", "error": "execution_uncertain"},
    )
    driver.tick()
    assert all(job["kind"] != "grade" for job in queue.jobs.values())
    assert adapter.get(body["rollout_job_id"])["status"] == "failed"


def test_message_cancellation_waits_for_worker_cleanup(tmp_path, peer):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.jobs[actor_id].update(
        state="running",
        active_attempt="attempt-actor",
        progress={"phase": "model_generation", "model_calls": 1},
    )

    assert adapter.release(body["rollout_job_id"])["status"] == "cancelled"
    driver.tick()
    pending = adapter.get(body["rollout_job_id"])
    assert pending["status"] == "running"
    assert pending["progress"]["phase"] == "cancelling"

    queue.cancel(actor_id)
    driver.tick()
    assert adapter.get(body["rollout_job_id"])["status"] == "cancelled"

@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_message_request_rejects_invalid_turn_limit(limit):
    body = request()
    body["max_turns"] = limit
    with pytest.raises(ValueError, match="max_turns"):
        MessageRequest.from_dict(body)


def test_message_request_accepts_explicit_unbounded_turn_limit():
    body = request()
    body["max_turns"] = None

    assert MessageRequest.from_dict(body).max_turns is None


@pytest.mark.parametrize("key", ["max_model_calls", "max_tool_calls"])
def test_message_request_rejects_legacy_count_budgets(key):
    body = request()
    body["budgets"][key] = 100
    with pytest.raises(ValueError, match="max_turns"):
        MessageRequest.from_dict(body)


def test_missing_task_binding_rejected_before_queueing(tmp_path, peer):
    queue, client = peer
    body = request()
    cfg = config(body)
    cfg["tasks"] = {}
    adapter = MilesAdapter(Driver(client, Ledger(tmp_path / "ledger")), cfg)
    with pytest.raises(ValueError, match="environment_ref"):
        adapter.submit(body)
    assert queue.jobs == {}


def test_task_environment_mismatch_is_rejected_before_queueing(tmp_path, peer):
    queue, client = peer
    body = request()
    cfg = config(body)
    cfg["tasks"][body["task_id"]]["environment_ref"]["revision"] = (
        "sha256:" + "b" * 64
    )
    adapter = MilesAdapter(Driver(client, Ledger(tmp_path / "ledger")), cfg)

    with pytest.raises(ValueError, match="environment_ref does not match"):
        adapter.submit(body)
    assert queue.jobs == {}


@pytest.mark.parametrize(
    "repository",
    [
        {"workdir": "relative", "base_commit": "1" * 40},
        {"workdir": "/repo", "base_commit": "short"},
        {"workdir": "/repo", "base_commit": "1" * 40, "command": "unsafe"},
    ],
)
def test_task_repository_preflight_schema_is_closed(tmp_path, peer, repository):
    queue, client = peer
    body = request()
    cfg = config(body)
    cfg["tasks"][body["task_id"]]["repository"] = repository
    adapter = MilesAdapter(Driver(client, Ledger(tmp_path / "ledger")), cfg)

    with pytest.raises(ValueError, match="repository|base_commit|workdir"):
        adapter.submit(body)
    assert queue.jobs == {}


def test_hint_removal_does_not_delete_assistant_or_tool_text():
    source = messages()
    source[2]["content"] = mark_hint("literal tool output")
    source[-1]["content"] = mark_hint("literal assistant output")
    result = clean_messages(source)
    assert len(result) == len(source) - 1
    assert result[2] == source[2]
    assert result[-1] == source[-1]
    assert source[3]["content"].startswith("<ash_training_hint>")
    bad = deepcopy(source)
    bad[3]["content"] = "<ash_training_hint>incomplete"
    with pytest.raises(ValueError, match="Incomplete"):
        clean_messages(bad)


def test_native_branch_history_keeps_prefix_and_pairs_tools():
    entries = [
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                             "content": [{"type": "input_text", "text": "original task"}]}},
        {"type": "response_item", "payload": {"type": "function_call", "call_id": "tool",
                                             "name": "shell", "arguments": '{"command":"pwd"}'}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "tool", "output": "/repo"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                             "content": [{"type": "input_text", "text": mark_hint("new direction")}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                             "content": [{"type": "output_text", "text": "new patch"}]}},
    ]
    result = clean_messages(codex_messages(entries))
    assert [m["role"] for m in result] == ["user", "assistant", "tool", "assistant"]
    assert result[0]["content"] == "original task"
    assert result[-1]["content"] == "new patch"


@pytest.mark.parametrize("shape,length_field,stop_field", [
    ("responses", "max_output_tokens", "stop"), ("messages", "max_tokens", "stop_sequences"),
])
def test_sampling_values_survive_native_mapping(shape, length_field, stop_field):
    controls = request()["sampling_params"]
    mapped = parameters(controls, shape)
    assert mapped[length_field] == controls["max_new_tokens"]
    assert mapped[stop_field] == controls["stop"]
    for key in ("temperature", "top_p", "top_k"):
        assert mapped[key] == controls[key]
    assert controls == request()["sampling_params"]


@pytest.mark.parametrize("shape", ["responses", "messages"])
def test_sampling_seed_survives_native_mapping(shape):
    assert parameters({"seed": 20260915}, shape)["seed"] == 20260915


def test_unsupported_controls_and_wire_fields_are_not_silently_accepted():
    with pytest.raises(ValueError, match="Unknown"):
        parameters({"stop_token_ids": [7]}, "responses")
    body = request()
    body["prompt_token_ids"] = [1, 2]
    with pytest.raises(ValueError, match="Unknown"):
        MessageRequest.from_dict(body)


def test_swe_rebench_rewards_need_all_tests_and_nonempty_expectations():
    task = {"FAIL_TO_PASS": ["new test [10 ms]"], "PASS_TO_PASS": ["old test"]}
    assert resolved({"new test": "PASSED", "old test": "PASSED"}, task)
    assert not resolved({"new test": "PASSED", "old test": "FAILED"}, task)
    assert not resolved({"old test": "PASSED"}, task)
    assert not resolved({}, {"FAIL_TO_PASS": [], "PASS_TO_PASS": []})


def test_tool_schemas_preserved():
    tools = export_tools([{"type": "rollout.model_tools", "shape": "messages", "tools": [
        {"name": "shell", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    ]}])
    assert tools[0]["function"]["parameters"]["properties"]["command"] == {"type": "string"}
