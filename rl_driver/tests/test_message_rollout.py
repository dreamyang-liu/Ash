from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from runstore.message_sampling import parameters
from rl_driver.driver import Driver
from rl_driver.ledger import Ledger
from rl_driver.message_protocol import MessageRequest
from rl_driver.miles import MilesAdapter
from rl_driver.server import create_app
from rl_driver.tests.test_driver import peer as peer
from runstore.message_export import clean_messages, codex_messages, export_tools, mark_hint
from runstore.swerebench import resolved


def request():
    return json.loads((Path(__file__).parent / "fixtures/miles-message-request.json").read_text())


def config(body):
    return {
        "profile": "codex", "run_defaults": {"slot": "codex", "tools": "shell_only"},
        "image_resources": {"cpu": 2, "memory_mb": 12288},
        "tasks": {body["task_id"]: {"grade": {"profile": "grade", "spec": {
            "benchmark": "swe-rebench-v2", "instance_id": body["task_id"],
            "dataset_path": "/ash/tasks.jsonl", "dataset_sha256": "fixture",
            "grader_revision": "sha256:fixture", "parser_path": "/ash/log_parsers.py",
        }}}},
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


def test_real_miles_request_to_queue_grade_and_message_response(tmp_path, peer):
    queue, client = peer
    body = request()
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
        assert http.post("/rollout-groups", json=body).json()["status"] == "completed"
        deleted = http.delete("/rollout-groups/" + body["rollout_job_id"])
        assert deleted.json()["protocol_version"] == "ash-rollout-v3"
        assert len(queue.jobs) == len(actors) * 2
    submissions = [row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs"]
    actor = next(row for row in submissions if row["kind"] == "rollout")
    assert actor["spec"]["sandbox_image"] == "docker.io/swerebenchv2/task:base"
    assert actor["spec"]["extra"]["rollout_contract"]["sampling_params"]["top_k"] == 20
    assert actor["spec"]["extra"]["rollout_contract"]["message_export"] is True
    assert actor["spec"]["extra"]["rollout_contract"]["max_turns"] == body["max_turns"]


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

@pytest.mark.parametrize("limit", [None, 0, -1, True, 1.5])
def test_message_request_rejects_invalid_turn_limit(limit):
    body = request()
    body["max_turns"] = limit
    with pytest.raises(ValueError, match="max_turns"):
        MessageRequest.from_dict(body)


@pytest.mark.parametrize("key", ["max_model_calls", "max_tool_calls"])
def test_message_request_rejects_legacy_count_budgets(key):
    body = request()
    body["budgets"][key] = 100
    with pytest.raises(ValueError, match="max_turns"):
        MessageRequest.from_dict(body)


def test_missing_grader_rejected_before_queueing(tmp_path, peer):
    queue, client = peer
    body = request()
    cfg = config(body)
    cfg["tasks"] = {}
    adapter = MilesAdapter(Driver(client, Ledger(tmp_path / "ledger")), cfg)
    with pytest.raises(ValueError, match="grade"):
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


@pytest.mark.parametrize("truncated,resolved,expected", [(True, True, 0.5), (True, False, 0), (False, True, 1)])
@pytest.mark.parametrize("actual_tokens", [123, 81921])
def test_length_cap_grades_retained_snapshot_and_discounts_only_truncated_success(
    tmp_path, peer, truncated, resolved, expected, actual_tokens, monkeypatch,
):
    def count(exported_messages, tools, path):
        assert "check the parser" not in json.dumps(exported_messages)
        assert path == "/fixture/tokenizer"
        return actual_tokens
    monkeypatch.setattr("runstore.sequence_limits.token_count", count)
    queue, client = peer
    body = request()
    body.update(max_sequence_tokens=81920, truncated_reward_scale=0.5)
    settings = config(body)
    settings["sequence_tokenizers"] = {body["model"]: "/fixture/tokenizer"}
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MilesAdapter(driver, settings)
    adapter.submit(body)
    driver.tick()
    actor = next(iter(queue.jobs))
    queue.finish(actor)
    queue.jobs[actor]["result"].update(
        training_messages=messages(), final_snapshot_id="retained-prefix-state",
        training_token_count=120,  # Deliberately stale worker metadata.
        raw_final_snapshot_id="discarded-later-state",
        status="truncated" if truncated else "completed",
        stop_reason="max_sequence_tokens" if truncated else None,
    )
    driver.tick()
    grade = next(key for key, job in queue.jobs.items() if job["kind"] == "grade")
    submission = next(row[2] for row in queue.requests
                      if row[0] == "POST" and row[1] == "/v1/jobs" and row[2]["kind"] == "grade")
    assert submission["spec"]["snapshot_id"] == "retained-prefix-state"
    queue.finish(grade, resolved=resolved)
    driver.tick()
    output = adapter.get(body["rollout_job_id"])
    if actual_tokens > body["max_sequence_tokens"]:
        assert output["status"] == "failed" and output["actual_samples"] == 0
        assert "exceeds max_sequence_tokens" in output["stop_reason"]
        return
    assert output["status"] == "completed"
    trajectory = output["trajectories"][0]
    assert trajectory["reward"] == expected and trajectory["metadata"]["raw_reward"] == float(resolved)
    assert trajectory["metadata"]["graded_snapshot_id"] == "retained-prefix-state"
    assert trajectory["metadata"]["training_token_count"] == actual_tokens
