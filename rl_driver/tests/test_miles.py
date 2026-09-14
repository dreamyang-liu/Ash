from copy import deepcopy
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

from rl_driver.driver import Driver
from rl_driver.ledger import Ledger
from rl_driver.miles import MilesAdapter, internal_id
from rl_driver.protocol import RolloutGroupRequest, Trajectory
from rl_driver.server import create_app
from rl_driver.tests.test_driver import peer


def miles_request():
    # Verbatim request returned by the pinned haixin branch's HTTP e2e fixture.
    return json.loads((Path(__file__).parent / "fixtures/miles-request.json").read_text())


def config(body):
    return {"environment_catalog": {"environments": [{**body["environment_ref"], "spawn_ref": "prepared-final"}]},
            "profile": "codex", "run_defaults": {"slot": "codex", "model": "served-model"},
            "resources": {"standard": {"cpu": 2, "memory_mb": 8192}}}


def recorded_state(body, output=90, version="42"):
    inputs = body["prompt_token_ids"]
    return {"metadata": {"accumulated_token_ids": [*inputs, output]}, "records": [{
        "request": {"input_ids": inputs, "messages": [{"role": "user", "content": body["prompt"]}]},
        "response": {"id": f"response-{output}", "choices": [{"message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop", "meta_info": {"weight_version": version, "output_token_logprobs": [[-0.3, output]]}}]},
    }]}


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
        assert http.delete("/rollout-groups/" + body["rollout_job_id"]).json()["status"] == "completed"
    restored = MilesAdapter(Driver(backend, Ledger(tmp_path / "ledger")), config(body))
    assert restored.get(body["rollout_job_id"]) == result


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


def test_main_example_uses_original_request_type():
    example = json.loads((Path(__file__).parents[1] / "group.example.json").read_text())
    parsed = RolloutGroupRequest.from_dict(example)
    assert parsed.protocol_version == "ash-rollout-v2"
    assert len(parsed.sample_slots) == 2
