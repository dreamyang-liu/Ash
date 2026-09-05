from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from swebench.rollout_groups.protocol import (
    GeneratedSpan,
    RolloutGroupRequest,
    RolloutGroupResult,
    Trajectory,
)
from swebench.rollout_groups.runner import GroupRolloutService
from swebench.rollout_groups.server import RolloutGroupsHTTPServer


def request_payload(job_id="job-1"):
    return {
        "rollout_job_id": job_id,
        "rollout_id": 0,
        "prompt_group_id": "group-1",
        "sample_slots": [{"sample_slot_id": "slot-0", "sample_index": 0}],
        "max_samples": 1,
        "minimum_returned_samples": 1,
        "prompt": "hello",
        "prompt_token_ids": [1],
        "model_endpoint": "http://model",
        "expected_weight_version": "7",
        "return_rollout_logprobs": False,
        "sampling_params": {},
        "budgets": {"max_model_calls": 2, "max_tool_calls": 3,
                     "max_wall_time_seconds": 10},
    }


def complete_result(request, _context):
    span = GeneratedSpan(
        response_id="response-1", start=1, end=2,
        input_token_ids=(1,), output_token_ids=(2,), weight_version="7",
        finish_reason="stop",
    )
    trajectory = Trajectory(
        sample_slot_id=request.sample_slots[0].sample_slot_id,
        branch_id="root", messages=[{"role": "user", "content": "hello"}],
        token_ids=[1, 2], prompt_length=1, generated_spans=[span],
        response_text="world",
    )
    return RolloutGroupResult(
        rollout_job_id=request.rollout_job_id,
        prompt_group_id=request.prompt_group_id,
        status="completed", max_samples=1, trajectories=[trajectory],
    )


class _ResultStrategy:
    def __init__(self, fn):
        self.fn = fn

    def run(self, request, context):
        return self.fn(request, context)


def test_protocol_round_trip_and_span_validation():
    request = RolloutGroupRequest.from_dict(request_payload())
    assert request.to_dict()["prompt_token_ids"] == [1]
    with pytest.raises(ValueError, match="output_token_ids length"):
        GeneratedSpan.from_dict({
            "response_id": "r", "start": 1, "end": 3,
            "input_token_ids": [1], "output_token_ids": [2],
            "weight_version": "1", "finish_reason": "stop",
        })


def test_service_idempotency_and_cancel():
    service = GroupRolloutService(lambda request, context: _ResultStrategy(complete_result))
    request = RolloutGroupRequest.from_dict(request_payload())
    first = service.submit(request)
    second = service.submit(request)
    assert first.rollout_job_id == second.rollout_job_id
    deadline = time.time() + 2
    while time.time() < deadline and service.get(request.rollout_job_id).status in {"queued", "running"}:
        time.sleep(0.01)
    assert service.get(request.rollout_job_id).status == "completed"
    with pytest.raises(ValueError):
        service.submit(RolloutGroupRequest.from_dict(request_payload(job_id="job-1") | {
            "prompt_group_id": "different",
        }))


def test_http_endpoints():
    service = GroupRolloutService(lambda request, context: _ResultStrategy(complete_result))
    server = RolloutGroupsHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        body = json.dumps(request_payload("http-job")).encode()
        req = urllib.request.Request(base + "/rollout-groups", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            assert response.status == 202
            assert json.load(response)["rollout_job_id"] == "http-job"
        deadline = time.time() + 2
        while time.time() < deadline:
            with urllib.request.urlopen(base + "/rollout-groups/http-job") as response:
                result = json.load(response)
            if result["status"] == "completed":
                break
            time.sleep(0.01)
        assert result["actual_samples"] == 1
        assert result["trajectories"][0]["branch_id"] == "root"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_strategy_cannot_leave_job_non_terminal():
    def bad_strategy(request, _context):
        return _ResultStrategy(lambda req, ctx: RolloutGroupResult(
            rollout_job_id=req.rollout_job_id,
            prompt_group_id=req.prompt_group_id,
            status="running",
            max_samples=req.max_samples,
        ))

    service = GroupRolloutService(bad_strategy)
    request = RolloutGroupRequest.from_dict(request_payload("non-terminal"))
    service.submit(request)
    deadline = time.time() + 2
    while time.time() < deadline:
        result = service.get(request.rollout_job_id)
        if result.status == "failed":
            break
        time.sleep(0.01)
    assert result.status == "failed"
    assert "terminal" in (result.stop_reason or "")
