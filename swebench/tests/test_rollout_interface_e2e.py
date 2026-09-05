from __future__ import annotations

import json
import threading
import time
import urllib.request

from swebench.rollout_groups.runner import GroupRolloutService
from swebench.rollout_groups.server import RolloutGroupsHTTPServer
from swebench.rollout_groups.strategies.sequential import SequentialRolloutStrategy


def _request(job_id: str) -> dict:
    return {
        "protocol_version": "ash-rollout-v1",
        "rollout_job_id": job_id,
        "rollout_id": 1,
        "prompt_group_id": "group-e2e",
        "sample_slots": [
            {"sample_slot_id": f"{job_id}:slot:0", "sample_index": 0},
            {"sample_slot_id": f"{job_id}:slot:1", "sample_index": 1},
        ],
        "max_samples": 2,
        "minimum_returned_samples": 2,
        "prompt": "run the task",
        "prompt_token_ids": [11, 12],
        "model_endpoint": "http://unused",
        "expected_weight_version": "42",
        "return_rollout_logprobs": False,
        "sampling_params": {},
        "budgets": {"max_model_calls": 4, "max_tool_calls": 4, "max_wall_time_seconds": 5},
    }


def _wait(base: str, job_id: str) -> dict:
    deadline = time.time() + 2
    while time.time() < deadline:
        with urllib.request.urlopen(f"{base}/rollout-groups/{job_id}") as response:
            result = json.load(response)
        if result["status"] not in {"queued", "running"}:
            return result
        time.sleep(0.01)
    raise AssertionError("rollout did not finish")


def test_http_to_sequential_strategy_end_to_end():
    service = GroupRolloutService(
        lambda _request, _context: SequentialRolloutStrategy(),
    )
    server = RolloutGroupsHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    payload = _request("baseline-e2e")
    try:
        request = urllib.request.Request(
            f"{base}/rollout-groups",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
        result = _wait(base, payload["rollout_job_id"])
        assert result["status"] == "completed"
        assert result["actual_samples"] == 2
        assert [item["sample_slot_id"] for item in result["trajectories"]] == [
            "baseline-e2e:slot:0",
            "baseline-e2e:slot:1",
        ]
        assert result["trajectories"][0]["metadata"]["strategy"] == "sequential"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        index = kwargs["request"]["sample_index"]
        return {
            "output_token_ids": [3000 + index, 4000 + index],
            "text": f"real-sample-{index}",
            "weight_version": "42",
            "finish_reason": "stop",
        }


class _Environment:
    def __init__(self):
        self.spawned = []
        self.destroyed = []

    def spawn(self, request):
        sandbox = type("Sandbox", (), {"sandbox_id": f"sandbox-{len(self.spawned)}"})()
        self.spawned.append(sandbox)
        return sandbox

    def destroy(self, sandbox):
        self.destroyed.append(sandbox.sandbox_id)

def test_http_to_executable_sequential_strategy_uses_model_and_environment():
    model = _Model()
    environment = _Environment()
    service = GroupRolloutService(
        lambda _request, _context: SequentialRolloutStrategy(allow_deterministic_fallback=False),
        model_client=model,
        environment_provider=environment,
    )
    server = RolloutGroupsHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    payload = _request("executable-sequential-e2e")
    try:
        request = urllib.request.Request(
            f"{base}/rollout-groups", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
        result = _wait(base, payload["rollout_job_id"])
        assert result["status"] == "completed"
        assert result["actual_samples"] == 2
        assert [call["request"]["sample_index"] for call in model.calls] == [0, 1]
        assert [call["request"]["sandbox_id"] for call in model.calls] == ["sandbox-0", "sandbox-1"]
        assert environment.destroyed == ["sandbox-0", "sandbox-1"]
        assert result["trajectories"][0]["token_ids"] == [11, 12, 3000, 4000]
        assert result["trajectories"][0]["metadata"]["model_call"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
