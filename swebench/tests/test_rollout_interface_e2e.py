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
