from __future__ import annotations

import json
import threading

from swebench.rollout_groups.profiling import JSONLProfileWriter, trajectory_profile
from swebench.rollout_groups.protocol import (
    GeneratedSpan,
    RolloutGroupRequest,
    RolloutGroupResult,
    Trajectory,
)


def _request(job_id: str = "job") -> RolloutGroupRequest:
    return RolloutGroupRequest.from_dict(
        {
            "rollout_job_id": job_id,
            "rollout_id": 3,
            "prompt_group_id": "group",
            "task_id": "task",
            "environment_ref": {
                "kind": "template",
                "id": "env",
                "revision": "v1",
                "resource_profile": "standard",
            },
            "sample_slots": [{"sample_slot_id": "slot", "sample_index": 0}],
            "max_samples": 1,
            "minimum_returned_samples": 1,
            "prompt": "task",
            "prompt_token_ids": [1, 2],
            "model_endpoint": "http://model",
            "expected_weight_version": "7",
            "return_rollout_logprobs": False,
            "sampling_params": {},
            "budgets": {
                "max_model_calls": 2,
                "max_tool_calls": 1,
                "max_wall_time_seconds": 10,
            },
        }
    )


def _result(request: RolloutGroupRequest) -> RolloutGroupResult:
    trajectory = Trajectory(
        sample_slot_id="slot",
        branch_id=f"{request.rollout_job_id}:root",
        messages=[
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call", "function": {"name": "shell"}}],
            },
            {"role": "tool", "tool_call_id": "call", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        prompt_length=2,
        generated_spans=[
            GeneratedSpan("r1", 2, 4, (1, 2), (3, 4), "7", "tool_calls"),
            GeneratedSpan("r2", 6, 8, (1, 2, 3, 4, 5, 6), (7, 8), "7", "stop"),
        ],
        response_text="done",
        reward=1.0,
        metadata={"verifier_seconds": 1.25},
    )
    return RolloutGroupResult(
        rollout_job_id=request.rollout_job_id,
        prompt_group_id="group",
        status="completed",
        max_samples=1,
        trajectories=[trajectory],
        consumed_budget={"model_calls": 2, "tool_calls": 1},
    )


def test_profile_uses_exact_session_tokens_without_retokenizing():
    request = _request()
    record = trajectory_profile(request, _result(request), _result(request).trajectories[0])

    assert record["prompt_tokens"] == 2
    assert record["assistant_generated_tokens"] == 4
    assert record["non_generated_response_tokens"] == 2
    assert record["final_transcript_tokens"] == 8
    assert record["peak_context_tokens"] == 6
    assert record["all_model_calls_input_tokens"] == 8
    assert record["model_call_count"] == 2
    assert record["tool_call_count"] == 1


def test_jsonl_writer_keeps_concurrent_records_whole(tmp_path):
    path = tmp_path / "profile.jsonl"
    writer = JSONLProfileWriter(path)

    def write(index: int) -> None:
        request = _request(f"job-{index}")
        writer.write(request, _result(request), elapsed_seconds=index + 0.5)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 8
    assert {row["rollout_job_id"] for row in rows} == {f"job-{i}" for i in range(8)}
    assert {row["wall_time_seconds"] for row in rows} == {i + 0.5 for i in range(8)}
