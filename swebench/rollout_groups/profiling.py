"""JSONL profiling records derived from the stable rollout wire contract."""

from __future__ import annotations

import json
import os
import threading
import hashlib
from dataclasses import asdict
from pathlib import Path

from .protocol import RolloutGroupRequest, RolloutGroupResult, Trajectory


PROFILE_SCHEMA_VERSION = "ash-rollout-profile-v1"


def _duration(metadata: dict, key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 3)
    return None


def trajectory_profile(
    request: RolloutGroupRequest,
    result: RolloutGroupResult,
    trajectory: Trajectory,
    *,
    elapsed_seconds: float | None = None,
) -> dict:
    """Build one auditable record without retokenizing model-visible text."""
    generated = sum(len(span.output_token_ids) for span in trajectory.generated_spans)
    response_tokens = len(trajectory.token_ids) - trajectory.prompt_length
    tool_calls = sum(
        len(message.get("tool_calls") or [])
        for message in trajectory.messages
        if message.get("role") == "assistant"
    )
    finish_reasons = [span.finish_reason for span in trajectory.generated_spans]
    token_bytes = ",".join(str(token) for token in trajectory.token_ids).encode()
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "rollout_job_id": request.rollout_job_id,
        "rollout_id": request.rollout_id,
        "prompt_group_id": request.prompt_group_id,
        "task_id": request.task_id,
        "sample_slot_id": trajectory.sample_slot_id,
        "branch_id": trajectory.branch_id,
        "parent_branch_id": trajectory.parent_branch_id,
        "branch_point_token_count": trajectory.branch_point_token_count,
        "environment_ref": request.environment_ref.to_dict(),
        "model": request.model,
        "sampling_params": dict(request.sampling_params),
        "seed": request.sampling_params.get("seed"),
        "expected_weight_version": request.expected_weight_version,
        "prompt_tokens": trajectory.prompt_length,
        "assistant_generated_tokens": generated,
        "non_generated_response_tokens": response_tokens - generated,
        "final_transcript_tokens": len(trajectory.token_ids),
        "train_sequence_tokens": len(trajectory.token_ids),
        "peak_context_tokens": max(
            len(span.input_token_ids) for span in trajectory.generated_spans
        ),
        "all_model_calls_input_tokens": sum(
            len(span.input_token_ids) for span in trajectory.generated_spans
        ),
        "model_call_count": len(trajectory.generated_spans),
        "generated_spans": [
            {
                "response_id": span.response_id,
                "start": span.start,
                "end": span.end,
                "input_tokens": len(span.input_token_ids),
                "output_tokens": len(span.output_token_ids),
                "weight_version": span.weight_version,
                "finish_reason": span.finish_reason,
            }
            for span in trajectory.generated_spans
        ],
        "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "tool_call_count": tool_calls,
        "finish_reasons": finish_reasons,
        "wall_time_seconds": elapsed_seconds,
        "model_time_seconds": _duration(trajectory.metadata, "model_time_seconds"),
        "tool_time_seconds": _duration(trajectory.metadata, "tool_time_seconds"),
        "sandbox_setup_seconds": _duration(
            trajectory.metadata, "sandbox_setup_seconds"
        ),
        "verifier_time_seconds": _duration(
            trajectory.metadata, "verifier_seconds"
        ),
        "trajectory_status": trajectory.status,
        "job_status": result.status,
        "stop_reason": result.stop_reason,
        "reward": trajectory.reward,
        "search_branches": result.search_branches,
        "requested_budget": asdict(request.budgets),
        "consumed_budget": dict(result.consumed_budget),
        "trajectory_metadata": dict(trajectory.metadata),
    }


class JSONLProfileWriter:
    """Append one durable line per returned trajectory from concurrent jobs."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(
        self,
        request: RolloutGroupRequest,
        result: RolloutGroupResult,
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        lines = [
            json.dumps(
                trajectory_profile(
                    request,
                    result,
                    trajectory,
                    elapsed_seconds=elapsed_seconds,
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for trajectory in result.trajectories
        ]
        if not lines:
            return
        with self._lock, self.path.open("a", encoding="utf-8") as output:
            output.write("\n".join(lines) + "\n")
