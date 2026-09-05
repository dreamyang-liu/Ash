"""Deterministic sequential rollout used as the interface baseline.

This is intentionally a small strategy, not a branch algorithm. It exercises
the complete request -> strategy -> trajectory result path without requiring a
checkpoint implementation, a model provider, or a benchmark-specific harness.
Production strategies can replace it through ``RolloutStrategy``.
"""

from __future__ import annotations

from ..protocol import GeneratedSpan, RolloutGroupRequest, RolloutGroupResult, Trajectory
from ..runner import RolloutContext


class SequentialRolloutStrategy:
    """Return one deterministic trajectory for every allocated sample slot."""

    def run(self, request: RolloutGroupRequest, context: RolloutContext) -> RolloutGroupResult:
        trajectories = []
        for slot in request.sample_slots[: request.max_samples]:
            context.check_cancelled()
            output_token = 1000 + slot.sample_index
            prompt_tokens = list(request.prompt_token_ids)
            span = GeneratedSpan(
                response_id=f"{request.rollout_job_id}:response:{slot.sample_index}",
                start=len(prompt_tokens),
                end=len(prompt_tokens) + 1,
                input_token_ids=tuple(prompt_tokens),
                output_token_ids=(output_token,),
                weight_version=request.expected_weight_version or "unknown",
                finish_reason="stop",
            )
            trajectories.append(
                Trajectory(
                    sample_slot_id=slot.sample_slot_id,
                    branch_id=f"{request.rollout_job_id}:root:{slot.sample_index}",
                    messages=[
                        {"role": "user", "content": request.prompt},
                        {"role": "assistant", "content": f"sample-{slot.sample_index}"},
                    ],
                    token_ids=prompt_tokens + [output_token],
                    prompt_length=len(prompt_tokens),
                    generated_spans=[span],
                    response_text=f"sample-{slot.sample_index}",
                    metadata={"strategy": "sequential"},
                )
            )
        return RolloutGroupResult(
            rollout_job_id=request.rollout_job_id,
            prompt_group_id=request.prompt_group_id,
            status="completed",
            max_samples=request.max_samples,
            trajectories=trajectories,
            consumed_budget={"model_calls": len(trajectories), "tool_calls": 0},
        )
