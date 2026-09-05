"""Sequential rollout baseline.

The strategy is deliberately branch-free, but it is executable: when the
service is given a :class:`~swebench.rollout_groups.runner.ModelClient`, each
slot is sent to that client and the response is exported as a validated
trajectory.  A deterministic fallback remains available for protocol tests
that do not have a model endpoint.
"""

from __future__ import annotations

from typing import Any

from ..protocol import GeneratedSpan, RolloutGroupRequest, RolloutGroupResult, Trajectory
from ..runner import RolloutContext


class SequentialRolloutStrategy:
    """Run one independent model call for every allocated sample slot.

    ``allow_deterministic_fallback`` is intended for unit tests only.  A
    production service should provide ``context.model_client`` and set it to
    ``False`` so a missing model binding fails loudly instead of producing
    synthetic training data.
    """

    def __init__(self, *, allow_deterministic_fallback: bool = True):
        self.allow_deterministic_fallback = allow_deterministic_fallback

    def run(self, request: RolloutGroupRequest, context: RolloutContext) -> RolloutGroupResult:
        trajectories = []
        for slot in request.sample_slots[: request.max_samples]:
            context.check_cancelled()
            prompt_tokens = list(request.prompt_token_ids)
            sandbox = None
            if context.environment_provider is not None:
                sandbox = context.environment_provider.spawn(request)
            if context.model_client is None:
                try:
                    if not self.allow_deterministic_fallback:
                        raise RuntimeError("sequential rollout requires a model client")
                    response = {
                        "output_token_ids": [1000 + slot.sample_index],
                        "text": f"sample-{slot.sample_index}",
                        "finish_reason": "stop",
                    }
                finally:
                    if sandbox is not None:
                        context.environment_provider.destroy(sandbox)
            else:
                try:
                    response = context.model_client.generate(
                        endpoint=request.model_endpoint,
                        prompt_token_ids=prompt_tokens,
                        sampling_params=dict(request.sampling_params),
                        request={
                            "rollout_job_id": request.rollout_job_id,
                            "prompt_group_id": request.prompt_group_id,
                            "sample_slot_id": slot.sample_slot_id,
                            "sample_index": slot.sample_index,
                            "prompt": request.prompt,
                            "sandbox_id": getattr(sandbox, "sandbox_id", None),
                        },
                    )
                finally:
                    if sandbox is not None:
                        context.environment_provider.destroy(sandbox)
            if not isinstance(response, dict):
                raise ValueError("model client must return a JSON object")
            output_tokens = _output_tokens(response)
            if not output_tokens:
                raise ValueError("model response must contain output_token_ids or output_ids")
            response_text = str(response.get("text", response.get("response_text", "")))
            if not response_text:
                response_text = f"sample-{slot.sample_index}"
            weight_version = str(
                response.get("weight_version")
                or request.expected_weight_version
                or "unknown"
            )
            span = GeneratedSpan(
                response_id=f"{request.rollout_job_id}:response:{slot.sample_index}",
                start=len(prompt_tokens),
                end=len(prompt_tokens) + len(output_tokens),
                input_token_ids=tuple(prompt_tokens),
                output_token_ids=tuple(output_tokens),
                weight_version=weight_version,
                finish_reason=str(response.get("finish_reason", "stop")),
                output_token_log_probs=_optional_log_probs(response, len(output_tokens)),
            )
            trajectories.append(
                Trajectory(
                    sample_slot_id=slot.sample_slot_id,
                    branch_id=f"{request.rollout_job_id}:root:{slot.sample_index}",
                    messages=[
                        {"role": "user", "content": request.prompt},
                        {"role": "assistant", "content": response_text},
                    ],
                    token_ids=prompt_tokens + output_tokens,
                    prompt_length=len(prompt_tokens),
                    generated_spans=[span],
                    response_text=response_text,
                    metadata={
                        "strategy": "sequential",
                        "model_call": context.model_client is not None,
                    },
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


def _output_tokens(response: dict[str, Any]) -> list[int]:
    """Accept common SGLang/OpenAI adapter spellings without guessing text tokens."""
    raw = response.get("output_token_ids", response.get("output_ids"))
    if raw is None and isinstance(response.get("meta_info"), dict):
        raw = response["meta_info"].get("output_token_ids")
    if not isinstance(raw, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in raw):
        return []
    return list(raw)


def _optional_log_probs(response: dict[str, Any], length: int) -> tuple[float, ...] | None:
    raw = response.get("output_token_log_probs", response.get("logprobs"))
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError("model response log-probability count must match output token count")
    return tuple(float(value) for value in raw)
