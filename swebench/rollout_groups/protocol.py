"""Wire types for the versioned ``/rollout-groups`` contract.

This module is intentionally dependency-light.  Miles can use its own Pydantic
models while Ash keeps the service usable from a plain Python process.  The
validation rules mirror Miles' contract and fail before a worker or sandbox is
created.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any


PROTOCOL_VERSION = "ash-rollout-v2"
JOB_STATUSES = {"queued", "running", "completed", "early_stopped", "failed", "cancelled"}
TRAJECTORY_STATUSES = {"completed", "truncated", "failed", "aborted"}
ENVIRONMENT_KINDS = {"image", "template", "snapshot"}
OCI_DIGEST = re.compile(r"sha256:[0-9a-fA-F]{64}\Z")


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RolloutBudget:
    max_model_calls: int
    max_tool_calls: int
    max_wall_time_seconds: float

    @classmethod
    def from_dict(cls, value: Any) -> "RolloutBudget":
        if not isinstance(value, dict):
            raise ValueError("budgets must be an object")
        _reject_unknown_keys(
            value,
            {"max_model_calls", "max_tool_calls", "max_wall_time_seconds"},
            "budgets",
        )
        model_calls = value.get("max_model_calls")
        tool_calls = value.get("max_tool_calls")
        wall_time = value.get("max_wall_time_seconds")
        if (
            not isinstance(model_calls, int)
            or isinstance(model_calls, bool)
            or model_calls <= 0
        ):
            raise ValueError("budgets.max_model_calls must be > 0")
        if (
            not isinstance(tool_calls, int)
            or isinstance(tool_calls, bool)
            or tool_calls < 0
        ):
            raise ValueError("budgets.max_tool_calls must be >= 0")
        if (
            not isinstance(wall_time, (int, float))
            or isinstance(wall_time, bool)
            or not math.isfinite(wall_time)
            or wall_time <= 0
        ):
            raise ValueError("budgets.max_wall_time_seconds must be > 0")
        return cls(model_calls, tool_calls, float(wall_time))


@dataclass(frozen=True)
class SampleSlot:
    sample_slot_id: str
    sample_index: int

    @classmethod
    def from_dict(cls, value: Any) -> "SampleSlot":
        if not isinstance(value, dict):
            raise ValueError("sample_slots entries must be objects")
        _reject_unknown_keys(value, {"sample_slot_id", "sample_index"}, "sample slot")
        return cls(
            _required_string(value.get("sample_slot_id"), "sample_slot_id"),
            _nonnegative_int(value.get("sample_index"), "sample_index"),
        )


@dataclass(frozen=True)
class EnvironmentRef:
    """Trusted logical reference to one sandbox creation artifact."""

    kind: str
    id: str
    revision: str
    resource_profile: str

    @classmethod
    def from_dict(cls, value: Any) -> "EnvironmentRef":
        if not isinstance(value, dict):
            raise ValueError("environment_ref must be an object")
        _reject_unknown_keys(
            value,
            {"kind", "id", "revision", "resource_profile"},
            "environment_ref",
        )
        kind = _required_string(value.get("kind"), "environment_ref.kind")
        if kind not in ENVIRONMENT_KINDS:
            raise ValueError(
                "environment_ref.kind must be one of: image, template, snapshot"
            )
        result = cls(
            kind=kind,
            id=_required_string(value.get("id"), "environment_ref.id"),
            revision=_required_string(
                value.get("revision"), "environment_ref.revision"
            ),
            resource_profile=_required_string(
                value.get("resource_profile"),
                "environment_ref.resource_profile",
            ),
        )
        if result.kind == "image" and not OCI_DIGEST.fullmatch(result.revision):
            raise ValueError(
                "environment_ref.revision must be a sha256 digest when kind is image"
            )
        return result

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RolloutGroupRequest:
    rollout_job_id: str
    rollout_id: int
    prompt_group_id: str
    task_id: str
    environment_ref: EnvironmentRef
    sample_slots: tuple[SampleSlot, ...]
    max_samples: int
    minimum_returned_samples: int
    prompt: str | list[dict[str, Any]]
    prompt_token_ids: tuple[int, ...]
    model_endpoint: str
    expected_weight_version: str | None
    return_rollout_logprobs: bool
    sampling_params: dict[str, Any]
    budgets: RolloutBudget
    # When set, Ash runs its agent loop through Miles' v2 Session Server.
    # ``model_endpoint`` remains the raw generation endpoint for simple
    # strategies and backwards compatibility.
    session_server_endpoint: str | None = None
    model: str | None = None
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, value: Any) -> "RolloutGroupRequest":
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        _reject_unknown_keys(
            value,
            {
                "protocol_version",
                "rollout_job_id",
                "rollout_id",
                "prompt_group_id",
                "task_id",
                "environment_ref",
                "sample_slots",
                "max_samples",
                "minimum_returned_samples",
                "prompt",
                "prompt_token_ids",
                "model_endpoint",
                "session_server_endpoint",
                "model",
                "expected_weight_version",
                "return_rollout_logprobs",
                "sampling_params",
                "budgets",
            },
            "request body",
        )
        version = value.get("protocol_version", PROTOCOL_VERSION)
        if version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version: {version!r}")
        raw_slots = value.get("sample_slots")
        if not isinstance(raw_slots, list) or not raw_slots:
            raise ValueError("sample_slots must be a non-empty list")
        slots = tuple(SampleSlot.from_dict(item) for item in raw_slots)
        if len({slot.sample_slot_id for slot in slots}) != len(slots):
            raise ValueError("sample_slot_id values must be unique")
        if len({slot.sample_index for slot in slots}) != len(slots):
            raise ValueError("sample_index values must be unique")
        max_samples = value.get("max_samples")
        minimum = value.get("minimum_returned_samples", 1)
        if (
            not isinstance(max_samples, int)
            or isinstance(max_samples, bool)
            or max_samples <= 0
            or max_samples > len(slots)
        ):
            raise ValueError("max_samples must be between 1 and the number of sample slots")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or minimum < 1
            or minimum > max_samples
        ):
            raise ValueError("minimum_returned_samples must be between 1 and max_samples")
        prompt = value.get("prompt")
        if not isinstance(prompt, (str, list)):
            raise ValueError("prompt must be a string or message list")
        if isinstance(prompt, list) and any(not isinstance(message, dict) for message in prompt):
            raise ValueError("prompt messages must be objects")
        token_ids = value.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(
            not isinstance(token, int) or isinstance(token, bool) for token in token_ids
        ):
            raise ValueError("prompt_token_ids must be a non-empty integer list")
        sampling = value.get("sampling_params", {})
        if not isinstance(sampling, dict):
            raise ValueError("sampling_params must be an object")
        return cls(
            rollout_job_id=_required_string(value.get("rollout_job_id"), "rollout_job_id"),
            rollout_id=_nonnegative_int(value.get("rollout_id"), "rollout_id"),
            prompt_group_id=_required_string(value.get("prompt_group_id"), "prompt_group_id"),
            task_id=_required_string(value.get("task_id"), "task_id"),
            environment_ref=EnvironmentRef.from_dict(value.get("environment_ref")),
            sample_slots=slots,
            max_samples=max_samples,
            minimum_returned_samples=minimum,
            prompt=prompt,
            prompt_token_ids=tuple(token_ids),
            model_endpoint=_required_string(value.get("model_endpoint"), "model_endpoint"),
            expected_weight_version=(
                None if value.get("expected_weight_version") is None
                else _required_string(value.get("expected_weight_version"), "expected_weight_version")
            ),
            return_rollout_logprobs=_strict_bool(
                value.get("return_rollout_logprobs", False), "return_rollout_logprobs"
            ),
            sampling_params=dict(sampling),
            budgets=RolloutBudget.from_dict(value.get("budgets")),
            session_server_endpoint=(
                None if value.get("session_server_endpoint") is None
                else _required_string(value.get("session_server_endpoint"), "session_server_endpoint")
            ),
            model=(None if value.get("model") is None else _required_string(value.get("model"), "model")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sample_slots"] = [asdict(slot) for slot in self.sample_slots]
        data["environment_ref"] = self.environment_ref.to_dict()
        data["prompt_token_ids"] = list(self.prompt_token_ids)
        return data


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class GeneratedSpan:
    response_id: str
    start: int
    end: int
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    weight_version: str
    finish_reason: str
    output_token_log_probs: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "GeneratedSpan":
        if not isinstance(value, dict):
            raise ValueError("generated_spans entries must be objects")
        _reject_unknown_keys(
            value,
            {
                "response_id",
                "start",
                "end",
                "input_token_ids",
                "output_token_ids",
                "output_token_log_probs",
                "weight_version",
                "finish_reason",
            },
            "generated span",
        )
        start = _nonnegative_int(value.get("start"), "generated span start")
        end = value.get("end")
        if not isinstance(end, int) or isinstance(end, bool) or end <= start:
            raise ValueError("generated span end must be greater than start")
        output = value.get("output_token_ids")
        inputs = value.get("input_token_ids")
        if not isinstance(inputs, list) or not isinstance(output, list):
            raise ValueError("generated span token ids must be lists")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in inputs + output):
            raise ValueError("generated span token ids must be integers")
        if len(output) != end - start:
            raise ValueError("output_token_ids length must equal span length")
        logs = value.get("output_token_log_probs")
        if logs is not None:
            if not isinstance(logs, list) or len(logs) != len(output):
                raise ValueError("output_token_log_probs length must equal span length")
            if any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(item)
                for item in logs
            ):
                raise ValueError(
                    "output_token_log_probs must contain finite numbers"
                )
        return cls(
            response_id=_required_string(value.get("response_id"), "response_id"),
            start=start,
            end=end,
            input_token_ids=tuple(inputs),
            output_token_ids=tuple(output),
            weight_version=_required_string(value.get("weight_version"), "weight_version"),
            finish_reason=_required_string(value.get("finish_reason"), "finish_reason"),
            output_token_log_probs=None if logs is None else tuple(float(item) for item in logs),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_token_ids"] = list(self.input_token_ids)
        data["output_token_ids"] = list(self.output_token_ids)
        if self.output_token_log_probs is not None:
            data["output_token_log_probs"] = list(self.output_token_log_probs)
        return data


@dataclass(frozen=True)
class Trajectory:
    sample_slot_id: str
    branch_id: str
    messages: list[dict[str, Any]]
    token_ids: list[int]
    prompt_length: int
    generated_spans: list[GeneratedSpan]
    response_text: str
    status: str = "completed"
    parent_branch_id: str | None = None
    branch_point_token_count: int | None = None
    reward: float | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any) -> "Trajectory":
        if not isinstance(value, dict):
            raise ValueError("trajectory entries must be objects")
        _reject_unknown_keys(
            value,
            {
                "sample_slot_id",
                "branch_id",
                "parent_branch_id",
                "branch_point_token_count",
                "messages",
                "token_ids",
                "prompt_length",
                "generated_spans",
                "response_text",
                "reward",
                "status",
                "metadata",
            },
            "trajectory",
        )
        status = value.get("status", "completed")
        if status not in TRAJECTORY_STATUSES:
            raise ValueError(f"unknown trajectory status: {status!r}")
        tokens = value.get("token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("trajectory token_ids must be a non-empty list")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
            raise ValueError("trajectory token_ids must be integers")
        prompt_length = value.get("prompt_length")
        if (
            not isinstance(prompt_length, int)
            or isinstance(prompt_length, bool)
            or prompt_length < 1
            or prompt_length >= len(tokens)
        ):
            raise ValueError("prompt_length must leave at least one response token")
        messages = value.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("trajectory messages must be a non-empty list")
        if any(not isinstance(message, dict) for message in messages):
            raise ValueError("trajectory messages must contain objects")
        spans = [GeneratedSpan.from_dict(item) for item in value.get("generated_spans", [])]
        if not spans:
            raise ValueError("trajectory must contain at least one generated span")
        previous_end = prompt_length
        for span in spans:
            if span.start < previous_end or span.end > len(tokens):
                raise ValueError("generated spans must be ordered and within token_ids")
            if list(span.input_token_ids) != tokens[:span.start]:
                raise ValueError(f"input token prefix mismatch for {span.response_id!r}")
            if list(span.output_token_ids) != tokens[span.start:span.end]:
                raise ValueError(f"output token mismatch for {span.response_id!r}")
            previous_end = span.end
        parent = value.get("parent_branch_id")
        branch_point = value.get("branch_point_token_count")
        if parent is not None:
            parent = _required_string(parent, "parent_branch_id")
        if parent is None and branch_point is not None:
            raise ValueError("root trajectory cannot declare branch_point_token_count")
        if parent is not None and (
            not isinstance(branch_point, int)
            or isinstance(branch_point, bool)
            or branch_point < 0
        ):
            raise ValueError("child trajectory must declare branch_point_token_count")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("trajectory metadata must be an object")
        response_text = value.get("response_text", "")
        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string")
        reward = value.get("reward")
        if reward is not None and not isinstance(reward, dict):
            if (
                not isinstance(reward, (int, float))
                or isinstance(reward, bool)
                or not math.isfinite(reward)
            ):
                raise ValueError("trajectory reward must be a finite number or object")
        return cls(
            sample_slot_id=_required_string(value.get("sample_slot_id"), "sample_slot_id"),
            branch_id=_required_string(value.get("branch_id"), "branch_id"),
            parent_branch_id=parent,
            branch_point_token_count=branch_point,
            messages=messages,
            token_ids=[int(token) for token in tokens],
            prompt_length=prompt_length,
            generated_spans=spans,
            response_text=response_text,
            reward=reward,
            status=status,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_spans"] = [span.to_dict() for span in self.generated_spans]
        return data


@dataclass(frozen=True)
class RolloutGroupResult:
    rollout_job_id: str
    prompt_group_id: str
    status: str
    max_samples: int
    trajectories: list[Trajectory] = field(default_factory=list)
    stop_reason: str | None = None
    search_branches: int = 0
    consumed_budget: dict[str, int | float] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _required_string(self.rollout_job_id, "rollout_job_id")
        _required_string(self.prompt_group_id, "prompt_group_id")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol_version: {self.protocol_version!r}"
            )
        if (
            not isinstance(self.max_samples, int)
            or isinstance(self.max_samples, bool)
            or self.max_samples <= 0
        ):
            raise ValueError("max_samples must be > 0")
        if self.status not in JOB_STATUSES:
            raise ValueError(f"unknown job status: {self.status!r}")
        if self.status in {"queued", "running"} and self.trajectories:
            raise ValueError("non-terminal result cannot contain trajectories")
        if (
            not isinstance(self.search_branches, int)
            or isinstance(self.search_branches, bool)
            or self.search_branches < 0
        ):
            raise ValueError("search_branches must be a non-negative integer")
        if len(self.trajectories) > self.max_samples:
            raise ValueError("trajectory count exceeds max_samples")
        slots = [item.sample_slot_id for item in self.trajectories]
        branches = [item.branch_id for item in self.trajectories]
        if len(slots) != len(set(slots)) or len(branches) != len(set(branches)):
            raise ValueError("trajectory slot and branch ids must be unique")
        for name, value in self.consumed_budget.items():
            _required_string(name, "consumed_budget key")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"consumed_budget[{name!r}] must be a finite non-negative number")

    @property
    def actual_samples(self) -> int:
        return len(self.trajectories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "rollout_job_id": self.rollout_job_id,
            "prompt_group_id": self.prompt_group_id,
            "status": self.status,
            "max_samples": self.max_samples,
            "actual_samples": self.actual_samples,
            "stop_reason": self.stop_reason,
            "search_branches": self.search_branches,
            "consumed_budget": self.consumed_budget,
            "trajectories": [item.to_dict() for item in self.trajectories],
        }


@dataclass(frozen=True)
class RolloutSubmission:
    rollout_job_id: str
    status: str
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _required_string(self.rollout_job_id, "rollout_job_id")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol_version: {self.protocol_version!r}"
            )
        if self.status not in JOB_STATUSES:
            raise ValueError(f"unknown submission status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "rollout_job_id": self.rollout_job_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class RolloutDeletion:
    rollout_job_id: str
    status: str
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _required_string(self.rollout_job_id, "rollout_job_id")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol_version: {self.protocol_version!r}"
            )
        if self.status not in {"completed", "early_stopped", "failed", "cancelled"}:
            raise ValueError(f"deletion status must be terminal, got {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "rollout_job_id": self.rollout_job_id,
            "status": self.status,
        }
