"""Version 3: cleaned messages, image names, and Ash-owned rewards."""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import math

from rl_driver.protocol import SampleSlot, _nonnegative_int, _required_string

MESSAGE_VERSION = "ash-rollout-v3"


@dataclass(frozen=True)
class MessageBudget:
    max_wall_time_seconds: float

    @classmethod
    def from_dict(cls, value: object) -> "MessageBudget":
        if not isinstance(value, dict) or set(value) != {"max_wall_time_seconds"}:
            raise ValueError("v3 budgets contains only max_wall_time_seconds; use per-trajectory max_turns")
        wall_time = value["max_wall_time_seconds"]
        if type(wall_time) not in (int, float) or not math.isfinite(wall_time) or wall_time <= 0:
            raise ValueError("budgets.max_wall_time_seconds must be finite and positive")
        return cls(float(wall_time))


@dataclass(frozen=True)
class MessageRequest:
    rollout_job_id: str
    rollout_id: int
    prompt_group_id: str
    task_id: str
    image: str
    prompt: str | list
    model_endpoint: str
    sample_slots: tuple
    max_samples: int
    minimum_returned_samples: int
    max_turns: int
    sampling_params: dict
    budgets: MessageBudget
    model: str | None = None
    finalization_timeout_seconds: float = 1800.0
    protocol_version: str = MESSAGE_VERSION
    branching: bool = False

    @classmethod
    def from_dict(cls, body):
        if not isinstance(body, dict) or set(body) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown message request fields")
        if body.get("protocol_version") != MESSAGE_VERSION:
            raise ValueError("Expected ash-rollout-v3")
        value = deepcopy(body)
        if type(value.get("branching", False)) is not bool:
            raise ValueError("branching must be boolean")
        for key in ("rollout_job_id", "prompt_group_id", "task_id", "image", "model_endpoint"):
            _required_string(value.get(key), key)
        _nonnegative_int(value.get("rollout_id"), "rollout_id")
        if value.get("model") is not None:
            _required_string(value["model"], "model")
        prompt = value.get("prompt")
        if not isinstance(prompt, (str, list)) or not prompt:
            raise ValueError("prompt must be nonempty text or messages")
        slots = value.get("sample_slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError("sample_slots must be nonempty")
        value["sample_slots"] = tuple(SampleSlot.from_dict(slot) for slot in slots)
        if any(len({getattr(s, key) for s in value["sample_slots"]}) != len(slots)
               for key in ("sample_slot_id", "sample_index")):
            raise ValueError("Sample slots must have unique ids and indices")
        for key in ("max_samples", "minimum_returned_samples", "max_turns"):
            if _nonnegative_int(value.get(key), key) < 1:
                raise ValueError(f"{key} must be positive")
        if not value["minimum_returned_samples"] <= value["max_samples"] <= len(slots):
            raise ValueError("Invalid sample count bounds")
        value.setdefault("sampling_params", {})
        if not isinstance(value["sampling_params"], dict):
            raise ValueError("sampling_params must be an object")
        value["budgets"] = MessageBudget.from_dict(value.get("budgets"))
        finalization = value.get("finalization_timeout_seconds", 1800.0)
        if type(finalization) not in (int, float) or not math.isfinite(finalization) or finalization <= 0:
            raise ValueError("finalization_timeout_seconds must be finite and positive")
        value["finalization_timeout_seconds"] = float(finalization)
        return cls(**value)

    def to_dict(self):
        value = asdict(self)
        value["sample_slots"] = list(value["sample_slots"])
        if not value["branching"]:
            del value["branching"]
        return value
