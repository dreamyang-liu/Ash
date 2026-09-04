"""Adaptive checkpoint cadence controllers for long-horizon agent rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OverheadBudgetCheckpointPolicy:
    """Choose checkpoint spacing from an explicit snapshot-overhead budget.

    After the first materialized checkpoint, the policy estimates non-checkpoint
    rollout cost per step and chooses the smallest stride whose predicted snapshot
    fraction is no larger than ``target_snapshot_fraction``.

    The controller intentionally reasons only about snapshot materialization. Cache
    metadata registration and optional workspace-fingerprint cost are reported
    separately and can be added to the budget later if an experiment enables them.
    """

    target_snapshot_fraction: float = 0.05
    initial_stride: int = 10
    min_stride: int = 5
    max_stride: int = 100

    last_checkpoint_step: int = 0
    next_checkpoint_step: int = 0
    cumulative_snapshot_ms: float = 0.0
    observed_snapshot_ms: float | None = None
    chosen_stride: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_snapshot_fraction < 1.0:
            raise ValueError("target_snapshot_fraction must be in (0, 1)")
        if self.initial_stride < 1:
            raise ValueError("initial_stride must be >= 1")
        if self.min_stride < 1:
            raise ValueError("min_stride must be >= 1")
        if self.max_stride < self.min_stride:
            raise ValueError("max_stride must be >= min_stride")
        self.chosen_stride = self._clamp(self.initial_stride)
        self.next_checkpoint_step = self.chosen_stride

    def _clamp(self, stride: int) -> int:
        return max(self.min_stride, min(self.max_stride, int(stride)))

    def should_checkpoint(self, step_id: int) -> bool:
        return step_id > 0 and step_id >= self.next_checkpoint_step

    def observe_checkpoint(
        self,
        *,
        step_id: int,
        rollout_elapsed_ms: float | None,
        snapshot_ms: float,
    ) -> int:
        """Update the cadence after one materialized checkpoint.

        ``rollout_elapsed_ms`` is the total wall time measured immediately before
        the current snapshot. It includes earlier snapshot pauses, so those are
        subtracted before estimating model/tool work per step.
        """
        if step_id <= 0:
            raise ValueError("step_id must be > 0")
        if snapshot_ms < 0:
            raise ValueError("snapshot_ms must be >= 0")

        self.cumulative_snapshot_ms += snapshot_ms
        if self.observed_snapshot_ms is None:
            self.observed_snapshot_ms = snapshot_ms
        else:
            # Mild smoothing avoids one pathological Docker commit controlling the
            # rest of a very long trajectory.
            self.observed_snapshot_ms = 0.7 * self.observed_snapshot_ms + 0.3 * snapshot_ms

        stride = self.chosen_stride
        if rollout_elapsed_ms is not None and rollout_elapsed_ms > 0 and step_id > 0:
            previous_snapshot_ms = max(0.0, self.cumulative_snapshot_ms - snapshot_ms)
            non_snapshot_ms = max(1e-9, rollout_elapsed_ms - previous_snapshot_ms)
            per_step_ms = non_snapshot_ms / step_id
            f = self.target_snapshot_fraction
            required = self.observed_snapshot_ms * (1.0 - f) / (f * per_step_ms)
            stride = self._clamp(max(1, math.ceil(required)))

        self.last_checkpoint_step = step_id
        self.chosen_stride = stride
        self.next_checkpoint_step = step_id + stride
        return stride

    def observe_reuse(self, *, step_id: int) -> int:
        """Advance the schedule when an equivalent materialized snapshot is reused."""
        if step_id <= 0:
            raise ValueError("step_id must be > 0")
        self.last_checkpoint_step = step_id
        self.next_checkpoint_step = step_id + self.chosen_stride
        return self.chosen_stride

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "target_snapshot_fraction": self.target_snapshot_fraction,
            "initial_stride": self.initial_stride,
            "min_stride": self.min_stride,
            "max_stride": self.max_stride,
            "chosen_stride": self.chosen_stride,
            "last_checkpoint_step": self.last_checkpoint_step,
            "next_checkpoint_step": self.next_checkpoint_step,
            "cumulative_snapshot_ms": round(self.cumulative_snapshot_ms, 3),
            "observed_snapshot_ms": (
                round(self.observed_snapshot_ms, 3)
                if self.observed_snapshot_ms is not None
                else None
            ),
        }
