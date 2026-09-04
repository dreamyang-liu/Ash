from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

from .checkpoints import CheckpointRecord


@dataclass(frozen=True)
class BranchPointSignals:
    """Normalized signals attached to one checkpoint candidate.

    Values are expected in [0, 1]. Missing signals default to zero rather than being
    guessed by the selector. ``commitment`` represents how much the trajectory has
    made state-changing/route-constraining progress; ``remaining_budget`` represents
    the fraction of the original rollout budget still available.
    """

    uncertainty: float = 0.0
    novelty: float = 0.0
    commitment: float = 0.0
    value: float = 0.0
    verifier_progress: float = 0.0
    remaining_budget: float = 0.0
    estimated_branch_cost: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class BranchPointCandidate:
    task_id: str
    trajectory_id: str
    step_id: int
    snapshot_id: str
    prefix_hash: str
    signals: BranchPointSignals = field(default_factory=BranchPointSignals)
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        if self.step_id < 0:
            raise ValueError("step_id must be >= 0")
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        if not self.prefix_hash:
            raise ValueError("prefix_hash must be non-empty")


@dataclass(frozen=True)
class BranchPointScore:
    candidate: BranchPointCandidate
    score: float
    components: dict[str, float]


class BranchPointPolicy(Protocol):
    def rank(self, candidates: Sequence[BranchPointCandidate]) -> list[BranchPointScore]: ...

    def select(
        self,
        candidates: Sequence[BranchPointCandidate],
        *,
        count: int = 1,
    ) -> list[BranchPointCandidate]: ...


@dataclass(frozen=True)
class BranchSelectionWeights:
    uncertainty: float = 1.0
    novelty: float = 0.5
    commitment: float = 0.5
    value_gap: float = 0.5
    verifier_gap: float = 0.5
    remaining_budget: float = 0.5
    branch_cost: float = 0.5


class WeightedBranchPointPolicy:
    """Transparent first policy for ranking saved checkpoints.

    The policy favors uncertain/novel states with useful budget left, and can favor
    states where current value/verifier progress leaves room for improvement. Branch
    cost is subtracted. The weights are deliberately explicit so this baseline can
    later be replaced by a learned value/branching policy without changing the
    candidate interface.
    """

    def __init__(self, weights: BranchSelectionWeights | None = None):
        self.weights = weights or BranchSelectionWeights()

    def score(self, candidate: BranchPointCandidate) -> BranchPointScore:
        s = candidate.signals
        components = {
            "uncertainty": self.weights.uncertainty * s.uncertainty,
            "novelty": self.weights.novelty * s.novelty,
            "commitment": self.weights.commitment * s.commitment,
            "value_gap": self.weights.value_gap * (1.0 - s.value),
            "verifier_gap": self.weights.verifier_gap * (1.0 - s.verifier_progress),
            "remaining_budget": self.weights.remaining_budget * s.remaining_budget,
            "branch_cost": -self.weights.branch_cost * s.estimated_branch_cost,
        }
        return BranchPointScore(
            candidate=candidate,
            score=sum(components.values()),
            components=components,
        )

    def rank(self, candidates: Sequence[BranchPointCandidate]) -> list[BranchPointScore]:
        scored = [self.score(candidate) for candidate in candidates]
        return sorted(
            scored,
            key=lambda item: (item.score, item.candidate.step_id),
            reverse=True,
        )

    def select(
        self,
        candidates: Sequence[BranchPointCandidate],
        *,
        count: int = 1,
    ) -> list[BranchPointCandidate]:
        if count < 1:
            raise ValueError("count must be >= 1")
        return [item.candidate for item in self.rank(candidates)[:count]]


def signals_from_bridge_health(
    health: dict[str, Any],
    *,
    uncertainty: float = 0.0,
    novelty: float = 0.0,
    value: float = 0.0,
    verifier_progress: float = 0.0,
    estimated_branch_cost: float = 0.0,
) -> BranchPointSignals:
    """Convert current MCP/Ash telemetry into normalized branch-point signals."""
    completed = int(health.get("completed_tool_calls") or 0)
    maximum = health.get("max_tool_calls")
    effects = health.get("tool_effect_counts") or {}
    mutation = int(effects.get("mutation") or 0)
    commitment = min(1.0, mutation / completed) if completed > 0 else 0.0
    if maximum is None or int(maximum) <= 0:
        remaining_budget = 0.0
    else:
        remaining_budget = max(0.0, min(1.0, (int(maximum) - completed) / int(maximum)))
    return BranchPointSignals(
        uncertainty=uncertainty,
        novelty=novelty,
        commitment=commitment,
        value=value,
        verifier_progress=verifier_progress,
        remaining_budget=remaining_budget,
        estimated_branch_cost=estimated_branch_cost,
    )


class LatestCheckpointPolicy:
    """Simple baseline: branch from the latest available saved checkpoint."""

    def rank(self, candidates: Sequence[BranchPointCandidate]) -> list[BranchPointScore]:
        ordered = sorted(candidates, key=lambda item: item.step_id, reverse=True)
        return [
            BranchPointScore(candidate=item, score=float(item.step_id), components={"step": float(item.step_id)})
            for item in ordered
        ]

    def select(
        self,
        candidates: Sequence[BranchPointCandidate],
        *,
        count: int = 1,
    ) -> list[BranchPointCandidate]:
        if count < 1:
            raise ValueError("count must be >= 1")
        return [item.candidate for item in self.rank(candidates)[:count]]
