import pytest

from ash_sandbox.branch_selection import (
    BranchPointCandidate,
    BranchPointSignals,
    BranchSelectionWeights,
    LatestCheckpointPolicy,
    WeightedBranchPointPolicy,
    signals_from_bridge_health,
)


def _candidate(step: int, **signals) -> BranchPointCandidate:
    return BranchPointCandidate(
        task_id="task",
        trajectory_id="traj",
        step_id=step,
        snapshot_id=f"snapshot-{step}",
        prefix_hash=f"prefix-{step}",
        signals=BranchPointSignals(**signals),
    )


def test_signals_validate_normalized_range():
    with pytest.raises(ValueError):
        BranchPointSignals(uncertainty=1.1)


def test_latest_checkpoint_policy_prefers_latest_step():
    candidates = [_candidate(1), _candidate(4), _candidate(2)]
    selected = LatestCheckpointPolicy().select(candidates)
    assert [item.step_id for item in selected] == [4]


def test_weighted_policy_exposes_score_components_and_selects_high_uncertainty():
    candidates = [
        _candidate(1, uncertainty=0.1, remaining_budget=0.9),
        _candidate(2, uncertainty=0.9, remaining_budget=0.8),
    ]
    policy = WeightedBranchPointPolicy(
        BranchSelectionWeights(
            uncertainty=2.0,
            novelty=0.0,
            commitment=0.0,
            value_gap=0.0,
            verifier_gap=0.0,
            remaining_budget=0.1,
            branch_cost=0.0,
        )
    )
    ranked = policy.rank(candidates)
    assert ranked[0].candidate.step_id == 2
    assert ranked[0].components["uncertainty"] == pytest.approx(1.8)


def test_weighted_policy_penalizes_expensive_branch_point():
    cheap = _candidate(1, uncertainty=0.8, estimated_branch_cost=0.1)
    expensive = _candidate(2, uncertainty=0.8, estimated_branch_cost=0.9)
    policy = WeightedBranchPointPolicy(
        BranchSelectionWeights(
            uncertainty=1.0,
            novelty=0.0,
            commitment=0.0,
            value_gap=0.0,
            verifier_gap=0.0,
            remaining_budget=0.0,
            branch_cost=1.0,
        )
    )
    assert policy.select([expensive, cheap])[0] == cheap


def test_bridge_health_conversion_uses_completed_calls_and_mutations():
    signals = signals_from_bridge_health(
        {
            "completed_tool_calls": 2,
            "max_tool_calls": 5,
            "tool_effect_counts": {"mutation": 1, "read_only": 1},
        },
        uncertainty=0.7,
        novelty=0.2,
    )
    assert signals.commitment == pytest.approx(0.5)
    assert signals.remaining_budget == pytest.approx(0.6)
    assert signals.uncertainty == 0.7
    assert signals.novelty == 0.2


def test_select_rejects_non_positive_count():
    with pytest.raises(ValueError):
        WeightedBranchPointPolicy().select([_candidate(1)], count=0)
