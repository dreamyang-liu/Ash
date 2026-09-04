from __future__ import annotations

from swebench.checkpoint_policy import OverheadBudgetCheckpointPolicy


def test_policy_expands_stride_when_snapshot_is_expensive_relative_to_steps():
    policy = OverheadBudgetCheckpointPolicy(
        target_snapshot_fraction=0.05,
        initial_stride=10,
        min_stride=5,
        max_stride=100,
    )
    assert policy.should_checkpoint(9) is False
    assert policy.should_checkpoint(10) is True

    stride = policy.observe_checkpoint(
        step_id=10,
        rollout_elapsed_ms=120_000.0,
        snapshot_ms=22_800.0,
    )
    assert 35 <= stride <= 37
    assert policy.next_checkpoint_step == 10 + stride


def test_policy_keeps_small_stride_for_slow_long_horizon_rollout():
    policy = OverheadBudgetCheckpointPolicy(
        target_snapshot_fraction=0.05,
        initial_stride=10,
        min_stride=5,
        max_stride=100,
    )
    stride = policy.observe_checkpoint(
        step_id=10,
        rollout_elapsed_ms=300_000.0,
        snapshot_ms=12_000.0,
    )
    assert stride <= 8
    assert stride >= 5


def test_policy_subtracts_previous_snapshot_time_from_step_cost_estimate():
    policy = OverheadBudgetCheckpointPolicy(
        target_snapshot_fraction=0.05,
        initial_stride=10,
        min_stride=5,
        max_stride=100,
    )
    policy.observe_checkpoint(step_id=10, rollout_elapsed_ms=100_000.0, snapshot_ms=10_000.0)
    stride = policy.observe_checkpoint(step_id=20, rollout_elapsed_ms=210_000.0, snapshot_ms=10_000.0)
    # 210s before the second snapshot contains the first 10s snapshot pause;
    # estimated non-snapshot work is ~200s over 20 steps = 10s/step.
    assert 19 <= stride <= 20


def test_policy_clamps_stride_and_validates_budget():
    policy = OverheadBudgetCheckpointPolicy(
        target_snapshot_fraction=0.01,
        initial_stride=10,
        min_stride=5,
        max_stride=30,
    )
    assert policy.observe_checkpoint(
        step_id=10,
        rollout_elapsed_ms=20_000.0,
        snapshot_ms=30_000.0,
    ) == 30

    try:
        OverheadBudgetCheckpointPolicy(target_snapshot_fraction=1.0)
    except ValueError as exc:
        assert "target_snapshot_fraction" in str(exc)
    else:
        raise AssertionError("expected invalid fraction to be rejected")
