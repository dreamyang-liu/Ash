"""Addressing a checkpointed step, and preparing a branch point."""

import json

from swebench.replay import (BranchPointCache, load_step_snapshots,
                             messages_through_step, snapshot_for_step)


def write_trajectory(tmp_path, step_snapshots, messages=None):
    path = tmp_path / "traj.json"
    path.write_text(json.dumps({
        "instance_id": "x",
        "messages": messages or [],
        "info": {"checkpoints": {"step_snapshots": step_snapshots,
                                 "disk_only": True}},
    }))
    return path


class FakeSession:
    def __init__(self, squash_to="flat", layers=None):
        self._squash_to = squash_to
        self.squash_calls: list[str] = []
        self._layers = layers

    def squash_snapshot(self, snapshot_id, name=None):
        self.squash_calls.append(snapshot_id)
        return self._squash_to


# --- reading the map ------------------------------------------------------- #

def test_step_snapshots_load_with_integer_steps(tmp_path):
    path = write_trajectory(tmp_path, {"1": "a", "2": "a", "3": "b"})
    assert load_step_snapshots(path) == {1: "a", 2: "a", 3: "b"}


def test_missing_checkpoints_load_as_empty(tmp_path):
    path = tmp_path / "traj.json"
    path.write_text(json.dumps({"messages": [], "info": {}}))
    assert load_step_snapshots(path) == {}


def test_exact_step_wins():
    assert snapshot_for_step({1: "a", 2: "b"}, 2) == "b"


def test_unrecorded_step_falls_back_to_the_nearest_earlier_one():
    # A map holding only captures (steps 1 and 4) still addresses step 3: the
    # environment did not change between 1 and 4.
    assert snapshot_for_step({1: "a", 4: "d"}, 3) == "a"


def test_step_before_any_snapshot_has_none():
    assert snapshot_for_step({2: "b"}, 1) is None
    assert snapshot_for_step({}, 5) is None


# --- transcript prefix ----------------------------------------------------- #

def test_prefix_cuts_after_the_step_th_assistant_turn(tmp_path):
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "turn 1"},
        {"role": "tool_result", "content": "r1"},
        {"role": "assistant", "content": "turn 2"},
        {"role": "tool_result", "content": "r2"},
        {"role": "assistant", "content": "turn 3"},
    ]
    path = write_trajectory(tmp_path, {}, messages)

    through_1 = messages_through_step(path, 1)
    assert [m["content"] for m in through_1] == ["s", "task", "turn 1", "r1"]

    through_2 = messages_through_step(path, 2)
    assert through_2[-1]["content"] == "r2"

    # Asking for more steps than exist yields the whole transcript.
    assert len(messages_through_step(path, 99)) == len(messages)


def test_prefix_of_step_zero_is_the_setup_only(tmp_path):
    messages = [{"role": "system", "content": "s"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "turn 1"}]
    path = write_trajectory(tmp_path, {}, messages)
    assert [m["role"] for m in messages_through_step(path, 0)] == ["system", "user"]


# --- branch point preparation --------------------------------------------- #

def test_single_shallow_branch_is_used_as_is():
    session = FakeSession()
    cache = BranchPointCache(session=session)
    assert cache.prepare("snap", fan_out=1, layers=10) == "snap"
    assert session.squash_calls == []


def test_fan_out_squashes_even_a_shallow_chain():
    # Several children share one merge, so it pays immediately.
    session = FakeSession(squash_to="flat")
    cache = BranchPointCache(session=session)
    assert cache.prepare("snap", fan_out=4, layers=10) == "flat"
    assert session.squash_calls == ["snap"]


def test_deep_chain_squashes_for_a_single_child():
    session = FakeSession(squash_to="flat")
    cache = BranchPointCache(session=session, layer_threshold=128)
    assert cache.prepare("snap", fan_out=1, layers=200) == "flat"
    assert session.squash_calls == ["snap"]


def test_repeated_branch_points_squash_once():
    session = FakeSession(squash_to="flat")
    cache = BranchPointCache(session=session)
    first = cache.prepare("snap", fan_out=2, layers=10)
    second = cache.prepare("snap", fan_out=2, layers=10)
    third = cache.prepare("snap", fan_out=8, layers=10)
    assert first == second == third == "flat"
    assert session.squash_calls == ["snap"], "the merge is paid for once"


def test_failed_squash_falls_back_to_the_source():
    class Failing(FakeSession):
        def squash_snapshot(self, snapshot_id, name=None):
            self.squash_calls.append(snapshot_id)
            return snapshot_id          # session returns input on failure

    session = Failing()
    cache = BranchPointCache(session=session)
    assert cache.prepare("snap", fan_out=4, layers=10) == "snap"


def test_unknown_layer_count_does_not_squash_a_single_branch():
    # Without chain facts and with one child, there is nothing to justify a
    # merge; the branch just inherits what it inherits.
    session = FakeSession()
    cache = BranchPointCache(session=session)
    assert cache.prepare("snap", fan_out=1, layers=None) == "snap"
    assert session.squash_calls == []


# --- the map survives the trajectory round trip ---------------------------- #

def test_saved_trajectory_keeps_the_checkpoint_map(tmp_path):
    """The map is only useful if it reaches the file replay reads.

    `Trajectory.save` derives model_stats/exit_status/submission; everything
    else a harness attached has to survive alongside them.
    """
    from swebench.models import Trajectory

    traj = Trajectory(instance_id="inst")
    traj.add_message("assistant", "turn 1")
    traj.info = {
        "exit_status": "completed",
        "submission": "diff --git a b",
        "checkpoints": {"step_snapshots": {1: "snap-a", 2: "snap-a"},
                        "disk_only": True},
    }
    path = tmp_path / "traj.json"
    traj.save(path)

    assert load_step_snapshots(path) == {1: "snap-a", 2: "snap-a"}
    assert snapshot_for_step(load_step_snapshots(path), 2) == "snap-a"

    import json
    saved = json.loads(path.read_text())
    # The derived fields still win, so nothing an harness sets can shadow them.
    assert saved["info"]["exit_status"] == "completed"
    assert saved["info"]["submission"] == "diff --git a b"
    assert "model_stats" in saved["info"]
