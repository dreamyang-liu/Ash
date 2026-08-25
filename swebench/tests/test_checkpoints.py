"""Per-step checkpoint policy: what gets captured, and when to re-board."""

from dataclasses import dataclass

from swebench.agent.checkpoints import (Checkpointer, MutationTracker,
                                        call_mutates, install)
from swebench.agent.pipeline import CallContext


@dataclass
class FakeSnapshot:
    id: str
    rootfs_layers: int | None = None
    memory_layers: int | None = None
    chain_size_mb: int | None = None


class FakeSession:
    """Stands in for AshSession: records calls, hands out scripted snapshots."""

    def __init__(self, layers=None, fail_after=None, can_snapshot=True):
        #: Layer counts to report, one per capture.
        self._layers = list(layers or [])
        self._fail_after = fail_after
        self._can_snapshot = can_snapshot
        self.captures: list[dict] = []
        self.swaps: list[str] = []
        self.swap_result = True

    def supports_snapshot(self):
        return self._can_snapshot

    def snapshot(self, name=None, disk_only=True):
        if self._fail_after is not None and len(self.captures) >= self._fail_after:
            return None
        index = len(self.captures)
        self.captures.append({"name": name, "disk_only": disk_only})
        layers = self._layers[index] if index < len(self._layers) else None
        return FakeSnapshot(id=f"snap-{index}", rootfs_layers=layers,
                            chain_size_mb=10 * (index + 1))

    def swap_sandbox(self, snapshot):
        self.swaps.append(getattr(snapshot, "id", snapshot))
        return self.swap_result


def ctx(tool_name, args=None):
    return CallContext(agent_id="a", sandbox_id="s", tool_name=tool_name,
                       args=args or {})


# --- classifier ------------------------------------------------------------ #

def test_read_only_calls_do_not_mark_mutation():
    for tool, args in [("grep_files", {}), ("web_search", {}),
                       ("web_fetch", {}), ("wait_for_events", {}),
                       ("text_editor", {"command": "view"}),
                       ("process", {"command": "read"})]:
        assert not call_mutates(tool, args), (tool, args)


def test_writes_and_unknown_tools_mark_mutation():
    for tool, args in [("text_editor", {"command": "str_replace"}),
                       ("text_editor", {"command": "write"}),
                       ("process", {"command": "kill"}),
                       ("artifact", {}), ("some_custom_tool", {})]:
        assert call_mutates(tool, args), (tool, args)


def test_shell_is_always_treated_as_mutating():
    # A command's effect cannot be read off its text, and a bash-only panel
    # would otherwise misjudge every step.
    assert call_mutates("shell", {"command": "ls"})
    assert call_mutates("shell", {"command": "cat file"})


def test_tracker_latches_until_cleared():
    tracker = MutationTracker()
    assert not tracker.dirty
    tracker.before(ctx("grep_files"))
    assert not tracker.dirty
    tracker.before(ctx("text_editor", {"command": "write"}))
    tracker.before(ctx("grep_files"))
    assert tracker.dirty, "a later read must not clear an earlier write"
    tracker.clear()
    assert not tracker.dirty


# --- checkpoint policy ----------------------------------------------------- #

def test_clean_steps_reuse_the_previous_snapshot():
    session = FakeSession()
    tracker = MutationTracker()
    cp = Checkpointer(session=session, tracker=tracker)

    tracker.before(ctx("shell", {"command": "echo hi > f"}))
    first = cp.after_step(1)
    second = cp.after_step(2)          # nothing ran in between
    third = cp.after_step(3)

    assert len(session.captures) == 1, "only the mutating step is captured"
    assert first.captured and not second.captured and not third.captured
    # The map is still complete: every step resolves to the state it had.
    assert cp.step_map() == {1: "snap-0", 2: "snap-0", 3: "snap-0"}


def test_every_step_is_captured_when_always_is_set():
    session = FakeSession()
    cp = Checkpointer(session=session, tracker=MutationTracker(), always=True)
    for turn in (1, 2, 3):
        cp.after_step(turn)
    assert len(session.captures) == 3
    assert cp.step_map() == {1: "snap-0", 2: "snap-1", 3: "snap-2"}


def test_disk_only_is_the_default_and_is_passed_through():
    session = FakeSession()
    Checkpointer(session=session, always=True).after_step(1)
    assert session.captures[0]["disk_only"] is True

    session = FakeSession()
    Checkpointer(session=session, always=True, disk_only=False).after_step(1)
    assert session.captures[0]["disk_only"] is False


def test_layer_count_drop_triggers_reboard():
    # Growing chain, then the server compacts: 2, 3, 4, then 2.
    session = FakeSession(layers=[2, 3, 4, 2])
    cp = Checkpointer(session=session, always=True)
    records = [cp.after_step(turn) for turn in (1, 2, 3, 4)]

    assert session.swaps == ["snap-3"], "re-board exactly on the drop"
    assert [r.reboarded for r in records] == [False, False, False, True]


def test_layer_count_that_fails_to_grow_also_triggers_reboard():
    """A deep inherited prefix whose per-cycle suffix merges into one layer
    compacts at a CONSTANT count (13 -> 13): no drop ever comes, and waiting
    for one leaves the sandbox re-compacting on every single step. Growth is
    the only uncompacted behaviour, so anything else is the signal."""
    session = FakeSession(layers=[13, 13])
    cp = Checkpointer(session=session, always=True)
    cp.after_step(1)
    record = cp.after_step(2)
    assert session.swaps == ["snap-1"]
    assert record.reboarded


def test_capture_duration_is_recorded():
    session = FakeSession()
    record = Checkpointer(session=session, always=True).after_step(1)
    assert record.capture_seconds >= 0.0
    tracker = MutationTracker()
    cp = Checkpointer(session=FakeSession(), tracker=tracker)
    cp.after_step(1)                     # captures (tracker starts clean? no --
    # tracker is clean, so this step reuses nothing (no previous) and returns
    # None; make one dirty capture then a clean reuse to check the zero.
    tracker.before(ctx("shell", {"command": "touch f"}))
    first = cp.after_step(2)
    reused = cp.after_step(3)
    assert first.capture_seconds >= 0.0
    assert reused.capture_seconds == 0.0


def test_first_capture_never_reboards():
    # A fresh chain starts small; with no previous count there is no drop.
    session = FakeSession(layers=[2, 3])
    cp = Checkpointer(session=session, always=True)
    cp.after_step(1)
    assert session.swaps == []
    cp.after_step(2)
    assert session.swaps == []


def test_reboard_does_not_retrigger_on_the_next_capture():
    # After re-boarding, the new sandbox's chain starts over (2, 3, ...); the
    # step following the drop must not look like another drop.
    session = FakeSession(layers=[4, 2, 3])
    cp = Checkpointer(session=session, always=True)
    cp.after_step(1)
    cp.after_step(2)                    # drop 4 -> 2: re-board
    cp.after_step(3)                    # 3 > (fresh baseline): no re-board
    assert session.swaps == ["snap-1"]


def test_reboard_can_be_disabled():
    session = FakeSession(layers=[4, 2])
    cp = Checkpointer(session=session, always=True, reboard=False)
    cp.after_step(1)
    record = cp.after_step(2)
    assert session.swaps == []
    assert not record.reboarded


def test_failed_reboard_is_recorded_as_not_reboarded():
    session = FakeSession(layers=[4, 2])
    session.swap_result = False
    cp = Checkpointer(session=session, always=True)
    cp.after_step(1)
    record = cp.after_step(2)
    assert session.swaps == ["snap-1"]
    assert not record.reboarded


def test_failed_capture_falls_back_to_the_last_snapshot():
    session = FakeSession(fail_after=1)
    cp = Checkpointer(session=session, always=True)
    cp.after_step(1)
    record = cp.after_step(2)
    assert record is not None and not record.captured
    assert cp.step_map() == {1: "snap-0", 2: "snap-0"}


def test_failed_first_capture_records_nothing():
    session = FakeSession(fail_after=0)
    cp = Checkpointer(session=session, always=True)
    assert cp.after_step(1) is None
    assert cp.records == []


def test_disabled_backend_is_a_no_op():
    session = FakeSession(can_snapshot=False)
    cp = Checkpointer(session=session, always=True)
    assert cp.after_step(1) is None
    assert session.captures == []
    assert not cp.enabled()


def test_checkpoints_are_reported_to_the_listener():
    seen = []
    session = FakeSession()
    cp = Checkpointer(session=session, always=True,
                      on_checkpoint=seen.append)
    cp.after_step(1)
    cp.after_step(2)
    assert [r.turn for r in seen] == [1, 2]
    assert seen[0].chain_size_mb == 10


def test_names_use_the_prefix_when_given():
    session = FakeSession()
    cp = Checkpointer(session=session, always=True, name_prefix="ep7-")
    cp.after_step(4)
    assert session.captures[0]["name"] == "ep7-step-4"


# --- wiring ---------------------------------------------------------------- #

class FakeAgent:
    def __init__(self):
        self.before_query_hooks = ["existing"]
        self.pipeline = None

        class Cost:
            api_calls = 0
        self.cost = Cost()


def test_install_preserves_an_existing_pipeline():
    from swebench.agent.pipeline import ToolPipeline, ToolInterceptor

    class Custom(ToolInterceptor):
        pass

    agent = FakeAgent()
    agent.pipeline = ToolPipeline([Custom()])
    install(agent, FakeSession(), always=True)

    names = [type(i).__name__ for i in agent.pipeline.interceptors]
    assert names == ["MutationTracker", "Custom"], (
        "the tracker mounts outermost without dropping configured policy")


def test_install_mounts_tracker_and_hook():
    agent = FakeAgent()
    session = FakeSession()
    cp = install(agent, session, always=True)

    assert agent.checkpointer is cp
    assert agent.before_query_hooks[0] == "existing", "existing hooks are kept"
    assert len(agent.before_query_hooks) == 2
    names = [type(i).__name__ for i in agent.pipeline.interceptors]
    assert names[0] == "MutationTracker", "tracker must sit outermost"

    # The hook labels the step whose tool calls just finished.
    agent.cost.api_calls = 0
    agent.before_query_hooks[1](agent, None)
    agent.cost.api_calls = 1
    agent.before_query_hooks[1](agent, None)
    assert [r.turn for r in cp.records] == [0, 1]


# --- re-board safety ------------------------------------------------------- #

def test_reboard_that_lands_on_an_unreachable_sandbox_is_not_adopted():
    """A disk-only snapshot cold-boots, so its runtime only comes back if the
    template declares a startup command. Adopting an unreachable replacement
    would break every later tool call; the session must keep the old one."""
    import asyncio

    class FakeSandbox:
        def __init__(self, reachable):
            self.agent_id = "agent"
            self._container_id = "sb"
            self._reachable = reachable
            self.calls = 0

        async def call(self, tool, **kwargs):
            self.calls += 1
            if self._reachable:
                class R:
                    is_error = False
                return R()
            raise RuntimeError("502 Bad Gateway")

    class FakePool:
        def __init__(self, replacement):
            self.replacement = replacement
            self.destroyed: list = []

        def supports_snapshot(self):
            return True

        async def spawn(self, image=None, agent_id=""):
            return self.replacement

        async def destroy(self, sandbox):
            self.destroyed.append(sandbox)

    from swebench.sandbox import AshSession

    for reachable, expect_swap in ((True, True), (False, False)):
        session = AshSession(quiet=True)
        original = FakeSandbox(reachable=True)
        replacement = FakeSandbox(reachable=reachable)
        session._sandbox = original
        session._pool = FakePool(replacement)

        assert session.swap_sandbox("snap-x") is expect_swap
        if expect_swap:
            assert session._sandbox is replacement
            assert session._pool.destroyed == [original]
        else:
            # Unreachable: the old sandbox is still serving calls, and the
            # useless replacement is cleaned up rather than leaked.
            assert session._sandbox is original
            assert session._pool.destroyed == [replacement]


def test_every_capture_compacting_warns_once(caplog):
    """A compaction budget below a single step's writes degrades silently
    into a compact+swap every other step; the guard turns that into one
    visible log line pointing at the config value."""
    import logging
    # Alternating pattern of the degraded mode: fresh chain (+1), then a
    # compaction that holds the count (13 -> 13 after re-board resets).
    session = FakeSession(layers=[13, 13, 13, 13, 13])
    session.swap_result = False          # keep the baseline comparable
    cp = Checkpointer(session=session, always=True)
    with caplog.at_level(logging.WARNING):
        for turn in range(1, 6):
            cp.after_step(turn)
    warnings = [r for r in caplog.records if "compaction budget" in r.message]
    assert len(warnings) == 1, "warn once, not per step"


def test_memory_chain_compaction_also_triggers_reboard():
    """Full snapshots carry a memory chain that compacts on its own schedule
    (memory intervals dwarf disk deltas). Watching only rootfs sat through a
    live episode where the memory chain compacted on 10 of 32 captures --
    each writing a near-VM-sized merged layer -- while rootfs grew +1 every
    time and the detector saw nothing."""
    session = FakeSession(layers=[13, 14, 15, 16])
    # rootfs grows normally; memory chain: grows, then compacts (15 -> 2).
    mem = [2, 15, 2, 3]
    orig = session.snapshot
    def snapshot(name=None, disk_only=True):
        snap = orig(name=name, disk_only=disk_only)
        snap.memory_layers = mem[len(session.captures) - 1]
        return snap
    session.snapshot = snapshot

    cp = Checkpointer(session=session, always=True, disk_only=False)
    records = [cp.after_step(t) for t in (1, 2, 3, 4)]
    assert [r.reboarded for r in records] == [False, False, True, False]
    assert session.swaps == ["snap-2"], "re-board on the memory-chain collapse"


def test_disk_only_zero_memory_layers_never_reads_as_compaction():
    # Disk-only snapshots report 0 memory layers every time; 0 <= 0 must not
    # trigger a re-board on every step.
    session = FakeSession(layers=[2, 3, 4])
    orig = session.snapshot
    def snapshot(name=None, disk_only=True):
        snap = orig(name=name, disk_only=disk_only)
        snap.memory_layers = 0
        return snap
    session.snapshot = snapshot
    cp = Checkpointer(session=session, always=True)
    for t in (1, 2, 3):
        cp.after_step(t)
    assert session.swaps == []


def test_background_processes_are_flagged_on_records():
    """Disk-only replays lose live processes; the records must say which
    steps carried that risk so replay tooling can warn instead of diverging
    silently. The flag latches on background starts and over-reports (a kill
    clears one start without knowing which process died)."""
    session = FakeSession()
    tracker = MutationTracker()
    cp = Checkpointer(session=session, tracker=tracker, always=True)

    tracker.before(ctx("shell", {"command": "ls"}))
    r1 = cp.after_step(1)
    tracker.before(ctx("shell", {"command": "npm run dev", "background": True}))
    r2 = cp.after_step(2)
    r3 = cp.after_step(3)                       # still running
    tracker.before(ctx("process", {"command": "kill", "pid": 42}))
    r4 = cp.after_step(4)

    assert [r.live_background for r in (r1, r2, r3, r4)] == [
        False, True, True, False]


def test_replay_caveats_surface_background_steps(tmp_path):
    import json as _json
    from swebench.replay import replay_caveats
    path = tmp_path / "t.json"
    path.write_text(_json.dumps({"info": {"checkpoints": {"records": [
        {"turn": 1, "live_background": False, "disk_only": True},
        {"turn": 2, "live_background": True, "disk_only": True},
    ]}}}))
    assert replay_caveats(path, 1) == []
    assert len(replay_caveats(path, 2)) == 1
    assert "background" in replay_caveats(path, 2)[0]


# --- lineage squash ---------------------------------------------------------- #

class SquashingSession(FakeSession):
    def __init__(self, squash_to="flat-1", **kw):
        super().__init__(**kw)
        self.squash_calls: list[str] = []
        self._squash_to = squash_to

    def squash_snapshot(self, snapshot, name=None):
        self.squash_calls.append(snapshot.id)
        if self._squash_to is None:
            return snapshot          # squash failed; session returns input
        return FakeSnapshot(id=self._squash_to, rootfs_layers=1)


def test_deep_lineage_reboard_squashes_first():
    """Each re-board adds one permanent prefix layer; at the stack cap the
    lineage dies. When the re-board target is already deep, squash it first
    and board the flattened twin -- the ratchet resets and trajectory length
    loses its ceiling."""
    session = SquashingSession(layers=[150, 150])
    cp = Checkpointer(session=session, always=True, squash_lineage_at=128)
    cp.after_step(1)
    record = cp.after_step(2)          # 150 -> 150: compacted, deep

    assert session.squash_calls == ["snap-1"]
    assert session.swaps == ["flat-1"], "board the squashed twin"
    assert record.lineage_squashed and record.reboarded
    # The step still maps to the original snapshot: the squashed twin is an
    # equivalent, but the canonical checkpoint is what was captured.
    assert record.snapshot_id == "snap-1"


def test_shallow_lineage_reboards_plainly():
    session = SquashingSession(layers=[5, 5])
    cp = Checkpointer(session=session, always=True, squash_lineage_at=128)
    cp.after_step(1)
    record = cp.after_step(2)
    assert session.squash_calls == []
    assert session.swaps == ["snap-1"]
    assert not record.lineage_squashed


def test_failed_squash_falls_back_to_plain_reboard():
    session = SquashingSession(layers=[150, 150], squash_to=None)
    cp = Checkpointer(session=session, always=True, squash_lineage_at=128)
    cp.after_step(1)
    record = cp.after_step(2)
    assert session.squash_calls == ["snap-1"], "squash was attempted"
    assert session.swaps == ["snap-1"], "boarded the original anyway"
    assert record.reboarded and not record.lineage_squashed


def test_lineage_squash_can_be_disabled():
    session = SquashingSession(layers=[150, 150])
    cp = Checkpointer(session=session, always=True, squash_lineage_at=0)
    cp.after_step(1)
    cp.after_step(2)
    assert session.squash_calls == []


# --- persistence ------------------------------------------------------------ #

def test_every_checkpoint_writes_the_trajectory(tmp_path):
    """Snapshots outlive an interrupted run; the map from step to snapshot has
    to as well, or the surviving snapshots are unusable. A real 5-hour run was
    killed with 300 snapshots on the server and nothing on disk saying which
    step each belonged to."""
    import json

    from swebench.models import Trajectory

    path = tmp_path / "traj.json"

    class Agent:
        def __init__(self):
            self.before_query_hooks = []
            self.pipeline = None
            self.trajectory = Trajectory(instance_id="task")
            self.trajectory.add_message("assistant", "did a thing")

            class Cost:
                api_calls = 3
                def to_dict(self):
                    return {}
            self.cost = Cost()

    agent = Agent()
    session = FakeSession()
    cp = install(agent, session, always=True, trajectory_path=path)

    cp.after_step(1)
    assert path.exists(), "the first checkpoint already wrote the trajectory"
    saved = json.loads(path.read_text())
    assert saved["info"]["checkpoints"]["step_snapshots"] == {"1": "snap-0"}
    assert saved["info"]["exit_status"] == "in_progress"

    cp.after_step(2)
    saved = json.loads(path.read_text())
    assert set(saved["info"]["checkpoints"]["step_snapshots"]) == {"1", "2"}


def test_a_failed_write_does_not_lose_the_step():
    """Resumability from this step is worth less than the step itself."""
    def explode(_checkpointer):
        raise OSError("disk full")

    session = FakeSession()
    cp = Checkpointer(session=session, always=True, persist=explode)
    record = cp.after_step(1)
    assert record is not None and record.captured


def test_no_path_means_no_writing(tmp_path):
    """Checkpoints still work for callers that manage their own records."""
    agent = FakeAgent()
    cp = install(agent, FakeSession(), always=True)
    assert cp.persist is None
    assert cp.after_step(1).captured


def test_a_failed_capture_is_distinguishable_from_a_clean_step(caplog):
    """Both leave the step pointing at an older snapshot, but one is free and
    expected while the other means checkpointing has stopped working. Recording
    only `captured` conflated them, and a real run whose every capture was
    failing (an alias collision) looked exactly like a very clean workload --
    which is how the investigation started by blaming the mutation gate."""
    import logging

    tracker = MutationTracker()
    session = FakeSession(fail_after=1)      # first capture works, rest fail
    cp = Checkpointer(session=session, tracker=tracker)

    tracker.before(ctx("shell", {"command": "touch f"}))
    first = cp.after_step(1)
    assert first.captured and first.reason == "captured"

    # A clean step: nothing ran, so the previous snapshot IS this step's state.
    clean = cp.after_step(2)
    assert not clean.captured and clean.reason == "clean"

    # A dirty step whose capture fails: same snapshot id, different meaning.
    tracker.before(ctx("shell", {"command": "touch g"}))
    with caplog.at_level(logging.WARNING):
        failed = cp.after_step(3)
    assert not failed.captured and failed.reason == "failed"
    assert any("capture has failed" in r.message for r in caplog.records)

    info = cp.as_info()
    assert info["captured"] == 1 and info["failed_captures"] == 1


def test_repeated_failures_do_not_flood_the_log(caplog):
    import logging
    session = FakeSession(fail_after=1)
    cp = Checkpointer(session=session, always=True)
    cp.after_step(0)
    with caplog.at_level(logging.WARNING):
        for turn in range(1, 40):
            cp.after_step(turn)
    warnings = [r for r in caplog.records if "capture has failed" in r.message]
    assert len(warnings) == 2, "first and tenth, not one per step"


def test_every_checkpoint_records_which_sandbox_the_run_is_on(tmp_path):
    """Re-boarding changes a run's sandbox id, so the id is only knowable from
    the run itself -- and only if it keeps saying. A cleanup that trusted the
    id it saw at launch deleted a live 1473-turn run's sandbox; every later
    tool call answered 404 and the loop read the resulting prose as
    "completed"."""
    import json

    class SessionWithEnvironment(FakeSession):
        def __init__(self):
            super().__init__()
            self.sandbox_id = "sandbox-first"

        def environment(self):
            return {"sandbox_id": self.sandbox_id, "base_image": "img"}

    from swebench.models import CostTracker, Trajectory

    class Agent:
        pipeline = None
        before_query_hooks = []
        cost = CostTracker()
        trajectory = Trajectory()

    path = tmp_path / "t.json"
    session = SessionWithEnvironment()
    agent = Agent()
    checkpointer = install(agent, session, always=True, trajectory_path=path)
    agent.before_query_hooks[-1](agent, None)
    assert json.loads(path.read_text())["info"]["environment"]["sandbox_id"] \
        == "sandbox-first"

    # After a re-board the file must name the new sandbox, not the old one.
    session.sandbox_id = "sandbox-after-reboard"
    agent.cost.api_calls = 1
    agent.before_query_hooks[-1](agent, None)
    assert json.loads(path.read_text())["info"]["environment"]["sandbox_id"] \
        == "sandbox-after-reboard"
