"""SnapshotBridge: pairing environment snapshots with conversation refs.

Uses a fake Checkpointer/session so the pairing logic is tested without a real
sandbox. The real Checkpointer is exercised by swebench's own tests; what is new
here is the *pairing* and the quiesce rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from harness.checkpointing import SnapshotBridge
from harness.core import events as E
from harness.core.journal import JournalWriter, read_journal
from harness.rollback import fork_plan, load_checkpoints


@dataclass
class FakeRecord:
    turn: int
    snapshot_id: Optional[str]
    captured: bool = True
    reason: str = "captured"
    disk_only: bool = True


@dataclass
class FakeCheckpointer:
    """Stands in for swebench.agent.checkpoints.Checkpointer."""

    on_checkpoint: object = None
    is_enabled: bool = True
    calls: List[int] = field(default_factory=list)
    #: Steps that should report "clean" (reuse previous snapshot).
    clean_steps: tuple = ()
    latest: Optional[str] = None

    def enabled(self) -> bool:
        return self.is_enabled

    def after_step(self, turn: int):
        self.calls.append(turn)
        if turn in self.clean_steps:
            record = FakeRecord(turn=turn, snapshot_id=self.latest, reason="clean")
        else:
            self.latest = "snap-%d" % turn
            record = FakeRecord(turn=turn, snapshot_id=self.latest)
        if self.on_checkpoint:
            self.on_checkpoint(record)
        return record


def make(tmp_path, **kwargs):
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r1")
    checkpointer = FakeCheckpointer(**kwargs)
    bridge = SnapshotBridge.install(journal, session=object(), checkpointer=checkpointer)
    return journal, bridge, checkpointer


def test_turn_completed_triggers_a_paired_checkpoint(tmp_path):
    journal, bridge, checkpointer = make(tmp_path)
    journal.emit(E.SESSION_REF, native_session_id="ses_1")
    journal.emit(E.TURN_COMPLETED, usage={})

    assert checkpointer.calls == [1]
    checkpoint = bridge.ledger.checkpoints[-1]
    assert checkpoint.snapshot_id == "snap-1"
    assert checkpoint.session_ckpt == "ses_1"
    assert checkpoint.is_complete()
    journal.close()


def test_no_checkpoint_while_a_tool_call_is_in_flight(tmp_path):
    """Snapshotting mid-call pairs an unresolved call with an ambiguous fs."""
    journal, bridge, checkpointer = make(tmp_path)
    journal.emit(E.SESSION_REF, native_session_id="ses_1")
    journal.emit(E.TOOL_STARTED, call_id="c1", name="shell", args={})
    journal.emit(E.TURN_COMPLETED, usage={})       # not quiesced -> skipped
    assert checkpointer.calls == []

    journal.emit(E.TOOL_FINISHED, call_id="c1", status="ok", output="")
    journal.emit(E.TURN_COMPLETED, usage={})       # quiesced now
    assert checkpointer.calls == [1]
    journal.close()


def test_nested_tool_calls_track_depth(tmp_path):
    journal, bridge, checkpointer = make(tmp_path)
    for call_id in ("c1", "c2"):
        journal.emit(E.TOOL_STARTED, call_id=call_id, name="shell", args={})
    journal.emit(E.TOOL_FINISHED, call_id="c1", status="ok", output="")
    journal.emit(E.TURN_COMPLETED, usage={})       # one still open
    assert checkpointer.calls == []
    journal.emit(E.TOOL_FINISHED, call_id="c2", status="ok", output="")
    journal.emit(E.TURN_COMPLETED, usage={})
    assert checkpointer.calls == [1]
    journal.close()


def test_tool_boundary_callback_is_the_sdk_slot_trigger(tmp_path):
    """ClaudeCodeSlot(on_tool_boundary=bridge.on_tool_boundary)."""
    journal, bridge, checkpointer = make(tmp_path)
    journal.emit(E.SESSION_REF, native_session_id="ses_9")
    bridge.on_tool_boundary(1)
    bridge.on_tool_boundary(2)
    assert checkpointer.calls == [1, 2]
    assert [c.step for c in bridge.ledger.checkpoints] == [1, 2]
    assert all(c.session_ckpt == "ses_9" for c in bridge.ledger.checkpoints)
    journal.close()


def test_late_session_ref_is_backfilled(tmp_path):
    """A checkpoint can land before the agent discloses its session id."""
    journal, bridge, checkpointer = make(tmp_path)
    journal.emit(E.TURN_COMPLETED, usage={})       # no session ref yet
    first = bridge.ledger.checkpoints[0]
    assert first.snapshot_id == "snap-1" and first.session_ckpt is None

    journal.emit(E.SESSION_REF, native_session_id="ses_late")
    assert bridge.ledger.checkpoints[0].session_ckpt == "ses_late"

    # correction is an append, not a rewrite
    records = read_journal(tmp_path / "j.jsonl")
    reasons = [r["reason"] for r in records if r["type"] == E.CHECKPOINT_CAPTURED]
    assert reasons == ["captured", "session_ref_backfill"]
    journal.close()


def test_clean_steps_reuse_the_previous_snapshot(tmp_path):
    """Read-only steps map to the last capture, so the step map stays complete."""
    journal, bridge, checkpointer = make(tmp_path, clean_steps=(2,))
    journal.emit(E.SESSION_REF, native_session_id="ses_1")
    journal.emit(E.TURN_COMPLETED, usage={})
    journal.emit(E.TURN_COMPLETED, usage={})

    steps = {c.step: c.snapshot_id for c in bridge.ledger.checkpoints}
    assert steps == {1: "snap-1", 2: "snap-1"}
    assert bridge.ledger.at_step(2).snapshot_id == "snap-1"
    journal.close()


def test_disabled_checkpointer_is_a_no_op(tmp_path):
    journal, bridge, checkpointer = make(tmp_path, is_enabled=False)
    journal.emit(E.TURN_COMPLETED, usage={})
    assert checkpointer.calls == []
    assert bridge.ledger.checkpoints == []
    journal.close()


def test_existing_on_checkpoint_callback_is_preserved(tmp_path):
    """A harness may already report checkpoints somewhere; do not displace it."""
    seen = []
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r1")
    checkpointer = FakeCheckpointer(on_checkpoint=seen.append)
    bridge = SnapshotBridge.install(journal, session=object(), checkpointer=checkpointer)
    journal.emit(E.TURN_COMPLETED, usage={})
    assert len(seen) == 1                       # original still called
    assert len(bridge.ledger.checkpoints) == 1  # and ours ran
    journal.close()


def test_end_to_end_fork_plan_resolves_both_halves(tmp_path):
    journal, bridge, checkpointer = make(tmp_path)
    journal.emit(E.RUN_STARTED, slot="opencode", task_prompt="t")
    journal.emit(E.SESSION_REF, native_session_id="ses_1")
    for _ in range(3):
        journal.emit(E.TOOL_STARTED, call_id="c", name="shell", args={})
        journal.emit(E.TOOL_FINISHED, call_id="c", status="ok", output="")
        journal.emit(E.TURN_COMPLETED, usage={"input_tokens": 5})
    journal.close()

    path = tmp_path / "j.jsonl"
    assert [c.step for c in load_checkpoints(path)] == [1, 2, 3]

    plan = fork_plan(path, 2)
    assert plan["snapshot_id"] == "snap-2"
    assert plan["session_ckpt"] == "ses_1"
    assert plan["complete"] is True
    assert plan["copied_through_seq"] > 0


def test_observer_failure_cannot_break_the_run(tmp_path):
    """Journal subscribers are isolated: a broken bridge must not kill a run."""
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r1")

    def explode(_record):
        raise RuntimeError("bridge is broken")

    journal.subscribe(explode)
    record = journal.emit(E.AGENT_MESSAGE, text="still recorded")
    assert record["seq"] == 1
    assert read_journal(tmp_path / "j.jsonl")[0]["text"] == "still recorded"
    journal.close()
