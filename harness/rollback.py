"""Rollback pairing: journal seq <-> (env snapshot, native session ref).

The differentiating capability of this stack. An agent's state has two halves
and neither alone is enough to branch a run:

- **environment**: files, installed packages, background processes. Captured by
  AgentENV snapshots (swebench/agent/checkpoints.py drives the *when*).
- **conversation**: the agent's context. Owned by the agent -- a native session
  id for external slots (Claude Code / codex / opencode), or the transcript in
  the journal for in-house agents.

This module records the *pair* in the journal so any step can be reconstructed:
``checkpoint.captured {step, seq, snapshot_id, session_ckpt}``. Without the
pairing, restoring an env snapshot gives an agent whose memory disagrees with
the filesystem.

Quiesce: pair only at a step boundary with no in-flight tool call. For the SDK
slot that is the ``can_use_tool`` callback; for CLI slots it is between
``tool.finished`` and the next ``tool.started``. Snapshotting mid-call leaves an
unresolved call in the conversation and an ambiguous environment.

Fork support differs per slot (:class:`SlotCapabilities`):
- opencode: native ``--session <id> --fork``
- claude-code: SDK ``resume`` + ``fork_session``
- codex: resume only -- branch by replaying prompt against the env snapshot
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from harness.core.events import CHECKPOINT_CAPTURED
from harness.core.journal import JournalWriter, read_journal
from harness.normalize.claude_turns import completed_turn_steps


@dataclass
class Checkpoint:
    step: int
    seq: int
    snapshot_id: Optional[str]
    session_ckpt: Optional[str] = None
    reason: str = "captured"
    delta_empty: Optional[bool] = None
    call_id: Optional[str] = None
    pairing: Optional[str] = None
    prefix_complete: Optional[bool] = None

    def is_complete(self) -> bool:
        """Both halves present -> a full rollback point."""
        return bool(self.snapshot_id) and bool(self.session_ckpt)


@dataclass
class RollbackLedger:
    """Records checkpoint pairs into the journal and answers lookups."""

    journal: JournalWriter
    checkpoints: List[Checkpoint] = field(default_factory=list)

    def record(
        self,
        step: int,
        snapshot_id: Optional[str],
        *,
        session_ckpt: Optional[str] = None,
        reason: str = "captured",
        **extra,
    ) -> Checkpoint:
        record = self.journal.emit(
            CHECKPOINT_CAPTURED,
            step=step,
            snapshot_id=snapshot_id,
            session_ckpt=session_ckpt,
            reason=reason,
            **extra,
        )
        checkpoint = Checkpoint(
            step=step,
            seq=record["seq"],
            snapshot_id=snapshot_id,
            session_ckpt=session_ckpt,
            reason=reason,
            delta_empty=extra.get("delta_empty"),
            call_id=extra.get("call_id"),
            pairing=extra.get("pairing"),
            prefix_complete=extra.get("prefix_complete"),
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def step_map(self) -> Dict[int, Optional[str]]:
        return {c.step: c.snapshot_id for c in self.checkpoints}

    def at_step(self, step: int) -> Optional[Checkpoint]:
        """Latest checkpoint at or before ``step`` (clean steps reuse snapshots)."""
        best = None
        for checkpoint in self.checkpoints:
            if checkpoint.step <= step and checkpoint.snapshot_id:
                if best is None or checkpoint.step > best.step:
                    best = checkpoint
        return best


def load_checkpoints(journal_path, *, events=None) -> List[Checkpoint]:
    """Read checkpoint pairs back out of a journal (for resume/fork tooling)."""
    out: List[Checkpoint] = []
    for record in read_journal(journal_path) if events is None else events:
        if record.get("type") != CHECKPOINT_CAPTURED:
            continue
        out.append(
            Checkpoint(
                step=record.get("step") or 0,
                seq=record.get("seq") or 0,
                snapshot_id=record.get("snapshot_id"),
                session_ckpt=record.get("session_ckpt"),
                reason=record.get("reason") or "captured",
                delta_empty=record.get("delta_empty"),
                call_id=record.get("call_id"),
                pairing=record.get("pairing"),
                prefix_complete=record.get("prefix_complete"),
            )
        )
    return out


def branch_checkpoints(journal_path, *, events=None) -> Dict[int, Checkpoint]:
    """Exact successful/clean ledger pairs, not a server or native-cut probe."""
    latest: Dict[int, Checkpoint] = {}
    events = list(read_journal(journal_path) if events is None else events)
    exact = any(r.get("type") == "checkpoint.policy" and
                r.get("pairing") == "call-id-v1" for r in events)
    calls = [r.get("call_id") for r in events
             if r.get("type") == "tool.started" and r.get("call_id")]
    for checkpoint in load_checkpoints(journal_path, events=events):
        if type(checkpoint.step) is not int or checkpoint.step < 1:
            continue
        if checkpoint.reason == "session_ref_backfill":
            previous = latest.get(checkpoint.step)
            if previous and previous.snapshot_id == checkpoint.snapshot_id:
                latest[checkpoint.step] = replace(
                    previous, session_ckpt=checkpoint.session_ckpt or previous.session_ckpt)
        else:
            latest[checkpoint.step] = checkpoint

    eligible = {}
    current_snapshot = None
    for step, checkpoint in sorted(latest.items()):
        if exact or checkpoint.pairing is not None:
            if (checkpoint.pairing != "call-id-v1" or checkpoint.prefix_complete is not True
                    or step > len(calls) or checkpoint.call_id != calls[step - 1]):
                current_snapshot = None
                continue
        if checkpoint.reason == "captured":
            current_snapshot = checkpoint.snapshot_id
        elif checkpoint.reason != "clean" or checkpoint.snapshot_id != current_snapshot:
            current_snapshot = None
            continue
        if checkpoint.is_complete():
            eligible[step] = checkpoint
    return eligible


def turn_branch_checkpoints(journal_path) -> Dict[int, Checkpoint]:
    """Preserve the storage ledger; expose only completed turns for new runs.

    Native cut availability must still be checked by the slot-specific caller.
    Legacy journals retain their prior ledger interpretation.
    """
    events = list(read_journal(journal_path))
    points = branch_checkpoints(journal_path, events=events)
    turns = completed_turn_steps(events)
    return points if turns is None else {s: p for s, p in points.items() if s in turns}


def fork_plan(journal_path, step: int) -> dict:
    """Describe how to branch a recorded run at ``step``.

    Returns the pair plus the seq boundary an ATIF export needs for
    ``is_copied_context``.
    """
    checkpoints = load_checkpoints(journal_path)
    exact = any(r.get("type") == "checkpoint.policy" and r.get("pairing") == "call-id-v1"
                for r in read_journal(journal_path))
    if exact:
        candidate = turn_branch_checkpoints(journal_path).get(step)
        if candidate is None:
            raise ValueError("no exact restorable checkpoint at step %s" % step)
        return {"step": step, "snapshot_id": candidate.snapshot_id,
                "session_ckpt": candidate.session_ckpt, "copied_through_seq": candidate.seq,
                "call_id": candidate.call_id, "complete": True}
    candidate = None
    for checkpoint in checkpoints:
        if checkpoint.step <= step and checkpoint.snapshot_id:
            if candidate is None or checkpoint.step > candidate.step:
                candidate = checkpoint
    if candidate is None:
        raise ValueError("no snapshot at or before step %s in %s" % (step, journal_path))
    return {
        "step": candidate.step,
        "snapshot_id": candidate.snapshot_id,
        "session_ckpt": candidate.session_ckpt,
        "copied_through_seq": candidate.seq,
        "complete": candidate.is_complete(),
    }
