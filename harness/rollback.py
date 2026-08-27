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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from harness.core.events import CHECKPOINT_CAPTURED
from harness.core.journal import JournalWriter, read_journal


@dataclass
class Checkpoint:
    step: int
    seq: int
    snapshot_id: Optional[str]
    session_ckpt: Optional[str] = None
    reason: str = "captured"

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


def load_checkpoints(journal_path) -> List[Checkpoint]:
    """Read checkpoint pairs back out of a journal (for resume/fork tooling)."""
    out: List[Checkpoint] = []
    for record in read_journal(journal_path):
        if record.get("type") != CHECKPOINT_CAPTURED:
            continue
        out.append(
            Checkpoint(
                step=record.get("step") or 0,
                seq=record.get("seq") or 0,
                snapshot_id=record.get("snapshot_id"),
                session_ckpt=record.get("session_ckpt"),
                reason=record.get("reason") or "captured",
            )
        )
    return out


def fork_plan(journal_path, step: int) -> dict:
    """Describe how to branch a recorded run at ``step``.

    Returns the pair plus the seq boundary an ATIF export needs for
    ``is_copied_context``.
    """
    checkpoints = load_checkpoints(journal_path)
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
