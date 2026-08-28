"""Bridge: ``harness.execution.checkpoints.Checkpointer`` -> ``RollbackLedger``.

The environment half of rollback already exists and is battle-tested (mutation
tracking, clean-step snapshot reuse, layer-chain compaction re-boarding, lineage
squashing). What it lacked for external agents is the *other* half: which
conversation state each snapshot belongs to. This module supplies it.

It hooks in without any slot knowing:

- subscribes to the journal, so ``session.ref`` events keep the conversation
  reference current (a native session id for claude-code/codex/opencode);
- fires a checkpoint at each **quiesce point** -- ``turn.completed`` for CLI
  slots, or the PreToolUse boundary callback for the SDK slot;
- records the resulting pair via :class:`~harness.rollback.RollbackLedger`, so
  ``fork-plan`` can resolve both halves at any step.

Why turn boundaries and not "every N seconds": snapshotting mid tool call leaves
an unresolved call in the conversation and an ambiguous filesystem, so the pair
would not describe a state the agent could resume from.

Usage::

    bridge = SnapshotBridge.install(journal, session)          # CLI slots
    slot.run(task, journal, mcp)

    bridge = SnapshotBridge.install(journal, session)          # SDK slot
    slot = ClaudeCodeSlot(on_tool_boundary=bridge.on_tool_boundary)

``session`` only needs the ``supports_snapshot`` / ``snapshot`` /
``swap_sandbox`` / ``squash_snapshot`` surface (an ``AshSession``, or a small
test double).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from harness.core.events import SESSION_REF, TOOL_FINISHED, TOOL_STARTED, TURN_COMPLETED
from harness.core.journal import JournalWriter
from harness.rollback import Checkpoint, RollbackLedger


@dataclass
class SnapshotBridge:
    """Pairs environment snapshots with conversation references."""

    journal: JournalWriter
    session: Any
    ledger: RollbackLedger
    checkpointer: Any = None
    #: Snapshot every step rather than only mutating ones.
    always: bool = False
    session_ref: Optional[str] = None
    step: int = 0
    #: Depth of unfinished tool calls; a checkpoint is only safe at zero.
    _inflight: int = 0
    _pending: bool = False
    #: Captures declined because the caller was on an event loop thread.
    _skipped_on_loop: int = 0
    records: list = field(default_factory=list)

    # --- construction ------------------------------------------------------
    @classmethod
    def install(
        cls,
        journal: JournalWriter,
        session: Any,
        *,
        always: bool = False,
        disk_only: bool = True,
        name_prefix: str = "",
        tracker: Any = None,
        checkpointer: Any = None,
    ) -> "SnapshotBridge":
        """Wire a bridge onto ``journal``.

        ``checkpointer`` may be supplied (e.g. one a harness already mounted on
        its pipeline together with the ``MutationTracker``); otherwise one is
        built here. Passing an existing one is the normal path when the tool
        pipeline is owned elsewhere -- the tracker must sit on that pipeline for
        clean-step reuse to work.
        """
        bridge = cls(
            journal=journal,
            session=session,
            ledger=RollbackLedger(journal),
            always=always,
        )
        if checkpointer is None:
            checkpointer = _build_checkpointer(
                session,
                tracker=tracker,
                always=always,
                disk_only=disk_only,
                name_prefix=name_prefix,
                on_checkpoint=bridge._on_checkpoint,
            )
        else:
            _chain_on_checkpoint(checkpointer, bridge._on_checkpoint)
        bridge.checkpointer = checkpointer
        journal.subscribe(bridge._on_event)
        return bridge

    # --- journal subscription ---------------------------------------------
    def _on_event(self, record: dict) -> None:
        etype = record.get("type")

        if etype == SESSION_REF:
            ref = record.get("native_session_id")
            if ref:
                self.session_ref = ref
                if self._pending:
                    # A checkpoint landed before the agent disclosed its session
                    # id; complete the pair now rather than leaving a half.
                    self._pending = False
                    self._backfill_session_ref(ref)
            return

        if etype == TOOL_STARTED:
            self._inflight += 1
            return

        if etype == TOOL_FINISHED:
            self._inflight = max(0, self._inflight - 1)
            return

        if etype == TURN_COMPLETED:
            self.maybe_checkpoint()

    # --- checkpoint triggers ----------------------------------------------
    def on_tool_boundary(self, index: int) -> Optional[Checkpoint]:
        """Step boundary for an external agent: its tool call just executed.

        ``force`` is required here. The caller is asserting the executor has
        returned, but the journal's ``tool.finished`` event is emitted later (the
        SDK surfaces the ToolResultBlock on the next message), so the in-flight
        guard would still see depth 1 and skip every checkpoint -- the bug that
        made a rollback-capable run record zero snapshots.
        """
        return self.maybe_checkpoint(step=index, force=True)

    def maybe_checkpoint(
        self, step: Optional[int] = None, *, force: bool = False
    ) -> Optional[Checkpoint]:
        """Take a checkpoint if we are quiesced and the session supports it."""
        if self._inflight and not force:
            return None
        if self.checkpointer is None or not _enabled(self.checkpointer):
            return None
        if _on_running_loop():
            # AshSession drives a private loop with run_until_complete, which
            # cannot be entered from a thread that already has a running loop --
            # it fails and leaves the coroutine un-awaited. The SDK slot journals
            # from inside its event loop, and for that slot the tool boundary
            # (a worker thread) is the correct trigger anyway.
            self._skipped_on_loop += 1
            return None
        self.step = step if step is not None else self.step + 1
        record = self.checkpointer.after_step(self.step)
        if record is None:
            return None
        return self.records[-1] if self.records else None

    @property
    def skipped_on_loop(self) -> int:
        """Checkpoint opportunities dropped because a loop was already running.

        Read this. It was private and read by nobody, which is how a run could
        report checkpointing as enabled and record zero snapshots: an SDK slot
        journals its turn from inside its event loop, every opportunity was skipped
        here, and the count sat in a field no caller looked at. Whoever asked for
        checkpoints is entitled to know they did not happen.
        """
        return self._skipped_on_loop

    # --- Checkpointer callback --------------------------------------------
    def _on_checkpoint(self, record: Any) -> None:
        snapshot_id = getattr(record, "snapshot_id", None)
        reason = getattr(record, "reason", "captured")
        checkpoint = self.ledger.record(
            getattr(record, "turn", self.step),
            snapshot_id,
            session_ckpt=self.session_ref,
            reason=reason,
            captured=bool(getattr(record, "captured", False)),
            disk_only=bool(getattr(record, "disk_only", True)),
        )
        self.records.append(checkpoint)
        if snapshot_id and not self.session_ref:
            self._pending = True

    def _backfill_session_ref(self, ref: str) -> None:
        """Attach a late-arriving session id to the pairs already recorded.

        Appends a correction event rather than rewriting history (the journal is
        append-only); ``load_checkpoints`` sees the corrected pair because later
        records for the same step win.
        """
        for checkpoint in list(self.ledger.checkpoints):
            if checkpoint.snapshot_id and not checkpoint.session_ckpt:
                checkpoint.session_ckpt = ref
                self.ledger.record(
                    checkpoint.step,
                    checkpoint.snapshot_id,
                    session_ckpt=ref,
                    reason="session_ref_backfill",
                )


# --- helpers ---------------------------------------------------------------
def _on_running_loop() -> bool:
    """True when this thread already runs an asyncio loop."""
    import asyncio

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _enabled(checkpointer: Any) -> bool:
    enabled = getattr(checkpointer, "enabled", None)
    if callable(enabled):
        try:
            return bool(enabled())
        except Exception:  # noqa: BLE001
            return False
    return True


def _build_checkpointer(
    session: Any,
    *,
    tracker: Any,
    always: bool,
    disk_only: bool,
    name_prefix: str,
    on_checkpoint,
):
    from harness.execution.checkpoints import Checkpointer, MutationTracker

    return Checkpointer(
        session=session,
        tracker=tracker if tracker is not None else MutationTracker(),
        always=always,
        disk_only=disk_only,
        name_prefix=name_prefix,
        on_checkpoint=on_checkpoint,
    )


def _chain_on_checkpoint(checkpointer: Any, callback) -> None:
    """Add our callback without displacing one the caller already set."""
    existing = getattr(checkpointer, "on_checkpoint", None)
    if existing is None:
        checkpointer.on_checkpoint = callback
        return

    def chained(record):
        try:
            existing(record)
        finally:
            callback(record)

    checkpointer.on_checkpoint = chained
