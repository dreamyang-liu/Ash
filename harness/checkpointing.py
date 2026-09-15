"""Bridge: ``harness.execution.checkpoints.Checkpointer`` -> ``RollbackLedger``.

The environment half of rollback already exists and is battle-tested (mutation
tracking, clean-step snapshot reuse, layer-chain compaction re-boarding, lineage
squashing). What it lacked for external agents is the *other* half: which
conversation state each snapshot belongs to. This module supplies it.

It hooks in without any slot knowing:

- subscribes to the journal, so ``session.ref`` events keep the conversation
  reference current (a native session id for claude-code/codex/opencode);
- pairs captures at the executor's serialized tool boundary with the native
  call identity supplied by the approval hook (not with a completion counter);
- retains turn-boundary capture for legacy callers without exact identities;
- records the resulting pair via :class:`~harness.rollback.RollbackLedger`, so
  ``fork-plan`` can resolve both halves at any step.

Why turn boundaries and not "every N seconds": snapshotting mid tool call leaves
an unresolved call in the conversation and an ambiguous filesystem, so the pair
would not describe a state the agent could resume from.

Usage::

    bridge = SnapshotBridge.install(journal, session)          # CLI slots
    slot.run(task, journal, mcp)

    # The orchestrator wires exact_mode, the slot's identity hook and the
    # server's ToolBoundary together; an approval hook must NOT take a snapshot.

``session`` only needs the ``supports_snapshot`` / ``snapshot`` /
``swap_sandbox`` / ``squash_snapshot`` surface (an ``AshSession``, or a small
test double).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Optional

from harness.core.events import SESSION_REF, TOOL_FINISHED, TOOL_STARTED, TURN_COMPLETED
from harness.core.journal import JournalWriter
from harness.core.checkpoint_identity import CALL_IDENTITY_VERSION
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
    model_position: Optional[dict] = None
    step: int = 0
    #: Depth of unfinished tool calls; a checkpoint is only safe at zero.
    _inflight: int = 0
    _pending: bool = False
    #: Captures declined because the caller was on an event loop thread.
    _skipped_on_loop: int = 0
    records: list = field(default_factory=list)
    exact_mode: bool = False
    _executed: set = field(default_factory=set)
    _capture_identity: dict = field(default_factory=dict)
    _record_lock: Any = field(default_factory=threading.RLock, repr=False)
    _closed: bool = False

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
        exact_mode: bool = False,
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
            exact_mode=exact_mode,
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
        if exact_mode:
            journal.emit("checkpoint.policy", pairing=CALL_IDENTITY_VERSION)
        return bridge

    # --- journal subscription ---------------------------------------------
    def _on_event(self, record: dict) -> None:
        if self._closed:
            return
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

        if etype == "rollout.model_response":
            position = record.get("model_position")
            if isinstance(position, dict):
                self.model_position = dict(position)
                self._backfill_model_position(self.model_position)
            return

        if etype == "rollout.model_position_unavailable":
            # Never let a checkpoint after response N silently inherit the
            # model position from response N-1.  Such a snapshot remains useful
            # for environment cleanup/debugging but is not a joint branch point.
            self.model_position = None
            return

        if etype == TOOL_STARTED:
            self._inflight += 1
            return

        if etype == TOOL_FINISHED:
            self._inflight = max(0, self._inflight - 1)
            return

        if etype == TURN_COMPLETED and not self.exact_mode:
            self.maybe_checkpoint()

    # --- checkpoint triggers ----------------------------------------------
    def validate_call(self, identity: dict, name: str, args: dict) -> None:
        """Only an approved native call may claim an exact execution boundary."""
        call_id = identity["call_id"]
        record = next((r for r in self.journal.tool_calls()
                       if r["call_id"] == call_id), None)
        if (record is None or record["step"] != identity["step"]
                or record["name"].split("__", 2)[-1] != name
                or record["args"] != args):
            raise ValueError("checkpoint identity does not match the approved tool call")
        if self.journal.tool_finished(call_id):
            # A timed-out request that arrived late must not modify a later
            # prefix after that prefix has already been captured.
            raise ValueError("tool call already finished or timed out before dispatch")

    def record_unavailable(self, step: int, call_id: Optional[str], reason: str, *, detail=None) -> None:
        self.record_pair(step, None, captured=False, reason=reason,
                         call_id=call_id, pairing=CALL_IDENTITY_VERSION,
                         prefix_complete=False, execution_detail=detail)

    def finalize_calls(self) -> None:
        """Keep holes explicit; never renumber a call that did reach execution."""
        if not self.exact_mode:
            return
        recorded = {c.call_id for c in self.ledger.checkpoints}
        for record in self.journal.tool_calls():
            if record["call_id"] not in recorded:
                self.record_unavailable(record["step"], record["call_id"], "not_executed")

    def close(self) -> None:
        """No late capture may publish into a journal whose owner has left."""
        with self._record_lock:
            self._closed = True

    def on_tool_boundary(self, index: int, *, call_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Step boundary for an external agent: its tool call just executed.

        The server holds the same-sandbox gate through execution and capture.
        Journal tool.finished may arrive later (or early on a client timeout),
        so that event is not an executor-quiescence lock. Exact captures also
        carry call_id and proof that executed calls form this linear prefix.
        """
        if self._closed:
            return None
        if self.exact_mode and not call_id:
            self.record_unavailable(index, None, "missing_call_identity")
            return None
        if call_id:
            self._executed.add(call_id)
            calls = self.journal.tool_calls()
            prefix_complete = all(
                (r["call_id"] in self._executed or self.journal.tool_finished(r["call_id"]))
                if r["step"] <= index else r["call_id"] not in self._executed
                for r in calls)
            self._capture_identity = dict(call_id=call_id, pairing=CALL_IDENTITY_VERSION,
                                          prefix_complete=prefix_complete)
        try:
            checkpoint = self.maybe_checkpoint(step=index, force=True)
            if checkpoint is None and call_id:
                self.record_unavailable(index, call_id, "capture_unavailable")
            return checkpoint
        finally:
            self._capture_identity = {}

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
        previous_records = len(self.records)
        record = self.checkpointer.after_step(self.step)
        if record is None or len(self.records) == previous_records:
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

    def record_pair(self, step: int, snapshot_id: Optional[str], *,
                    captured: bool = True, reason: str = "captured",
                    **extra) -> Optional[Checkpoint]:
        """Record a pair whose snapshot somebody ELSE took.

        The stdio server captures in its own process and streams the map back as
        JSONL; the tailer feeds each line here. Pairing is the same as for a
        snapshot this bridge took itself: the current conversation ref if the
        slot has disclosed one, and the backfill machinery otherwise -- a ref
        that arrives after the pair was recorded corrects it retroactively
        rather than leaving half a pair.
        """
        with self._record_lock:
            if self._closed:
                return None
            checkpoint = self.ledger.record(step, snapshot_id, session_ckpt=self.session_ref,
                                            model_position=self.model_position,
                                            reason=reason, captured=captured, **extra)
            self.records.append(checkpoint)
            if snapshot_id and not self.session_ref:
                self._pending = True
            return checkpoint

    # --- Checkpointer callback --------------------------------------------
    def _on_checkpoint(self, record: Any) -> None:
        snapshot_id = getattr(record, "snapshot_id", None)
        reason = getattr(record, "reason", "captured")
        self.record_pair(
            getattr(record, "turn", self.step),
            snapshot_id,
            reason=reason,
            captured=bool(getattr(record, "captured", False)),
            disk_only=bool(getattr(record, "disk_only", True)),
            delta_empty=getattr(record, "delta_empty", None),
            **self._capture_identity,
        )

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
                    delta_empty=checkpoint.delta_empty,
                    call_id=checkpoint.call_id,
                    pairing=checkpoint.pairing,
                    prefix_complete=checkpoint.prefix_complete,
                    model_position=checkpoint.model_position,
                )

    def _backfill_model_position(self, position: dict) -> None:
        """Attach a committed Miles response to captures made by its tools.

        Claude can begin executing a streamed tool call before the gateway has
        received the response's final frame. The environment snapshot is
        therefore captured first; Miles publishes the authoritative response
        position moments later. Append a correction for the still-unpositioned
        suffix instead of mutating the checkpoint journal in place.
        """
        for checkpoint in list(self.ledger.checkpoints):
            if checkpoint.snapshot_id and checkpoint.model_position is None:
                checkpoint.model_position = dict(position)
                self.ledger.record(
                    checkpoint.step,
                    checkpoint.snapshot_id,
                    session_ckpt=checkpoint.session_ckpt,
                    reason="model_position_backfill",
                    delta_empty=checkpoint.delta_empty,
                    call_id=checkpoint.call_id,
                    pairing=checkpoint.pairing,
                    prefix_complete=checkpoint.prefix_complete,
                    model_position=checkpoint.model_position,
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
