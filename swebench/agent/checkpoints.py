"""Per-step environment checkpoints for a rollout.

An RL rollout that can restart from any step needs, for each step, a snapshot
of the environment as it stood *after* that step. The naive reading — snapshot
every step — pays for hundreds of captures per episode, most of which record
nothing new: an agent spends most of its steps reading files, grepping, and
thinking.

Two observations make it cheap:

**Only mutations matter.** The runtime's shell is stateless (every call is a
fresh ``sh -c`` with an explicit working directory), so nothing carries across
steps except the filesystem and background processes. A step that only read
things leaves the environment byte-identical, so the previous snapshot already
*is* that step's state. :class:`MutationTracker` marks the steps that could
have changed something; the rest reuse the last snapshot id. The map from step
to snapshot stays complete either way, so replay is unaffected.

**A checkpoint need not include memory.** Replay restores the disk and
re-feeds the transcript, so the VM's memory image is dead weight — hence
``disk_only`` snapshots by default. The cost of one is roughly the bytes the
step wrote.

One requirement comes with ``disk_only``: a sandbox created from such a
snapshot **cold-boots**, so its runtime is only there if the microVM template
declares a startup command that launches it (AgentENV re-runs a snapshot's
startup command after a cold boot; a template captured from a hand-started
process carries none). Without one, re-boarding and replaying land on
sandboxes whose runtime is missing; build the template through AgentENV's
template API with ``startCmd`` set (``aenv snapshot create`` records none). Ash cannot repair that from outside --
it reaches a sandbox only through its runtime -- so
:meth:`~swebench.sandbox.AshSession.swap_sandbox` probes a replacement before
adopting it and keeps the current sandbox when the probe fails, and
``mode: full`` (resume, which restores the running runtime) is the alternative
when a template cannot declare one.

The remaining subtlety is layer accounting. Each capture adds a layer to the
sandbox's chain; the server compacts the chain once it crosses a configured
budget, and a compacted capture is visible here as the layer count *dropping*.
The running sandbox's own layer stack is never compacted, so from then on every
capture would re-compact the whole chain — unless the session continues on a
sandbox started from that compact snapshot. That is what :meth:`Checkpointer.
after_step` does when it sees the count drop: it re-boards. Watching for the
drop rather than counting steps means the policy follows whatever trigger the
server is configured with (layer count, chain size, or both) with nothing to
keep in sync.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .pipeline import CallContext, ToolInterceptor, Verdict, Continue


#: Tools that cannot change the environment: they read, search, or fetch.
#: Anything not listed is assumed to mutate, because a wrong "clean" verdict
#: silently loses a step's state while a wrong "dirty" one only costs a cheap
#: capture.
READ_ONLY_TOOLS = frozenset({
    "grep_files",
    "web_fetch",
    "web_search",
    "wait_for_events",
})

#: ``text_editor`` sub-commands that only read.
READ_ONLY_EDITOR_COMMANDS = frozenset({"view"})

#: ``process`` sub-commands that only read. ``kill`` changes what is running,
#: which is environment state a later step can observe.
READ_ONLY_PROCESS_COMMANDS = frozenset({"read", "peek", "list", "status"})


def _failed_to_grow(previous: Optional[int], current: Optional[int]) -> bool:
    return previous is not None and current is not None and current <= previous


def call_mutates(tool_name: str, args: dict) -> bool:
    """Whether a tool call could have changed the environment.

    Conservative by construction: ``shell`` is always treated as mutating
    because a command's effects cannot be read off its text (``ls`` reads,
    ``ls > out`` writes), and a bash-only panel would otherwise have every
    step misjudged.
    """
    if tool_name in READ_ONLY_TOOLS:
        return False
    if tool_name == "text_editor":
        return str(args.get("command", "")) not in READ_ONLY_EDITOR_COMMANDS
    if tool_name == "process":
        return str(args.get("command", "")) not in READ_ONLY_PROCESS_COMMANDS
    return True


class MutationTracker(ToolInterceptor):
    """Flags whether any call since the last checkpoint could have mutated.

    Mounted outermost so it also sees calls the inner guardrails reject: a
    rejected call changes nothing, but it is cheaper to over-count than to
    reason about which rejections are total.

    Also tracks whether the episode may have live background processes: a
    disk-only checkpoint captures the filesystem but not processes, so a
    replay of a step taken while a background process ran diverges -- the
    process is gone, its pids answer errors, its unflushed output never made
    it to disk. `may_have_background` turns from best-effort bookkeeping into
    a per-record flag the replay tooling can warn about. It latches on
    `shell(background=true)` and clears only on `process(kill)` with no way
    to know *which* process died, so it over-reports -- the flag means
    "replay may diverge", not "will".
    """

    fail_mode = "open"

    def __init__(self) -> None:
        self._dirty = False
        self._background_starts = 0

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def may_have_background(self) -> bool:
        return self._background_starts > 0

    def clear(self) -> None:
        self._dirty = False

    def before(self, ctx: CallContext) -> Verdict:
        if call_mutates(ctx.tool_name, ctx.args):
            self._dirty = True
        if ctx.tool_name == "shell" and bool(ctx.args.get("background")):
            self._background_starts += 1
        elif (ctx.tool_name == "process"
              and str(ctx.args.get("command", "")) == "kill"
              and self._background_starts > 0):
            self._background_starts -= 1
        return Continue()


@dataclass
class CheckpointRecord:
    """One step's checkpoint, as recorded in the trajectory."""

    turn: int
    snapshot_id: str
    #: False when this step did not produce its own snapshot.
    captured: bool
    #: Why, when it did not. "clean" means nothing could have changed and the
    #: previous snapshot already is this step's state -- free and expected.
    #: "failed" means a capture was attempted and did not happen, which is an
    #: alarm: checkpointing has stopped working. Recording only `captured`
    #: conflated the two, and a run whose captures were all failing looked
    #: exactly like a run with a very clean workload.
    reason: str = "captured"
    disk_only: bool = False
    #: Wall time of the capture call; 0.0 for reused steps.
    capture_seconds: float = 0.0
    rootfs_layers: Optional[int] = None
    memory_layers: Optional[int] = None
    chain_size_mb: Optional[int] = None
    #: Set when this checkpoint triggered continuing on a new sandbox.
    reboarded: bool = False
    #: Set when the re-board first squashed the whole chain to reset the
    #: lineage's inherited-prefix ratchet.
    lineage_squashed: bool = False
    #: True when background processes may have been alive at this step. A
    #: disk-only replay of such a step will not have them.
    live_background: bool = False


@dataclass
class Checkpointer:
    """Drives per-step checkpoints and layer-chain upkeep for one episode.

    ``session`` is an :class:`~swebench.sandbox.AshSession`; only its
    ``snapshot`` / ``swap_sandbox`` / ``supports_snapshot`` surface is used, so
    a test double is a small object.
    """

    session: object
    tracker: Optional[MutationTracker] = None
    #: Snapshot every step, not just mutating ones. Useful when a run's
    #: analysis wants a distinct snapshot id per step.
    always: bool = False
    disk_only: bool = True
    #: Continue on a new sandbox when a capture shows the chain was compacted.
    reboard: bool = True
    #: Re-boarding resets the *suffix* but ratchets the inherited prefix up by
    #: one layer per cycle -- captures can never compact layers the sandbox
    #: did not produce, so the prefix only grows, and at the format's
    #: 255-layer stack cap the lineage dies (~20k captures from a fresh
    #: start). Squashing is the repository-side merge that CAN fold the
    #: prefix; doing it at a re-board whose target is already this deep
    #: resets the ratchet and removes the ceiling on trajectory length.
    #: Costs one repo-side merge of the whole chain (chain bytes / store
    #: write bandwidth) plus a duplicate of that data until the old layers
    #: are reclaimed. 0 disables.
    squash_lineage_at: int = 128
    name_prefix: str = ""
    #: Reports each checkpoint (e.g. to a trace stream).
    on_checkpoint: Optional[Callable[[CheckpointRecord], None]] = None
    #: Called after every recorded checkpoint to persist the run so far. A
    #: checkpoint without the transcript and the step map beside it is not
    #: resumable: the snapshots survive an interrupted run but nothing says
    #: which step each belongs to. Learned the hard way -- a 5-hour run was
    #: killed with 300 snapshots on the server and no trajectory on disk,
    #: because saving happened only after a clean finish.
    persist: Optional[Callable[["Checkpointer"], None]] = None

    records: list[CheckpointRecord] = field(default_factory=list)
    latest_snapshot_id: Optional[str] = None
    _failed_captures: int = 0
    _previous_layers: Optional[int] = None
    _previous_memory_layers: Optional[int] = None
    _consecutive_compactions: int = 0
    _warned_budget: bool = False

    def enabled(self) -> bool:
        return bool(getattr(self.session, "supports_snapshot", lambda: False)())

    def step_map(self) -> dict[int, str]:
        """Step number -> snapshot holding that step's environment."""
        return {record.turn: record.snapshot_id for record in self.records}

    def after_step(self, turn: int) -> Optional[CheckpointRecord]:
        """Checkpoint the environment as it stands after step ``turn``.

        Returns the record, or ``None`` when checkpointing is unavailable and
        nothing was recorded.
        """
        if not self.enabled():
            return None

        live_background = bool(self.tracker
                               and self.tracker.may_have_background)
        mutated = self.always or self.tracker is None or self.tracker.dirty
        if not mutated and self.latest_snapshot_id:
            # Nothing could have changed: the previous snapshot is this step's
            # state, so record the mapping without paying for a capture.
            return self._record(CheckpointRecord(
                turn=turn, snapshot_id=self.latest_snapshot_id,
                captured=False, reason="clean", disk_only=self.disk_only,
                live_background=live_background))

        name = f"{self.name_prefix}step-{turn}" if self.name_prefix else None
        capture_started = time.monotonic()
        snapshot = self.session.snapshot(name=name, disk_only=self.disk_only)
        capture_seconds = time.monotonic() - capture_started
        if snapshot is None:
            # Capture failed (or the backend declined). Fall back to the last
            # good snapshot so the map stays complete and monotonic -- but say
            # so, because a run that has silently stopped checkpointing is
            # indistinguishable from one with nothing to capture.
            self._failed_captures += 1
            if self._failed_captures in (1, 10, 100) or (
                    self._failed_captures % 250 == 0):
                import logging
                logging.getLogger(__name__).warning(
                    "checkpoint capture has failed %d time(s); steps are "
                    "mapping to an older snapshot and replay from them will "
                    "be wrong", self._failed_captures)
            if not self.latest_snapshot_id:
                return None
            return self._record(CheckpointRecord(
                turn=turn, snapshot_id=self.latest_snapshot_id,
                captured=False, reason="failed", disk_only=self.disk_only,
                live_background=live_background))

        if self.tracker is not None:
            self.tracker.clear()
        self.latest_snapshot_id = snapshot.id
        record = CheckpointRecord(
            turn=turn, snapshot_id=snapshot.id, captured=True,
            disk_only=self.disk_only,
            capture_seconds=capture_seconds,
            rootfs_layers=snapshot.rootfs_layers,
            memory_layers=snapshot.memory_layers,
            chain_size_mb=snapshot.chain_size_mb,
            live_background=live_background,
        )

        # Every uncompacted capture adds exactly one layer to each chain it
        # writes, so a count that FAILED TO GROW means the server compacted
        # that chain and this snapshot is its compact base: continue from it,
        # or every later capture re-compacts. `<=` rather than `<`: a chain
        # whose per-cycle suffix merges into a single layer compacts at the
        # same count it had before -- seen live as a constant 13 with a
        # re-compaction every step and a drop that never came.
        #
        # BOTH chains are watched. Full snapshots carry a memory chain that
        # compacts on its own schedule (memory intervals are much larger than
        # disk deltas, so it hits a shared size budget far sooner), and each
        # of its compactions writes a merged layer close to the whole VM's
        # memory. Watching only rootfs sat through a live episode where the
        # memory chain compacted on 10 of 32 captures -- ~8 GB of merged
        # layers -- while the rootfs count grew +1 every time.
        compacted = _failed_to_grow(self._previous_layers,
                                    snapshot.rootfs_layers)
        # Only a real memory chain participates: disk-only snapshots report
        # zero memory layers, and 0 <= 0 must not read as compaction.
        if (snapshot.memory_layers or 0) >= 1:
            compacted = compacted or _failed_to_grow(
                self._previous_memory_layers, snapshot.memory_layers)
        if snapshot.rootfs_layers is not None or snapshot.memory_layers:
            if compacted and self.reboard:
                target = snapshot
                if self._lineage_needs_squash(snapshot):
                    squashed = self.session.squash_snapshot(snapshot)
                    squashed_id = getattr(squashed, "id", None)
                    if squashed_id and squashed_id != snapshot.id:
                        target = squashed
                        record.lineage_squashed = True
                    # A failed squash falls back to the plain re-board: the
                    # ratchet keeps walking and the next re-board retries.
                record.reboarded = bool(self.session.swap_sandbox(target))
            if record.reboarded:
                self._previous_layers = None
                self._previous_memory_layers = None
            else:
                self._previous_layers = snapshot.rootfs_layers
                self._previous_memory_layers = (
                    snapshot.memory_layers
                    if (snapshot.memory_layers or 0) >= 1 else None)

            # A budget smaller than a single step's writes degrades silently:
            # every capture compacts, and with re-boarding that means a swap
            # every other step. It still works -- say so once, loudly, because
            # the fix is a config value, not code.
            self._consecutive_compactions = (
                self._consecutive_compactions + 1 if compacted else 0)
            if self._consecutive_compactions >= 3 and not self._warned_budget:
                self._warned_budget = True
                import logging
                logging.getLogger(__name__).warning(
                    "every recent capture compacted the layer chain: the "
                    "server's compaction budget (max_chain_size_mib) is "
                    "likely smaller than a single step's writes for this "
                    "workload; raise it to amortize compaction over more "
                    "steps")

        return self._record(record)

    def _lineage_needs_squash(self, snapshot) -> bool:
        """Whether a re-board target's inherited prefix has grown deep.

        A just-compacted snapshot is prefix + one merged layer, so its layer
        count IS the lineage depth. Either chain qualifying is enough: full
        mode's memory prefix ratchets exactly like the rootfs one.
        """
        if self.squash_lineage_at <= 0:
            return False
        if not callable(getattr(self.session, "squash_snapshot", None)):
            return False
        return ((snapshot.rootfs_layers or 0) > self.squash_lineage_at
                or (snapshot.memory_layers or 0) > self.squash_lineage_at)

    def _record(self, record: CheckpointRecord) -> CheckpointRecord:
        self.records.append(record)
        if self.on_checkpoint:
            self.on_checkpoint(record)
        if self.persist:
            # Best effort, and never in the way: a failed write costs
            # resumability from this step, while raising here would cost the
            # step itself.
            try:
                self.persist(self)
            except Exception as error:
                import logging
                logging.getLogger(__name__).warning(
                    "could not persist checkpoint state: %s", error)
        return record

    def as_info(self) -> dict:
        """The checkpoint block a trajectory carries, for replay to read."""
        return {
            "step_snapshots": self.step_map(),
            "disk_only": self.disk_only,
            "captured": sum(1 for r in self.records if r.captured),
            # Surfaced rather than left to be counted: a nonzero value means
            # some steps map to an older snapshot than they should.
            "failed_captures": self._failed_captures,
            "records": [vars(record) for record in self.records],
        }


def install(agent, session, *, always: bool = False, disk_only: bool = True,
            reboard: bool = True, squash_lineage_at: int = 128,
            name_prefix: str = "",
            trajectory_path=None,
            on_checkpoint: Optional[Callable[[CheckpointRecord], None]] = None,
            ) -> Checkpointer:
    """Wire per-step checkpoints into an agent's loop.

    Mounts a :class:`MutationTracker` outermost on the agent's tool pipeline
    and checkpoints from ``before_query``, which fires once per step *before*
    the next model call — that is, after the previous step's tool calls have
    all completed, which is exactly the state a replay of that step needs. The
    first firing therefore records the environment as it stood before step 1.
    """
    tracker = MutationTracker()
    # Compose rather than replace: an agent may already carry a pipeline from
    # `execution.interceptors`, and clobbering it would silently drop that
    # policy. Either way the tracker ends up outermost, where it also sees
    # calls the inner guardrails reject.
    from .interceptors import default_pipeline
    from .pipeline import ToolPipeline
    existing = getattr(agent, "pipeline", None)
    agent.pipeline = (ToolPipeline([tracker, *existing.interceptors])
                      if existing is not None
                      else default_pipeline(extra=[tracker]))

    # Persisting is part of checkpointing, not a harness's responsibility to
    # remember: whoever turns checkpoints on gets a resumable record, and a
    # new harness cannot forget to add it.
    persist = None
    if trajectory_path is not None:
        def persist(checkpointer_ref, path=trajectory_path, agent_ref=agent):
            agent_ref.trajectory.info.setdefault("exit_status", "in_progress")
            agent_ref.trajectory.info["checkpoints"] = checkpointer_ref.as_info()
            agent_ref.trajectory.cost = agent_ref.cost
            agent_ref.trajectory.save(path)

    checkpointer = Checkpointer(
        session=session, tracker=tracker, always=always, disk_only=disk_only,
        reboard=reboard, squash_lineage_at=squash_lineage_at,
        name_prefix=name_prefix, on_checkpoint=on_checkpoint, persist=persist,
    )

    def checkpoint_before_query(agent_ref, _conv) -> None:
        # api_calls counts completed model calls, so this labels the step whose
        # tool calls just finished; 0 on the first firing (the initial state).
        # turn_base shifts a resumed-with-history run so its steps continue
        # the seeded transcript's numbering -- turn N must always mean "the
        # transcript's first N assistant messages", whichever run wrote them.
        checkpointer.after_step(
            getattr(agent_ref, "turn_base", 0) + agent_ref.cost.api_calls)

    agent.before_query_hooks = list(agent.before_query_hooks) + [
        checkpoint_before_query]
    agent.checkpointer = checkpointer
    return checkpointer
