"""Restart an episode from any step it checkpointed.

A rollout that checkpointed per step (``swebench/agent/checkpoints.py``) leaves
two things behind: the transcript, and a map from step to the snapshot holding
the environment as it stood after that step. Together they make a step
addressable — restore that snapshot, re-feed the prefix of the transcript, and
the agent is back where it was, free to continue with different sampling or to
fan out into several branches.

Restoring a ``disk_only`` checkpoint cold-boots the sandbox, so the microVM
template must declare a startup command that launches the runtime -- otherwise
the restored sandbox has no runtime to talk to. Checkpoints taken in ``full``
mode resume instead, and bring their processes back with them.

Two facts shape the branching path:

**A branch inherits the snapshot's layers.** Layers a sandbox did not produce
itself can never be compacted by it, so a child of a deep snapshot starts with
a deep immutable prefix: its own checkpoint cycles get shorter, and across
generations the chain approaches the format's stack limit. Squashing the branch
point first hands children a one-layer prefix instead.

**Squashing is worth memoising.** It costs one merge (chain size / store write
bandwidth) and yields a snapshot any number of children can share, so a point
that is branched more than once should be squashed once and reused. This module
keeps that cache per process; a longer-lived cache belongs to whatever drives
the search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

#: Inherit more layers than this and a branch point is squashed before use even
#: for a single child: below it the prefix costs little, above it the child's
#: checkpoint cycles get noticeably shorter.
DEFAULT_SQUASH_LAYER_THRESHOLD = 128


def load_step_snapshots(trajectory_path: Path | str) -> dict[int, str]:
    """Read the step -> snapshot map a checkpointed run saved.

    Keys come back as ints: JSON object keys are strings, and callers index by
    step number.
    """
    data = json.loads(Path(trajectory_path).read_text())
    checkpoints = (data.get("info") or {}).get("checkpoints") or {}
    return {int(step): snapshot_id
            for step, snapshot_id in (checkpoints.get("step_snapshots") or {}).items()}


def load_environment(trajectory_path: Path | str) -> dict:
    """What the saved run ran against: image asked for, resolved reference,
    repository commit, sandbox id. Empty for trajectories saved before this
    was recorded.
    """
    data = json.loads(Path(trajectory_path).read_text())
    return (data.get("info") or {}).get("environment") or {}


#: Environment fields that must agree between a run and its replay.
#:
#: ``base_image`` is the environment the episode descends from, pinned when the
#: task started (digest-pinned for a cold start), so it stays comparable even
#: though a replay starts from a checkpoint. ``base_commit`` is the repository
#: state inside it.
#:
#: Deliberately not ``base_ref`` (nor ``requested_image``): a replay's
#: immediate source *is* the checkpoint snapshot, so comparing it would flag
#: every replay.
COMPARABLE_ENVIRONMENT_FIELDS = ("base_image", "base_commit")


def environment_mismatch(recorded: dict, current: dict) -> list[str]:
    """Fields where a replay's environment disagrees with the recorded one.

    Compared only when both sides know the value, so a trajectory saved before
    environments were recorded does not look like a mismatch.

    A differing ``base_commit`` means the replay is looking at different code.
    A differing ``base_image`` catches the case a name cannot: the same mutable
    tag resolving to different bits, which produces results that look valid and
    are not.
    """
    differences = []
    for field_name in COMPARABLE_ENVIRONMENT_FIELDS:
        before, now = recorded.get(field_name), current.get(field_name)
        if before and now and before != now:
            differences.append(field_name)
    return differences


def replay_caveats(trajectory_path: Path | str, step: int) -> list[str]:
    """Reasons a replay of ``step`` may diverge from the recorded run.

    A disk-only checkpoint captures the filesystem, not processes: a step
    taken while a background process ran replays without it -- its pids
    answer errors, tmpfs scratch and unflushed output are gone. Empty for
    the common all-synchronous episode.
    """
    data = json.loads(Path(trajectory_path).read_text())
    checkpoints = (data.get("info") or {}).get("checkpoints") or {}
    caveats = []
    for record in checkpoints.get("records") or []:
        if record.get("turn") == step and record.get("live_background") \
                and record.get("disk_only", True):
            caveats.append(
                "background processes may have been alive at this step; a "
                "disk-only replay will not have them (their pids will answer "
                "errors, and anything they had not flushed to disk is gone)")
    return caveats


def snapshot_for_step(step_snapshots: dict[int, str], step: int) -> Optional[str]:
    """The snapshot holding the environment as of ``step``.

    Steps that changed nothing are recorded pointing at the previous snapshot,
    so an exact hit is the normal case; the fallback to the nearest earlier
    step covers maps written by a run that only recorded captures.
    """
    if step in step_snapshots:
        return step_snapshots[step]
    earlier = [s for s in step_snapshots if s <= step]
    return step_snapshots[max(earlier)] if earlier else None


def messages_through_step(trajectory_path: Path | str, step: int) -> list[dict]:
    """The transcript prefix belonging to the first ``step`` model calls.

    Cuts after the tool results of the ``step``-th assistant message, which is
    the transcript the agent held when the matching snapshot was taken.
    Assistant messages carry their ``tool_calls``, so the prefix records the
    actions taken and not merely the prose around them -- without those, a
    resumed run could see that the agent said it would edit a file but not
    what edit it made.
    """
    data = json.loads(Path(trajectory_path).read_text())
    messages = data.get("messages") or []
    prefix = []
    assistant_turns = 0
    for message in messages:
        if message.get("role") == "assistant":
            assistant_turns += 1
            if assistant_turns > step:
                break
        wire = _wire_message(message)
        if wire is not None:
            prefix.append(wire)
    return prefix


def _wire_message(message: dict) -> Optional[dict]:
    """A trajectory row as the model-facing message it recorded, or None.

    The trajectory is a record, not a transcript: tool results are stored as
    ``tool_result`` rows carrying evaluation metadata (tool_name, tool_args,
    success), and ``error`` rows exist only in the record -- add_error never
    put them in front of the model. Re-feeding rows verbatim was rejected by
    the first provider to see one ("Invalid Message ... role 'tool_result'"),
    so the translation back to wire format lives here, next to the reader.
    """
    role = message.get("role")
    if role == "error":
        return None
    if role in ("tool_result", "tool"):
        # Both spellings occur: a run records "tool_result", and a run that was
        # itself resumed saves the wire-format "tool" rows it was seeded with.
        # Missing this second case cost a chained resume its whole run -- the
        # id was stripped by the fallback below, litellm invented UUIDs for the
        # results, and Bedrock rejected the conversation ("Expected toolResult
        # blocks ... for the following Ids").
        return {"role": "tool",
                "tool_call_id": message.get("tool_call_id"),
                "content": message.get("content") or ""}
    if role == "assistant":
        wire = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            wire["tool_calls"] = message["tool_calls"]
        return wire
    if role in ("system", "user"):
        return {"role": role, "content": message.get("content") or ""}
    # An unknown role is not silently reshaped into a message that looks
    # plausible and is not: dropping it is the honest failure, and the roles
    # this stack writes are all handled above.
    return None


@dataclass
class BranchPointCache:
    """Memoises squashed equivalents of snapshots used as branch points."""

    session: Any
    layer_threshold: int = DEFAULT_SQUASH_LAYER_THRESHOLD
    _squashed: dict[str, str] = field(default_factory=dict)

    def prepare(self, snapshot_id: str, *, fan_out: int = 1,
                layers: Optional[int] = None) -> str:
        """The best snapshot id to branch from, squashing when it pays.

        Squashes when several children will share the cost, or when the chain
        is deep enough that even one child would inherit an awkward prefix.
        Returns ``snapshot_id`` unchanged when squashing is unavailable or
        fails: a deep chain still works.
        """
        if snapshot_id in self._squashed:
            return self._squashed[snapshot_id]

        if layers is None:
            layers = self._layers_of(snapshot_id)
        worth_it = fan_out >= 2 or (layers is not None
                                    and layers > self.layer_threshold)
        if not worth_it:
            return snapshot_id

        squashed = self.session.squash_snapshot(snapshot_id)
        squashed_id = getattr(squashed, "id", squashed) or snapshot_id
        self._squashed[snapshot_id] = squashed_id
        return squashed_id

    def _layers_of(self, snapshot_id: str) -> Optional[int]:
        pool = getattr(self.session, "_pool", None)
        get_snapshot = getattr(pool, "get_snapshot", None)
        if get_snapshot is None:
            return None
        try:
            loop = self.session._get_loop()
            return loop.run_until_complete(get_snapshot(snapshot_id)).rootfs_layers
        except Exception:
            # Chain facts are an optimisation input; without them, branch
            # from the snapshot as-is.
            return None
