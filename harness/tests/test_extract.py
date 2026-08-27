"""Post-hoc extraction from snapshots.

The point of these: extraction is a function of a snapshot, so it needs no hook in
the execution plane, can run at any step, and can be re-run after the fact. A fake
pool stands in for AgentENV.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from harness.core import events as E
from harness.core.journal import JournalWriter
from harness.extract import (ExtractContext, SnapshotExtractor, patch_extractor,
                             run_extract)


# --- fakes -----------------------------------------------------------------
@dataclass
class FakeResult:
    output: str = ""
    is_error: bool = False


@dataclass
class FakeSandbox:
    snapshot_id: str
    files: Dict[str, str] = field(default_factory=dict)
    calls: List[str] = field(default_factory=list)
    destroyed: bool = False

    async def call(self, tool, **kwargs):
        command = kwargs.get("command", "")
        self.calls.append(command)
        if "rev-parse" in command:
            return FakeResult(output="deadbeef\n")
        if "--others" in command:
            return FakeResult(output="\n".join(self.files.get("untracked", [])))
        if "diff" in command:
            return FakeResult(output=self.files.get("diff", ""))
        return FakeResult()


class FakePool:
    """Records every spawn/destroy so leaks are visible."""

    def __init__(self, per_snapshot: Optional[Dict[str, dict]] = None):
        self.per_snapshot = per_snapshot or {}
        self.spawned: List[FakeSandbox] = []
        self.fail_on: set = set()

    async def spawn(self, image: str):
        if image in self.fail_on:
            raise RuntimeError("cannot restore %s" % image)
        sandbox = FakeSandbox(snapshot_id=image, files=dict(self.per_snapshot.get(image, {})))
        self.spawned.append(sandbox)
        return sandbox

    async def destroy(self, sandbox):
        sandbox.destroyed = True

    def supports_snapshot(self) -> bool:
        return True


def journal_with_steps(tmp_path, steps, session="ses_1"):
    """A journal whose checkpoints are ``[(step, snapshot_id), ...]``."""
    path = tmp_path / "j.jsonl"
    journal = JournalWriter(path, run_id="r1")
    journal.emit(E.SESSION_REF, native_session_id=session)
    for step, snapshot in steps:
        journal.emit(E.CHECKPOINT_CAPTURED, step=step, snapshot_id=snapshot,
                     session_ckpt=session, reason="captured", captured=True,
                     disk_only=True)
    journal.close()
    return path


async def echo_extractor(sandbox, context: ExtractContext):
    return {"snapshot": sandbox.snapshot_id, "step": context.step}


# --- the default: extract the final state ----------------------------------
def test_extracts_the_last_step_by_default(tmp_path):
    path = journal_with_steps(tmp_path, [(0, "snap-0"), (1, "snap-1"), (2, "snap-2")])
    pool = FakePool()
    results = asyncio.run(SnapshotExtractor(pool).extract_journal(path, echo_extractor))
    assert [r.answer["snapshot"] for r in results] == ["snap-2"]
    assert [r.step for r in results] == [2]


def test_every_sandbox_is_destroyed(tmp_path):
    """A per-step sweep must not leak one sandbox per step."""
    path = journal_with_steps(tmp_path, [(0, "snap-0"), (1, "snap-1")])
    pool = FakePool()
    asyncio.run(SnapshotExtractor(pool).extract_journal(path, echo_extractor,
                                                       every_step=True))
    assert all(s.destroyed for s in pool.spawned)


# --- any step, which is what in-band extraction cannot do ------------------
def test_extracts_a_chosen_step(tmp_path):
    path = journal_with_steps(tmp_path, [(0, "snap-0"), (1, "snap-1"), (2, "snap-2")])
    results = asyncio.run(
        SnapshotExtractor(FakePool()).extract_journal(path, echo_extractor, step=1)
    )
    assert results[0].answer == {"snapshot": "snap-1", "step": 1}


def test_unknown_step_is_an_error_not_a_silent_empty(tmp_path):
    path = journal_with_steps(tmp_path, [(1, "snap-1")])
    with pytest.raises(KeyError):
        asyncio.run(SnapshotExtractor(FakePool()).extract_journal(
            path, echo_extractor, step=99))


def test_every_step_walks_the_run(tmp_path):
    path = journal_with_steps(tmp_path, [(0, "snap-0"), (1, "snap-1"), (2, "snap-2")])
    results = asyncio.run(SnapshotExtractor(FakePool()).extract_journal(
        path, echo_extractor, every_step=True))
    assert [r.step for r in results] == [0, 1, 2]


def test_clean_steps_are_not_restored_twice(tmp_path):
    """A read-only step maps to the previous capture: same environment, so
    restoring it again would cost a sandbox for an identical answer."""
    path = journal_with_steps(tmp_path, [(1, "snap-1"), (2, "snap-1"), (3, "snap-3")])
    pool = FakePool()
    results = asyncio.run(SnapshotExtractor(pool).extract_journal(
        path, echo_extractor, every_step=True))
    assert [r.snapshot_id for r in results] == ["snap-1", "snap-3"]
    assert len(pool.spawned) == 2


# --- step 0 removes the need for a baseline probe --------------------------
def test_pristine_snapshot_is_offered_to_the_extractor(tmp_path):
    """Step 0 is the pristine environment, so an extractor can diff against it
    instead of guessing a baseline before the agent starts."""
    path = journal_with_steps(tmp_path, [(0, "snap-0"), (1, "snap-1")])
    pool = FakePool()
    seen = {}

    async def extractor(sandbox, context):
        seen["pristine"] = context.pristine.snapshot_id if context.pristine else None
        return "ok"

    asyncio.run(SnapshotExtractor(pool, with_pristine=True).extract_journal(
        path, extractor))
    assert seen["pristine"] == "snap-0"


def test_no_pristine_when_step_zero_is_absent(tmp_path):
    path = journal_with_steps(tmp_path, [(3, "snap-3")])
    seen = {}

    async def extractor(sandbox, context):
        seen["pristine"] = context.pristine
        return "ok"

    asyncio.run(SnapshotExtractor(FakePool(), with_pristine=True).extract_journal(
        path, extractor))
    assert seen["pristine"] is None


def test_pristine_is_not_restored_unless_asked(tmp_path):
    path = journal_with_steps(tmp_path, [(0, "snap-0"), (1, "snap-1")])
    pool = FakePool()
    asyncio.run(SnapshotExtractor(pool).extract_journal(path, echo_extractor))
    assert [s.snapshot_id for s in pool.spawned] == ["snap-1"]


# --- failure handling ------------------------------------------------------
def test_a_failed_restore_is_reported_not_raised(tmp_path):
    path = journal_with_steps(tmp_path, [(1, "snap-1"), (2, "snap-2")])
    pool = FakePool()
    pool.fail_on.add("snap-1")
    results = asyncio.run(SnapshotExtractor(pool).extract_journal(
        path, echo_extractor, every_step=True))
    by_step = {r.step: r for r in results}
    assert not by_step[1].ok and "cannot restore" in by_step[1].error
    assert by_step[2].ok               # the sweep continued


def test_a_raising_extractor_does_not_stop_the_sweep(tmp_path):
    path = journal_with_steps(tmp_path, [(1, "snap-1"), (2, "snap-2")])

    async def flaky(sandbox, context):
        if context.step == 1:
            raise ValueError("bad parse")
        return "fine"

    results = asyncio.run(SnapshotExtractor(FakePool()).extract_journal(
        path, flaky, every_step=True))
    assert [r.ok for r in results] == [False, True]
    assert "bad parse" in results[0].error


def test_journal_without_checkpoints_yields_nothing(tmp_path):
    path = tmp_path / "empty.jsonl"
    journal = JournalWriter(path, run_id="r")
    journal.emit(E.RUN_STARTED, slot="codex")
    journal.close()
    assert asyncio.run(SnapshotExtractor(FakePool()).extract_journal(
        path, echo_extractor)) == []


# --- the SWE-bench extractor ----------------------------------------------
def test_patch_extractor_stages_only_agent_created_files(tmp_path):
    """The baseline comes from the pristine snapshot, so an image-shipped
    build/ tree is not staged."""
    path = journal_with_steps(tmp_path, [(0, "snap-0"), (1, "snap-1")])
    pool = FakePool({
        "snap-0": {"untracked": ["build/lib.so"]},
        "snap-1": {"untracked": ["build/lib.so", "new_feature.py"],
                   "diff": "diff --git a/new_feature.py b/new_feature.py"},
    })
    results = asyncio.run(SnapshotExtractor(pool, with_pristine=True).extract_journal(
        path, patch_extractor()))
    assert results[0].answer.startswith("diff --git a/new_feature.py")

    target = [s for s in pool.spawned if s.snapshot_id == "snap-1"][0]
    staged = [c for c in target.calls if "add" in c]
    assert any("new_feature.py" in c for c in staged)
    assert not any("build/lib.so" in c for c in staged)


def test_patch_extractor_reads_the_base_commit_from_the_sandbox(tmp_path):
    path = journal_with_steps(tmp_path, [(1, "snap-1")])
    pool = FakePool({"snap-1": {"diff": "diff --git a/x b/x"}})
    results = asyncio.run(SnapshotExtractor(pool).extract_journal(
        path, patch_extractor()))
    assert results[0].ok
    assert any("rev-parse" in c for c in pool.spawned[0].calls)


def test_run_extract_is_the_sync_entry_point(tmp_path):
    path = journal_with_steps(tmp_path, [(1, "snap-1")])
    results = run_extract(FakePool(), path, echo_extractor)
    assert results[0].answer["snapshot"] == "snap-1"
