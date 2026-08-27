"""ATIF export + rollback pairing."""

from __future__ import annotations

import pytest

from harness.atif import ATIF_VERSION, journal_to_atif
from harness.core import events as E
from harness.core.journal import JournalWriter, read_journal
from harness.rollback import RollbackLedger, fork_plan, load_checkpoints


def build_journal(path, *, with_checkpoints=True):
    with JournalWriter(path, run_id="run1", agent_id="a1") as journal:
        journal.emit(
            E.RUN_STARTED, slot="opencode", slot_version="1.18.5", model="m1", task_prompt="task"
        )
        ledger = RollbackLedger(journal)
        # turn 1
        journal.emit(E.AGENT_THINKING, text="think 1")
        journal.emit(E.TOOL_STARTED, call_id="c1", name="read", args={"p": "a.txt"})
        journal.emit(E.TOOL_FINISHED, call_id="c1", status="ok", output="contents")
        if with_checkpoints:
            ledger.record(1, "snap-1", session_ckpt="ses_1")
        journal.emit(
            E.TURN_COMPLETED,
            usage={"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 2, "cost_usd": 0.1},
        )
        # turn 2
        journal.emit(E.AGENT_MESSAGE, text="all done")
        if with_checkpoints:
            ledger.record(2, "snap-2", session_ckpt="ses_1")
        journal.emit(
            E.TURN_COMPLETED, usage={"input_tokens": 20, "output_tokens": 7, "cost_usd": 0.2}
        )
        journal.emit(E.SESSION_REF, native_session_id="ses_1")
        journal.emit(E.RUN_FINISHED, status="completed", usage={"input_tokens": 30})
    return read_journal(path)


def test_atif_document_shape(tmp_path):
    document = journal_to_atif(build_journal(tmp_path / "j.jsonl"))
    assert document["schema_version"] == ATIF_VERSION
    assert document["agent"] == {"name": "opencode", "version": "1.18.5", "model": "m1"}
    assert document["session_id"] == "ses_1"
    assert len(document["steps"]) == 2
    # step_id must start at 1 and be contiguous (ATIF validator requirement)
    assert [s["step_id"] for s in document["steps"]] == [1, 2]


def test_atif_step_content_and_metrics(tmp_path):
    document = journal_to_atif(build_journal(tmp_path / "j.jsonl"))
    first, second = document["steps"]
    assert first["reasoning_content"] == "think 1"
    assert first["tool_calls"][0]["name"] == "read"
    assert first["observation"] == "contents"
    assert first["metrics"]["prompt_tokens"] == 10
    assert first["metrics"]["cached_tokens"] == 2
    assert second["message"] == "all done"
    assert document["final_metrics"]["prompt_tokens"] == 30
    assert document["final_metrics"]["cost_usd"] == pytest.approx(0.3)


def test_atif_marks_copied_context_for_forks(tmp_path):
    """Steps replayed from a parent must be filterable, or SFT data is polluted."""
    records = build_journal(tmp_path / "j.jsonl")
    boundary = next(r["seq"] for r in records if r["type"] == E.TURN_COMPLETED)
    document = journal_to_atif(records, copied_through_seq=boundary)
    flags = [s["is_copied_context"] for s in document["steps"]]
    assert flags == [True, False]

    fresh = journal_to_atif(records)
    assert all(s["is_copied_context"] is False for s in fresh["steps"])


def test_atif_carries_checkpoints_and_lineage(tmp_path):
    records = build_journal(tmp_path / "j.jsonl")
    document = journal_to_atif(records, continued_from="traj-parent")
    assert document["continued_trajectory_ref"] == "traj-parent"
    checkpoints = document["extra"]["checkpoints"]
    assert [c["snapshot_id"] for c in checkpoints] == ["snap-1", "snap-2"]


def test_atif_never_emits_zero_steps(tmp_path):
    """ATIF requires steps >= 1 even for a run that died immediately."""
    path = tmp_path / "empty.jsonl"
    with JournalWriter(path, run_id="r") as journal:
        journal.emit(E.RUN_STARTED, slot="codex", task_prompt="do it")
        journal.emit(E.RUN_FINISHED, status="error", error="auth failed")
    document = journal_to_atif(read_journal(path))
    assert len(document["steps"]) == 1
    assert document["steps"][0]["source"] == "system"
    assert document["extra"]["status"] == "error"


# --- rollback --------------------------------------------------------------
def test_ledger_records_pairs_into_journal(tmp_path):
    path = tmp_path / "j.jsonl"
    build_journal(path)
    checkpoints = load_checkpoints(path)
    assert [(c.step, c.snapshot_id, c.session_ckpt) for c in checkpoints] == [
        (1, "snap-1", "ses_1"),
        (2, "snap-2", "ses_1"),
    ]
    assert all(c.is_complete() for c in checkpoints)


def test_incomplete_pair_is_detected(tmp_path):
    """An env snapshot without a session ref cannot restore a black-box agent."""
    path = tmp_path / "half.jsonl"
    with JournalWriter(path) as journal:
        ledger = RollbackLedger(journal)
        checkpoint = ledger.record(1, "snap-1")  # no session_ckpt
    assert checkpoint.is_complete() is False
    assert fork_plan(path, 1)["complete"] is False


def test_fork_plan_picks_latest_snapshot_at_or_before_step(tmp_path):
    path = tmp_path / "j.jsonl"
    build_journal(path)
    plan = fork_plan(path, 5)
    assert plan["step"] == 2 and plan["snapshot_id"] == "snap-2"
    assert plan["session_ckpt"] == "ses_1"
    # the seq boundary is what ATIF export needs for is_copied_context
    assert plan["copied_through_seq"] > 0

    earlier = fork_plan(path, 1)
    assert earlier["snapshot_id"] == "snap-1"


def test_fork_plan_without_snapshots_raises(tmp_path):
    path = tmp_path / "none.jsonl"
    build_journal(path, with_checkpoints=False)
    with pytest.raises(ValueError, match="no snapshot"):
        fork_plan(path, 1)


def test_ledger_at_step_reuses_previous_snapshot_for_clean_steps(tmp_path):
    with JournalWriter(tmp_path / "j.jsonl") as journal:
        ledger = RollbackLedger(journal)
        ledger.record(1, "snap-1", session_ckpt="s")
        ledger.record(2, None, reason="clean")  # read-only step: nothing captured
        assert ledger.at_step(2).snapshot_id == "snap-1"
        assert ledger.step_map() == {1: "snap-1", 2: None}
