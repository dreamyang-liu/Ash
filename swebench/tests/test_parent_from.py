"""--parent-from: branch from the recorded single-pass parent, no parent re-run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swebench import fork_eval
from swebench.fork_eval import Grade, existing_parent, outcome_from_journal


def write_journal(path: Path, steps: int = 3, status: str = "completed") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"v": 2, "type": "run.started", "ts": "2026-09-04T00:00:00Z", "seq": 0,
                "run_id": "parent"}]
    seq = 1
    for step in range(1, steps + 1):
        records.append({"v": 2, "type": "tool.started", "ts": "2026-09-04T00:00:01Z",
                        "seq": seq, "run_id": "parent", "call_id": "c%d" % step,
                        "name": "mcp__ash__shell", "args": {"command": "ls"}})
        records.append({"v": 2, "type": "checkpoint.captured", "ts": "2026-09-04T00:00:02Z",
                        "seq": seq + 1, "run_id": "parent", "step": step,
                        "snapshot_id": "snap-%d" % step, "session_ckpt": "ses-1",
                        "reason": "captured"})
        records.append({"v": 2, "type": "tool.finished", "ts": "2026-09-04T00:00:03Z",
                        "seq": seq + 2, "run_id": "parent", "call_id": "c%d" % step,
                        "status": "ok", "output": "ok"})
        seq += 3
    records.append({"v": 2, "type": "run.result", "ts": "2026-09-04T00:00:09Z", "seq": seq,
                    "run_id": "parent", "text": "done"})
    records.append({"v": 2, "type": "run.finished", "ts": "2026-09-04T00:00:10Z",
                    "seq": seq + 1, "run_id": "parent", "status": status,
                    "usage": {"cost_usd": 1.5}, "error": None})
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_existing_parent_from_batch_dir_and_from_aggregate(tmp_path):
    j = write_journal(tmp_path / "batch" / "shard-3" / "task-a" / "parent.jsonl")
    assert existing_parent(str(tmp_path / "batch"), "task-a") == j
    assert existing_parent(str(tmp_path / "batch"), "task-b") is None
    agg = tmp_path / "final.json"
    agg.write_text(json.dumps({"tasks": [{"task": "task-a", "journal": str(j)}]}))
    assert existing_parent(str(agg), "task-a") == j
    assert existing_parent(str(agg), "task-b") is None


def test_outcome_from_journal_rebuilds_what_the_loop_reads(tmp_path):
    j = write_journal(tmp_path / "parent.jsonl", steps=3)
    outcome = outcome_from_journal(j)
    assert outcome.status == "completed" and outcome.ok
    assert outcome.checkpoints == 3
    assert outcome.usage == {"cost_usd": 1.5}
    assert outcome.final_text == "done"
    assert outcome.journal_path == j


class FakeBench(fork_eval.Benchmark):
    name = "fake"

    def __init__(self, grade):
        self._grade = grade
        self.graded = []

    def instance(self, raw):
        return {"instance_id": raw, "repo": "r", "image": "img", "problem": "p",
                "f2p": [], "p2p": []}

    def prompt(self, instance):
        return "prompt"

    def branch_prompt(self, instance, verdict, hint):
        return "branch"

    def grade(self, snapshot_id, instance, backend):
        self.graded.append(snapshot_id)
        return self._grade


def test_run_one_reuses_the_recorded_parent_and_never_runs_a_fresh_one(tmp_path, monkeypatch):
    src = write_journal(tmp_path / "base" / "shard-0" / "task-a" / "parent.jsonl", steps=4)

    def no_run(*a, **k):
        raise AssertionError("a fresh parent attempt was started")
    monkeypatch.setattr(fork_eval, "run_attempt", no_run)

    bench = FakeBench(Grade(resolved=True, f2p_pass=True, p2p_pass=True, p2p_ran=True))
    args = SimpleNamespace(rounds=2, slot="claude-code", model="m", analyst_model="m",
                           analyst_tokens=1000, timeout=10.0, runtime_bin="runtime/ash-runtime",
                           parent_from=str(tmp_path / "base"))
    out_dir = tmp_path / "run" / "task-a"
    attempts = fork_eval.run_one(orch=None, args=args, raw="task-a", schedule=[4, 3],
                                 out_dir=out_dir, bench=bench)

    assert [a.name for a in attempts] == ["parent"]
    assert attempts[0].grade.resolved
    # graded from the recorded run's LAST snapshot, and the journal now lives in
    # the run dir under the loop's own name so fork_plan / --regrade find it.
    assert bench.graded == ["snap-4"]
    assert (out_dir / "parent.jsonl").read_text() == src.read_text()
    assert attempts[0].outcome.journal_path == out_dir / "parent.jsonl"


def test_parent_from_without_a_journal_refuses_rather_than_silently_rerunning(tmp_path, monkeypatch):
    monkeypatch.setattr(fork_eval, "run_attempt",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    bench = FakeBench(Grade())
    args = SimpleNamespace(rounds=0, slot="s", model="m", analyst_model="m", analyst_tokens=1,
                           timeout=1.0, runtime_bin="runtime/ash-runtime",
                           parent_from=str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no parent journal"):
        fork_eval.run_one(None, args, "task-x", [1], tmp_path / "out", bench)
