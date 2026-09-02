"""Batch runner + resource ledger + reaper."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import timedelta

import pytest

from harness.orchestrator.batch import BatchRunner, is_retryable, load_tasks
from harness.core import events as E
from harness.core.journal import JournalWriter
from harness.orchestrator.reap import ReapPlan, Reaper, parse_duration
from harness.orchestrator.resources import ResourceLedger

# --- retry classification --------------------------------------------------
@pytest.mark.parametrize(
    "error",
    ["HTTP 429 Too Many Requests", "rate limit exceeded", "upstream 503", "connection reset"],
)
def test_transient_errors_are_retryable(error):
    assert is_retryable(error) is True


@pytest.mark.parametrize(
    "error",
    [
        None,
        "exit code 1",
        "context window exceeded (200000 tokens)",
        "stop_reason: refusal",
        "AgentSafetyRefusalError",
    ],
)
def test_agent_outcomes_are_not_retried(error):
    """Retrying a real outcome turns agent behaviour into environment noise."""
    assert is_retryable(error) is False


# --- ledger ----------------------------------------------------------------
def test_claim_then_release_is_not_an_orphan(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    with ledger.run("run-1") as claim:
        claim.sandbox("sb-1")
        claim.released("sandbox", "sb-1")
    assert ledger.orphans() == []


def test_unreleased_claim_from_a_dead_pid_is_an_orphan(tmp_path):
    path = tmp_path / "res.jsonl"
    ledger = ResourceLedger(path)
    # A pid that cannot exist: recorded by hand because the live process is us.
    ledger._append("claim", run_id="run-2", kind="snapshot", id="snap-9", pid=999999999)
    orphans = ledger.orphans()
    assert [(o.kind, o.id) for o in orphans] == [("snapshot", "snap-9")]


def test_live_process_claims_are_left_alone(tmp_path):
    """Reaping a resource a running batch still needs is the worst outcome."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="mine", kind="sandbox", id="sb-live", pid=os.getpid())
    assert ledger.orphans() == []


def test_keep_flag_protects_a_snapshot(tmp_path):
    """Snapshots a fork depends on must survive their creating run."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="r", kind="snapshot", id="pinned",
                   pid=999999999, keep=True)
    assert ledger.orphans() == []


def test_run_done_without_release_is_still_an_orphan(tmp_path):
    """A kill between "run finished" and "resource freed" must not hide a leak."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    with ledger.run("run-3") as claim:
        claim.snapshot("snap-x")
    assert [o.id for o in ledger.orphans()] == ["snap-x"]


def test_torn_line_does_not_abort_the_reap(tmp_path):
    path = tmp_path / "res.jsonl"
    ledger = ResourceLedger(path)
    ledger._append("claim", run_id="r", kind="snapshot", id="good", pid=999999999)
    with path.open("a") as fh:
        fh.write('{"event": "claim", "kind": "snapsh\n')   # truncated write
    assert [o.id for o in ledger.orphans()] == ["good"]


def test_attach_records_snapshots_from_journal_events(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    with ledger.run("run-4") as claim:
        with JournalWriter(tmp_path / "j.jsonl") as journal:
            claim.attach(journal)
            journal.emit(E.CHECKPOINT_CAPTURED, step=1, snapshot_id="snap-a")
            journal.emit(E.CHECKPOINT_CAPTURED, step=2, snapshot_id="snap-a")  # clean reuse
            journal.emit(E.CHECKPOINT_CAPTURED, step=3, snapshot_id="snap-b")
    assert claim.snapshots == ["snap-a", "snap-b"]        # deduped


def test_compact_drops_released_entries(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    with ledger.run("r") as claim:
        claim.sandbox("sb-1")
        claim.released("sandbox", "sb-1")
        claim.snapshot("snap-1")
    assert ledger.compact() > 0
    assert [r.id for r in ledger.resources()] == ["snap-1"]


# --- reaper ----------------------------------------------------------------
class FakeClient:
    def __init__(self, sandboxes=(), snapshots=()):
        self._sandboxes = list(sandboxes)
        self._snapshots = list(snapshots)
        self.deleted = []
        self.fail = set()

    def list_sandboxes(self):
        return list(self._sandboxes)

    def list_snapshots(self):
        return list(self._snapshots)

    def delete_sandbox(self, sandbox_id):
        return self._delete(sandbox_id)

    def delete_snapshot(self, snapshot_id):
        return self._delete(snapshot_id)

    def _delete(self, resource_id):
        if resource_id in self.fail:
            return False
        self.deleted.append(resource_id)
        return True


def test_snapshots_are_never_planned_even_as_ledger_orphans(tmp_path):
    """The policy this pins REVERSED. A checkpoint used to be "reclaimable
    unless a fork pins it", and the reaper planned dead runs' snapshots for
    deletion -- latent-only because the server answered DELETE with 405. The
    day the endpoint landed, one habitual `harness reap` would have eaten
    every finished run's forkability: "owning run died" describes EVERY
    snapshot worth keeping. Snapshots die only by `harness sweep <journal>
    --yes`, where naming the journal is the consent."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="snapshot", id="orphan", pid=999999999)
    ledger._append("claim", run_id="mine", kind="snapshot", id="mine", pid=os.getpid())
    client = FakeClient(snapshots=[
        {"id": "orphan", "created_at": "2026-01-01T00:00:00Z", "names": []},
        {"id": "mine", "created_at": "2026-01-01T00:00:00Z", "names": []},
    ])

    plan = Reaper(client, ledger).plan()
    assert plan.snapshots == []
    assert "orphan" in plan.kept


def test_include_unknown_needs_an_age_cutoff_and_spares_named(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    client = FakeClient(snapshots=[
        {"id": "old", "created_at": "2020-01-01T00:00:00Z", "names": []},
        {"id": "recent", "created_at": "2999-01-01T00:00:00Z", "names": []},
        {"id": "named", "created_at": "2020-01-01T00:00:00Z", "names": ["golden-base"]},
    ])
    plan = Reaper(client, ledger).plan(
        include_unknown=True, older_than=timedelta(hours=24)
    )
    # Even the widest sweep never touches snapshots: reap reclaims COMPUTE
    # (a running sandbox burns cpu/memory now); a snapshot is inert bytes
    # that might be the one thing someone meant to branch from.
    assert plan.snapshots == []


def test_stale_ledger_entry_for_a_missing_resource_is_skipped(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="snapshot", id="already-gone", pid=999999999)
    plan = Reaper(FakeClient(), ledger).plan()
    assert plan.total() == 0


def test_apply_deletes_sandboxes_before_snapshots(tmp_path):
    """A running sandbox holds its snapshot chain open."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    client = FakeClient()
    plan = ReapPlan(sandboxes=["sb-1"], snapshots=["snap-1"])
    done = Reaper(client, ledger).apply(plan)
    assert client.deleted == ["sb-1", "snap-1"]
    assert done["sandboxes"] == ["sb-1"] and done["snapshots"] == ["snap-1"]
    # released entries are recorded so a second pass does not retry them
    assert all(r.released for r in ledger.resources()) or not ledger.resources()


def test_apply_reports_failures_without_raising(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    client = FakeClient()
    client.fail.add("snap-bad")
    done = Reaper(client, ledger).apply(ReapPlan(snapshots=["snap-bad", "snap-ok"]))
    assert done["failed"] == ["snap-bad"]
    assert done["snapshots"] == ["snap-ok"]


@pytest.mark.parametrize(
    "text,expected",
    [("30s", 30), ("15m", 900), ("2h", 7200), ("1d", 86400)],
)
def test_parse_duration(text, expected):
    assert parse_duration(text).total_seconds() == expected


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("soon")


# --- batch runner ----------------------------------------------------------
FAKE_AGENT = textwrap.dedent(
    """
    import json, os, sys
    marker = os.environ.get("ASH_TEST_FAIL_ONCE")
    if marker and not os.path.exists(marker):
        open(marker, "w").write("x")
        sys.stderr.write("HTTP 429 rate limit\\n")
        sys.exit(1)
    print(json.dumps({"type": "thread.started", "thread_id": "th"}), flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"id": "i1", "item_type": "agent_message",
                               "text": "done " + os.environ.get("ASH_TASK", "?")}}), flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 5, "output_tokens": 2}}), flush=True)
    """
)


@pytest.fixture
def fake_slot(tmp_path, monkeypatch):
    """Register a slot backed by a scripted fake CLI."""
    script = tmp_path / "agent.py"
    script.write_text(FAKE_AGENT)

    from harness.normalize import codex as codex_normalize
    from harness.slots import _REGISTRY
    from harness.slots.cli_base import JsonlCliSlot

    class FakeSlot(JsonlCliSlot):
        name = "fake"
        binary = None
        normalizer = staticmethod(codex_normalize.normalize)

        def build_command(self, task, mcp):
            return [sys.executable, str(script)]

    module = sys.modules[__name__]
    setattr(module, "FakeSlot", FakeSlot)
    monkeypatch.setitem(_REGISTRY, "fake", "%s:FakeSlot" % __name__)
    return FakeSlot


def write_tasks(tmp_path, count, **extra):
    path = tmp_path / "tasks.jsonl"
    with path.open("w") as fh:
        for i in range(count):
            payload = {"id": "t%d" % i, "prompt": "p%d" % i, "cwd": str(tmp_path)}
            payload.update(extra)
            fh.write(json.dumps(payload) + "\n")
    return path


def test_batch_runs_all_tasks_concurrently(tmp_path, fake_slot):
    tasks = load_tasks(write_tasks(tmp_path, 6))
    runner = BatchRunner("fake", tmp_path / "out", workers=3)
    results = runner.run(tasks)

    assert len(results) == 6
    assert runner.counts() == {"completed": 6}
    assert all(r.usage["input_tokens"] == 5 for r in results)
    # each task gets its own journal
    assert len(list((tmp_path / "out").glob("t*.jsonl"))) == 6
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["tasks"] == 6 and summary["workers"] == 3


def test_batch_retries_only_transient_failures(tmp_path, fake_slot):
    marker = tmp_path / "failed-once"
    tasks = load_tasks(write_tasks(tmp_path, 1, env={"ASH_TEST_FAIL_ONCE": str(marker)}))
    runner = BatchRunner("fake", tmp_path / "out", workers=1, max_attempts=2)
    outcome = runner.run(tasks)[0]

    assert outcome.status == "completed"
    assert outcome.attempts == 2          # first attempt hit a 429
    assert marker.exists()


def test_batch_gives_up_after_max_attempts(tmp_path, fake_slot):
    tasks = load_tasks(write_tasks(tmp_path, 1, extra={}))
    # point the slot at a script that always fails
    bad = tmp_path / "bad.py"
    bad.write_text("import sys; sys.stderr.write('HTTP 429\\n'); sys.exit(1)")

    class AlwaysFail(fake_slot):
        def build_command(self, task, mcp):
            return [sys.executable, str(bad)]

    module = sys.modules[__name__]
    setattr(module, "AlwaysFail", AlwaysFail)
    from harness.slots import _REGISTRY

    _REGISTRY["fake"] = "%s:AlwaysFail" % __name__
    try:
        runner = BatchRunner("fake", tmp_path / "out", workers=1, max_attempts=3)
        outcome = runner.run(tasks)[0]
    finally:
        _REGISTRY["fake"] = "%s:FakeSlot" % __name__

    assert outcome.status == "error"
    assert outcome.attempts == 3


def test_batch_skips_tasks_already_finished(tmp_path, fake_slot):
    tasks = load_tasks(write_tasks(tmp_path, 3))
    first = BatchRunner("fake", tmp_path / "out", workers=2)
    first.run(tasks)

    second = BatchRunner("fake", tmp_path / "out", workers=2)
    results = second.run(tasks)
    assert all(r.skipped for r in results)
    assert second.counts() == {"skipped": 3}

    third = BatchRunner("fake", tmp_path / "out", workers=2, resume=False)
    assert all(not r.skipped for r in third.run(tasks))


def test_batch_isolates_a_crashing_task(tmp_path, fake_slot):
    tasks = load_tasks(write_tasks(tmp_path, 3))
    runner = BatchRunner("fake", tmp_path / "out", workers=2)

    original = runner._attempt

    def explode(task, task_id, attempt):
        if task_id == "t1":
            raise RuntimeError("boom")
        return original(task, task_id, attempt)

    runner._attempt = explode
    results = runner.run(tasks)

    statuses = {r.task_id: r.status for r in results}
    assert statuses["t1"] == "error"
    assert statuses["t0"] == "completed" and statuses["t2"] == "completed"


def test_batch_records_resources_for_reaping(tmp_path, fake_slot):
    tasks = load_tasks(write_tasks(tmp_path, 1))
    runner = BatchRunner("fake", tmp_path / "out", workers=1)
    runner.run(tasks)
    # the run finished cleanly, so its ledger marks the run done
    entries = list(runner.ledger.entries())
    assert any(e["event"] == "run_done" for e in entries)


def test_stop_prevents_new_attempts(tmp_path, fake_slot):
    tasks = load_tasks(write_tasks(tmp_path, 2))
    runner = BatchRunner("fake", tmp_path / "out", workers=1)
    runner.stop()
    results = runner.run(tasks)
    assert all(r.status == "killed" for r in results)


def test_snapshot_policy_does_not_depend_on_server_capability(tmp_path):
    """Before the server grew DELETE /snapshots, an "unsupported" listing
    explained why snapshots stayed. Now that deleting IS possible they stay for
    a better reason -- policy, not capability -- and the plan must read the
    same regardless of what the backend supports."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="snapshot", id="orphan", pid=999999999)

    class NoSnapshotDelete(FakeClient):
        def supports_snapshot_delete(self):
            return False

    for client_cls in (FakeClient, NoSnapshotDelete):
        client = client_cls(snapshots=[
            {"id": "orphan", "created_at": "2020-01-01T00:00:00Z", "names": []},
        ])
        plan = Reaper(client, ledger).plan()
        assert plan.snapshots == []
        assert "orphan" in plan.kept
        assert plan.total() == 0

def test_a_dead_runs_sandbox_is_reaped_while_its_snapshots_survive(tmp_path):
    """The asymmetry IS the policy: a dead run's sandbox burns compute right
    now; its snapshots are the branch-later capability the run existed to
    produce."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="sandbox", id="sb", pid=999999999)
    ledger._append("claim", run_id="dead", kind="snapshot", id="snap", pid=999999999)
    client = FakeClient(
        sandboxes=[{"id": "sb", "created_at": None, "names": []}],
        snapshots=[{"id": "snap", "created_at": None, "names": []}],
    )
    plan = Reaper(client, ledger).plan()
    assert plan.sandboxes == ["sb"] and plan.snapshots == []
    assert "snap" in plan.kept
    assert Reaper(client, ledger).apply(plan)["sandboxes"] == ["sb"]

def test_sqlite_contention_is_retryable():
    """opencode keeps sessions in SQLite; parallel lanes hit this immediately.
    It is contention in the agent's own store, not an agent outcome."""
    assert is_retryable("Unexpected error | database is locked") is True


def test_batch_isolates_opencode_state_per_task(tmp_path, monkeypatch):
    """Without a per-task XDG_DATA_HOME, concurrent opencode lanes deadlock on
    their shared session database."""
    from harness.slots.opencode import OpenCodeSlot

    runner = BatchRunner("opencode-cli", tmp_path / "out", workers=2)
    captured = {}

    def fake_run(self, spec, journal, mcp=None):
        captured["extra"] = dict(spec.extra)
        env = self.build_env(spec, mcp)
        captured["xdg"] = env.get("XDG_DATA_HOME")
        from harness.core.slot import SlotResult

        return SlotResult(status="completed")

    monkeypatch.setattr(OpenCodeSlot, "run", fake_run, raising=False)
    runner.run([{"id": "abc", "prompt": "p", "cwd": str(tmp_path)}])

    assert captured["extra"]["data_home"].endswith("state/abc")
    assert captured["xdg"] == captured["extra"]["data_home"]
    assert (tmp_path / "out" / "state" / "abc").is_dir()


# --- sweep: the one legitimate way a snapshot dies -----------------------------
def _journal_with_snapshots(path, ids):
    import json as _json

    lines = [{"v": 2, "type": "run.started", "seq": 1, "run_id": "r"}]
    for n, sid in enumerate(ids, 2):
        lines.append({"v": 2, "type": "checkpoint.captured", "seq": n,
                      "run_id": "r", "snapshot_id": sid})
    path.write_text("\n".join(_json.dumps(l) for l in lines) + "\n")


def test_sweep_is_a_dry_run_unless_told_otherwise(tmp_path, monkeypatch, capsys):
    """Deleting history must take two deliberate acts: naming the journal AND
    typing --yes. The default prints what WOULD die and touches nothing."""
    import json as _json

    import harness.cli as cli

    journal = tmp_path / "run.jsonl"
    _journal_with_snapshots(journal, ["s1", "s2"])
    deleted = []

    class FakeEnv:
        def delete_snapshot(self, sid):
            deleted.append(sid)
            return True

    monkeypatch.setattr(cli, "AgentEnvClient", FakeEnv)
    assert cli.main(["sweep", str(journal)]) == 0
    assert deleted == []
    assert "would delete 2" in capsys.readouterr().out
    # the journal is untouched -- no swept marker on a dry run
    assert "snapshots.swept" not in journal.read_text()

    assert cli.main(["sweep", str(journal), "--yes"]) == 0
    assert deleted == ["s1", "s2"]
    marker = [_json.loads(l) for l in journal.read_text().splitlines()][-1]
    assert marker["type"] == "snapshots.swept" and marker["deleted"] == 2


def test_sweep_refuses_to_double_sweep_and_missing_journals(tmp_path, monkeypatch, capsys):
    import harness.cli as cli

    journal = tmp_path / "run.jsonl"
    _journal_with_snapshots(journal, ["s1"])
    calls = []

    class FakeEnv:
        def delete_snapshot(self, sid):
            calls.append(sid)
            return True

    monkeypatch.setattr(cli, "AgentEnvClient", FakeEnv)
    assert cli.main(["sweep", str(journal), "--yes"]) == 0
    assert cli.main(["sweep", str(journal), "--yes"]) == 0  # 幂等
    assert calls == ["s1"], "a swept journal must not be swept again"
    assert cli.main(["sweep", str(tmp_path / "nope.jsonl")]) == 2


def test_nothing_in_the_run_lifecycle_calls_snapshot_deletion():
    """The design boundary in one test: deletion is never wired into teardown.
    The orchestrator, the checkpointing path and the session may CREATE
    snapshots; the only callers of deletion are the two explicit surfaces
    (harness sweep, and SDK users calling delete_snapshot themselves)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for module in ("harness/orchestrator/run.py", "harness/checkpointing.py",
                   "harness/execution/session.py", "harness/execution/server.py"):
        text = (root / module).read_text()
        assert "delete_snapshot" not in text, \
            "%s must not delete snapshots -- runs end, history stays" % module
