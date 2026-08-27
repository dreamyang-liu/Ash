"""Orchestrator: the sequence of one run, and what it guarantees.

These pin the two things the sequence exists for -- **ordering** (each step must
happen before the one that depends on it, and "too late" is silently wrong rather
than an error) and **teardown** (everything acquired is released, including when
the slot raises).

A fake slot stands in for an agent, so no model, sandbox or network is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from harness.core import events as E
from harness.core.journal import read_journal
from harness.core.slot import AgentSlot, SlotCapabilities, SlotResult
from harness.orchestrator import Orchestrator, RunSpec


# --- fakes -----------------------------------------------------------------
class RecordingSlot(AgentSlot):
    """Records what the orchestrator handed it, and emits a couple of events."""

    name = "fake"
    capabilities = SlotCapabilities()
    calls: List[dict] = []
    raises: Optional[Exception] = None

    def run(self, task, journal, mcp=None):
        RecordingSlot.calls.append({
            "prompt": task.prompt,
            "cwd": task.cwd,
            "model": task.model,
            "extra": dict(task.extra),
            "env": dict(task.env),
            "mcp": mcp,
        })
        if RecordingSlot.raises is not None:
            raise RecordingSlot.raises
        journal.emit(E.SESSION_REF, native_session_id="ses_fake")
        journal.emit(E.TURN_COMPLETED, usage={"input_tokens": 7})
        return SlotResult(status="completed", final_text="done",
                          usage={"input_tokens": 7}, native_session_id="ses_fake")


@pytest.fixture(autouse=True)
def fake_slot(monkeypatch):
    RecordingSlot.calls = []
    RecordingSlot.raises = None
    monkeypatch.setattr("harness.slots.load_slot", lambda name: RecordingSlot)
    yield RecordingSlot


@dataclass
class FakeProvisioned:
    sandbox_id: str = "sb-1"
    mcp: Any = "MCP_WIRING"
    destroyed: bool = False

    def destroy(self):
        self.destroyed = True


@dataclass
class FakeGateway:
    base_url: str = "http://127.0.0.1:9999"
    stopped: bool = False
    order: List[str] = field(default_factory=list)

    def start(self):
        return self

    def stop(self):
        self.stopped = True
        self.order.append("gateway.stop")

    def env_for(self, token):
        return {"ANTHROPIC_BASE_URL": self.base_url, "ANTHROPIC_AUTH_TOKEN": "tok"}


@dataclass
class FakeSession:
    """Snapshot-capable enough for the bridge to install."""

    snapshots: List[str] = field(default_factory=list)

    def supports_snapshot(self):
        return True

    def snapshot(self, name=None, disk_only=True):
        sid = "snap-%d" % (len(self.snapshots) + 1)
        self.snapshots.append(sid)
        return type("S", (), {"id": sid, "rootfs_layers": 1, "memory_layers": 0,
                              "chain_size_mb": 1, "disk_only": disk_only})()

    def swap_sandbox(self, snapshot):
        return True

    def squash_snapshot(self, snapshot, name=None):
        return snapshot


def spec(tmp_path, **kwargs) -> RunSpec:
    defaults = dict(prompt="do it", slot="fake", cwd=str(tmp_path),
                    journal_path=tmp_path / "j.jsonl")
    defaults.update(kwargs)
    return RunSpec(**defaults)


# --- the happy path --------------------------------------------------------
def test_a_bare_run_needs_no_sandbox_or_gateway(tmp_path):
    outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path))
    assert outcome.ok and outcome.final_text == "done"
    assert outcome.usage == {"input_tokens": 7}
    assert outcome.native_session_id == "ses_fake"
    assert RecordingSlot.calls[0]["mcp"] is None


def test_the_journal_records_the_run_even_though_the_outcome_is_returned(tmp_path):
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path))
    types = [r["type"] for r in read_journal(tmp_path / "j.jsonl")]
    assert E.SESSION_REF in types and E.TURN_COMPLETED in types


# --- ordering --------------------------------------------------------------
def test_the_gateway_env_reaches_the_slot(tmp_path, monkeypatch):
    """The agent reads base_url when it starts, so the gateway must be up and its
    env merged before the slot runs -- later is silently wrong, not an error."""
    gateway = FakeGateway()
    monkeypatch.setattr("harness.gateway.GatewayServer", lambda *a, **k: gateway)
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path, use_gateway=True))

    env = RecordingSlot.calls[0]["env"]
    assert env["ANTHROPIC_BASE_URL"] == gateway.base_url
    assert env["ANTHROPIC_AUTH_TOKEN"]


def test_the_bound_wiring_reaches_the_slot(tmp_path, monkeypatch):
    provisioned = FakeProvisioned()
    monkeypatch.setattr("harness.execution.provision.provision_http",
                        lambda *a, **k: provisioned)
    outcome = Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, mcp_url="http://x/mcp", sandbox_image="img"))

    assert RecordingSlot.calls[0]["mcp"] == "MCP_WIRING"
    assert outcome.sandbox_id == "sb-1"


def test_checkpoints_are_paired_when_a_session_is_given(tmp_path):
    """The bridge must be subscribed before the slot's first event, or the run's
    early checkpoints are silently lost."""
    session = FakeSession()
    outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, session=session))

    assert outcome.checkpoints == 1, "the turn boundary did not produce a pair"
    marks = [r for r in read_journal(tmp_path / "j.jsonl")
             if r["type"] == E.CHECKPOINT_CAPTURED]
    assert marks and marks[0]["session_ckpt"] == "ses_fake"


def test_no_session_means_no_checkpointing(tmp_path):
    outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path))
    assert outcome.checkpoints == 0


# --- teardown --------------------------------------------------------------
def test_the_sandbox_is_destroyed_on_success(tmp_path, monkeypatch):
    provisioned = FakeProvisioned()
    monkeypatch.setattr("harness.execution.provision.provision_http",
                        lambda *a, **k: provisioned)
    Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, mcp_url="http://x/mcp", sandbox_image="img"))
    assert provisioned.destroyed


def test_everything_is_released_when_the_slot_raises(tmp_path, monkeypatch):
    """A crashing agent must not strand a sandbox or a gateway."""
    provisioned = FakeProvisioned()
    gateway = FakeGateway()
    monkeypatch.setattr("harness.execution.provision.provision_http",
                        lambda *a, **k: provisioned)
    monkeypatch.setattr("harness.gateway.GatewayServer", lambda *a, **k: gateway)
    RecordingSlot.raises = RuntimeError("agent exploded")

    outcome = Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, mcp_url="http://x/mcp", sandbox_image="img", use_gateway=True))

    assert not outcome.ok and "agent exploded" in outcome.error
    assert provisioned.destroyed and gateway.stopped
    # and the failure is in the journal, not only in the return value
    assert any(r.get("status") == "error" for r in read_journal(tmp_path / "j.jsonl"))


def test_keep_sandbox_survives_the_run(tmp_path, monkeypatch):
    """Grading and post-hoc extraction need the sandbox alive after the agent
    stops."""
    provisioned = FakeProvisioned()
    monkeypatch.setattr("harness.execution.provision.provision_http",
                        lambda *a, **k: provisioned)
    Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, mcp_url="http://x/mcp", sandbox_image="img", keep_sandbox=True))
    assert not provisioned.destroyed


def test_a_failing_teardown_step_does_not_block_the_others(tmp_path, monkeypatch):
    class Stubborn(FakeProvisioned):
        def destroy(self):
            raise RuntimeError("backend unreachable")

    gateway = FakeGateway()
    monkeypatch.setattr("harness.execution.provision.provision_http",
                        lambda *a, **k: Stubborn())
    monkeypatch.setattr("harness.gateway.GatewayServer", lambda *a, **k: gateway)

    outcome = Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, mcp_url="http://x/mcp", sandbox_image="img", use_gateway=True))
    assert outcome.ok                 # the run itself succeeded
    assert gateway.stopped            # stopped before the sandbox failed to die


# --- the resource ledger ---------------------------------------------------
def test_resources_are_claimed_before_use_and_released_after(tmp_path, monkeypatch):
    """A killed *process* releases nothing, so `harness reap` reads the ledger --
    which means the claim has to be written before the run touches the sandbox."""
    from harness.orchestrator import ResourceLedger

    provisioned = FakeProvisioned()
    monkeypatch.setattr("harness.execution.provision.provision_http",
                        lambda *a, **k: provisioned)
    ledger = ResourceLedger(tmp_path / "ledger.jsonl")

    Orchestrator(out_dir=tmp_path, ledger=ledger).run(
        spec(tmp_path, run_id="r1", mcp_url="http://x/mcp", sandbox_image="img"))

    entries = list(ledger.entries())
    kinds = [(e.get("event"), e.get("kind"), e.get("id")) for e in entries]
    assert ("claim", "sandbox", "sb-1") in kinds
    assert ("release", "sandbox", "sb-1") in kinds


def test_snapshots_are_claimed_from_the_journal(tmp_path):
    """Claiming them from checkpoint events means a capture is reclaimable even if
    the process dies immediately after taking it."""
    from harness.orchestrator import ResourceLedger

    ledger = ResourceLedger(tmp_path / "ledger.jsonl")
    Orchestrator(out_dir=tmp_path, ledger=ledger).run(
        spec(tmp_path, run_id="r2", session=FakeSession()))

    claimed = [(e.get("kind"), e.get("id")) for e in ledger.entries()
               if e.get("event") == "claim"]
    assert ("snapshot", "snap-1") in claimed


# --- per-slot hygiene ------------------------------------------------------
def test_claude_code_ignores_the_developers_local_config(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.slots.load_slot", lambda name: RecordingSlot)
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path, slot="claude-code"))
    assert RecordingSlot.calls[0]["extra"]["setting_sources"] == []


def test_opencode_gets_an_isolated_session_store(tmp_path):
    """Concurrent opencode lanes sharing the SQLite db fail with
    "database is locked"."""
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path, slot="opencode", run_id="t1"))
    data_home = RecordingSlot.calls[0]["extra"]["data_home"]
    assert data_home.endswith("state/t1")


def test_resume_and_fork_are_passed_through(tmp_path):
    Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, resume_session_id="ses_old", fork=True))
    extra = RecordingSlot.calls[0]["extra"]
    assert extra["resume_session_id"] == "ses_old" and extra["fork"] is True


def test_caller_supplied_extra_wins_over_the_defaults(tmp_path):
    Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, slot="claude-code",
             extra={"setting_sources": ["project"]}))
    assert RecordingSlot.calls[0]["extra"]["setting_sources"] == ["project"]
