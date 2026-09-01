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


# --- naming a tool panel ---------------------------------------------------
def test_stdio_wiring_carries_the_named_panel():
    """The orchestrator starts this server, so it can say what to serve."""
    from harness.orchestrator.run import Orchestrator, RunSpec

    _, mcp = Orchestrator()._wire_sandbox(
        RunSpec(prompt="x", mcp_stdio_args=["--attach", "sb-1"], tools="default"),
        None)
    assert mcp.command[-2:] == ["--tools", "default"]


def test_an_explicit_tools_arg_is_not_duplicated():
    """A caller that already spelled it out wins; two --tools flags would leave
    argparse silently keeping the last, which is not obviously the intended one."""
    from harness.orchestrator.run import Orchestrator, RunSpec

    _, mcp = Orchestrator()._wire_sandbox(
        RunSpec(prompt="x", mcp_stdio_args=["--tools", "bash_only"], tools="default"),
        None)
    assert mcp.command.count("--tools") == 1
    assert "bash_only" in mcp.command


def test_naming_a_panel_over_http_is_refused_not_ignored():
    """The panel is compiled by the server process, and over HTTP that process is
    already running with whatever --tools it was given. Accepting this quietly
    would have the run report a tool surface it did not have."""
    import pytest

    from harness.orchestrator.run import Orchestrator, RunSpec

    with pytest.raises(ValueError) as exc:
        Orchestrator()._wire_sandbox(
            RunSpec(prompt="x", mcp_url="http://h:8400", tools="default"), None)
    assert "stdio-only" in str(exc.value)


def test_no_panel_named_leaves_the_wiring_alone():
    from harness.orchestrator.run import Orchestrator, RunSpec

    _, mcp = Orchestrator()._wire_sandbox(
        RunSpec(prompt="x", mcp_stdio_args=["--attach", "sb-1"]), None)
    assert "--tools" not in mcp.command


# --- owning the sandbox ----------------------------------------------------
#
# The orchestrator used to require its caller to build a sandbox and a session
# and pass them in, which is the opposite of owning a run: a caller that merely
# named an image got no environment half of a rollback pair, because nothing here
# held a handle to snapshot through. These pin the ownership, both transports, and
# the teardown order.

class _FakeSession:
    """A SandboxSession-shaped object: create, snapshot, destroy."""

    def __init__(self, sandbox_id="sb-owned", create_ok=True):
        self._id = sandbox_id
        self.create_ok = create_ok
        self.create_error = "backend said no" if not create_ok else ""
        self.created = None
        self.destroyed = False
        self.order = []
        self.sandbox = object()

    @property
    def sandbox_id(self):
        # Mirrors the real one: a destroyed session cannot name itself.
        return "unknown" if self.destroyed else self._id

    def create(self, image, resources=None):
        self.created = image
        self.order.append("create")
        return self.create_ok

    def supports_snapshot(self):
        return False

    def destroy(self):
        self.order.append("destroy")
        self.destroyed = True


def _own(monkeypatch, session, **spec_kwargs):
    """Drive _own_sandbox with a fake session, returning (orchestrator, owned)."""
    import harness.execution.session as session_module

    from harness.orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    spec = RunSpec(prompt="x", sandbox_image="img", **spec_kwargs)
    orch = Orchestrator()
    return orch, orch._own_sandbox(spec, None), spec


def test_naming_an_image_makes_the_orchestrator_create_the_sandbox(monkeypatch):
    session = _FakeSession()
    _, owned, _ = _own(monkeypatch, session)
    assert session.created == "img"
    assert owned.session is session


def test_stdio_transport_lends_the_sandbox_by_id(monkeypatch):
    """--attach, not --image: the subprocess serves calls into a sandbox this
    process owns, so the handle needed for snapshots stays here."""
    session = _FakeSession()
    _, owned, _ = _own(monkeypatch, session, transport="stdio", tools="default")
    command = owned.mcp.command
    assert "--attach" in command and "sb-owned" in command
    assert "--image" not in command
    assert command[command.index("--tools") + 1] == "default"
    assert owned.server is None, "stdio's server is the slot's own subprocess"


def test_http_transport_serves_the_sandbox_in_process(monkeypatch):
    """In-process because the session lives here. An out-of-process server would
    have to create its own sandbox, and the owner could then not snapshot the
    environment its agent actually worked in."""
    session = _FakeSession()
    _, owned, spec = _own(monkeypatch, session, transport="http", tools="default")
    try:
        assert owned.server is not None
        assert owned.mcp.url.startswith("http://127.0.0.1:")
        assert owned.mcp.url.endswith("/mcp")
        # The sandbox is bound by header, so the model is served the
        # single-sandbox schema and never sees a sandbox_id argument.
        headers = {k.lower(): v for k, v in (owned.mcp.headers or {}).items()}
        assert headers.get("x-session-sandbox") == "sb-owned"
    finally:
        owned.release()


def test_the_in_process_pool_does_not_own_what_it_was_handed(monkeypatch):
    """adopt, not attach: the entry is marked external, so stopping the server
    releases the sandbox instead of destroying the environment its owner still
    needs for grading and extraction."""
    session = _FakeSession()
    _, owned, _ = _own(monkeypatch, session, transport="http")
    try:
        entry = owned.server.pool.get("sb-owned")
        assert entry is not None and entry.external is True
    finally:
        owned.release()
    assert session.destroyed, "the owner destroys it -- once, and here"


def test_an_unknown_transport_is_refused(monkeypatch):
    import pytest

    from harness.orchestrator.run import Orchestrator, RunSpec

    with pytest.raises(ValueError, match="unknown transport"):
        Orchestrator()._own_sandbox(
            RunSpec(prompt="x", sandbox_image="img", transport="carrier-pigeon"),
            None)


def test_a_failed_create_does_not_leave_a_half_wired_run(monkeypatch):
    import pytest

    session = _FakeSession(create_ok=False)
    with pytest.raises(RuntimeError, match="could not create a sandbox") as exc:
        _own(monkeypatch, session)
    # The reason travels with it: the session is quiet, so its own warning went
    # nowhere, and "could not create a sandbox from <image>" alone points at the
    # image when the cause is usually a missing setting.
    assert "backend said no" in str(exc.value)


def test_a_sandbox_is_not_leaked_when_wiring_fails(monkeypatch):
    """The sandbox exists but nothing can reach it. Raising without releasing
    would leave it running with the caller seeing only an exception."""
    import pytest

    import harness.orchestrator.run as run_module

    session = _FakeSession()
    monkeypatch.setattr("harness.execution.session.SandboxSession",
                        lambda **kw: session)
    monkeypatch.setattr(run_module.Orchestrator, "_serve_in_process",
                        lambda self, spec, owned: (_ for _ in ()).throw(
                            RuntimeError("port in use")))
    with pytest.raises(RuntimeError, match="port in use"):
        run_module.Orchestrator()._own_sandbox(
            run_module.RunSpec(prompt="x", sandbox_image="img",
                               transport="http"), None)
    assert session.destroyed, "a sandbox nothing can reach must not be left running"


def test_owning_the_sandbox_gives_checkpoints_without_the_caller_asking(monkeypatch):
    """The payoff. Rollback pairs used to need the caller to build a session and
    hand it down; a run that named an image got no environment half at all."""
    from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec

    installed = []

    class FakeBridge:
        ledger = type("L", (), {"checkpoints": []})()

    monkeypatch.setattr("harness.checkpointing.SnapshotBridge.install",
                        classmethod(lambda cls, journal, session, always=False,
                                    tracker=None:
                                    installed.append(session) or FakeBridge()))
    session = _FakeSession()
    owned = OwnedSandbox(session=session, sandbox_id="sb-owned")
    Orchestrator()._wire_checkpoints(RunSpec(prompt="x"), object(), owned)
    assert installed == [session]


def test_a_remote_sandbox_has_no_session_to_snapshot(monkeypatch):
    """Not an error: a sandbox on somebody else's server cannot be snapshotted
    from here, which is precisely the limitation owning one removes."""
    from harness.orchestrator.run import Orchestrator, RunSpec

    class Remote:
        sandbox_id = "sb-remote"          # no `session` attribute

    assert Orchestrator()._wire_checkpoints(
        RunSpec(prompt="x"), object(), Remote()) is None


def test_the_outcome_can_still_name_its_sandbox_after_teardown():
    """Teardown runs before the outcome is assembled, and a destroyed session
    answers "unknown" -- so reading the id lazily left every completed run unable
    to say which sandbox produced it."""
    from harness.orchestrator.run import OwnedSandbox

    session = _FakeSession()
    owned = OwnedSandbox(session=session, sandbox_id=session.sandbox_id)
    owned.release()
    assert session.destroyed
    assert owned.sandbox_id == "sb-owned"


def test_the_server_stops_even_when_the_sandbox_is_kept(monkeypatch):
    """--keep leaves the sandbox for grading. A server still serving it is a
    process nobody will stop, holding a port nobody will reuse."""
    session = _FakeSession()
    _, owned, _ = _own(monkeypatch, session, transport="http", keep_sandbox=True)
    server = owned.server
    owned.release()
    assert owned.server is None
    assert not session.destroyed, "--keep means the sandbox survives"


# --- checkpoints at the tool boundary: one tracker, one trigger ---------------
def test_http_mounts_the_tracker_on_the_serving_pipeline(monkeypatch):
    """The interceptor half of checkpointing must watch the pipeline that serves
    the calls. The bridge used to build its own tracker, which nothing fed --
    dirty stayed False, and every step after the first was recorded as clean
    reuse of snapshot 1 while the agent was writing files."""
    from harness.execution.interceptors import MutationTracker

    session = _FakeSession()
    _, owned, _ = _own(monkeypatch, session, transport="http")
    try:
        assert isinstance(owned.tracker, MutationTracker)
        mounted = owned.server.pipeline.interceptors
        assert owned.tracker in mounted, "the tracker must sit on the serving pipeline"
    finally:
        owned.release()


def test_the_bridge_reads_the_same_tracker_the_pipeline_feeds(monkeypatch):
    from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec

    seen = {}

    class FakeBridge:
        ledger = type("L", (), {"checkpoints": []})()

        def on_tool_boundary(self, index):
            pass

    def fake_install(cls, journal, session, always=False, tracker=None):
        seen["tracker"] = tracker
        return FakeBridge()

    monkeypatch.setattr("harness.checkpointing.SnapshotBridge.install",
                        classmethod(fake_install))
    tracker = object()
    owned = OwnedSandbox(session=_FakeSession(), tracker=tracker, sandbox_id="sb")
    Orchestrator()._wire_checkpoints(RunSpec(prompt="x"), object(), owned)
    assert seen["tracker"] is tracker, \
        "two trackers is the unfed-tracker bug in miniature"


def test_the_boundary_lands_on_the_server_after_the_bridge_exists(monkeypatch):
    from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec

    class FakeBridge:
        ledger = type("L", (), {"checkpoints": []})()

        def on_tool_boundary(self, index):
            self.fired = index

    bridge = FakeBridge()
    monkeypatch.setattr("harness.checkpointing.SnapshotBridge.install",
                        classmethod(lambda cls, journal, session, always=False,
                                    tracker=None: bridge))

    class FakeServer:
        boundary = None

    server = FakeServer()
    owned = OwnedSandbox(session=_FakeSession(), server=server, sandbox_id="sb")
    Orchestrator()._wire_checkpoints(RunSpec(prompt="x"), object(), owned)
    assert server.boundary is not None
    import asyncio

    asyncio.run(server.boundary.after_call())
    assert bridge.fired == 1, "the trigger must reach the bridge's checkpointer"


# --- saying so when checkpointing produced nothing --------------------------
class _RecordingJournal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **payload):
        self.events.append((kind, payload))


def test_a_run_that_recorded_no_checkpoints_says_so():
    """Silence about a capability that did not happen is worse than the absence:
    it is discovered later, from the outside, by someone trying to branch a run
    that cannot be branched."""
    from harness.orchestrator.run import Orchestrator, RunSpec

    class Bridge:
        skipped_on_loop = 4
        ledger = type("L", (), {"checkpoints": []})()

    seen = []
    journal = _RecordingJournal()
    orch = Orchestrator(on_event=lambda k, p: seen.append((k, p)))
    orch._report_missing_checkpoints(
        RunSpec(prompt="x", transport="stdio"), journal, Bridge())

    kinds = [k for k, _ in journal.events]
    assert "checkpoint.unavailable" in kinds
    payload = dict(journal.events[0][1])
    assert payload["skipped"] == 4 and payload["transport"] == "stdio"
    assert "http" in payload["reason"], "the message must name the remedy"
    assert seen and seen[0][0] == "checkpoint.unavailable"


def test_a_run_that_did_checkpoint_stays_quiet():
    from harness.orchestrator.run import Orchestrator, RunSpec

    class Bridge:
        skipped_on_loop = 3          # some skips, but snapshots were taken anyway
        ledger = type("L", (), {"checkpoints": [object()]})()

    journal = _RecordingJournal()
    Orchestrator()._report_missing_checkpoints(RunSpec(prompt="x"), journal, Bridge())
    assert journal.events == []


def test_no_checkpointing_requested_is_not_a_warning():
    """Nothing was asked for, so nothing is missing."""
    from harness.orchestrator.run import Orchestrator, RunSpec

    journal = _RecordingJournal()
    Orchestrator()._report_missing_checkpoints(RunSpec(prompt="x"), journal, None)
    assert journal.events == []


# --- stdio: the shutter is pressed in the subprocess -------------------------
#
# The checkpoint machinery sits at the tool path, in whichever process serves the
# calls. Over stdio that is the server subprocess: it sees every tool boundary
# (its loop is strictly sequential, so the hook always runs quiesced) and it has
# its own handle to the sandbox (attach). What travels back to this process is
# only the step->snapshot map, one JSON line per capture.

def test_the_stdio_command_asks_the_subprocess_to_checkpoint(monkeypatch, tmp_path):
    session = _FakeSession()
    session.supports_snapshot = lambda: True
    import harness.execution.session as session_module

    from harness.orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    orch = Orchestrator(out_dir=tmp_path)
    owned = orch._own_sandbox(
        RunSpec(prompt="x", sandbox_image="img", transport="stdio"), None)
    command = owned.mcp.command
    assert "--checkpoint-log" in command
    assert owned.checkpoint_log is not None
    assert str(owned.checkpoint_log) == command[command.index("--checkpoint-log") + 1]
    assert "--checkpoint-always" not in command


def test_snapshot_every_step_reaches_the_subprocess(monkeypatch, tmp_path):
    session = _FakeSession()
    session.supports_snapshot = lambda: True
    import harness.execution.session as session_module

    from harness.orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    owned = Orchestrator(out_dir=tmp_path)._own_sandbox(
        RunSpec(prompt="x", sandbox_image="img", transport="stdio",
                snapshot_every_step=True), None)
    assert "--checkpoint-always" in owned.mcp.command


def test_a_backend_that_cannot_snapshot_is_not_asked_to(monkeypatch, tmp_path):
    """Docker: the flag would only produce a one-line complaint per run."""
    session = _FakeSession()          # supports_snapshot() is False
    import harness.execution.session as session_module

    from harness.orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    owned = Orchestrator(out_dir=tmp_path)._own_sandbox(
        RunSpec(prompt="x", sandbox_image="img", transport="stdio"), None)
    assert "--checkpoint-log" not in owned.mcp.command
    assert owned.checkpoint_log is None


class _TailBridge:
    """Records record_pair calls; enough bridge for the tailer."""

    def __init__(self):
        self.recorded = []

    def record_pair(self, step, snapshot_id, *, captured=True, reason="",
                    **extra):
        self.recorded.append({"step": step, "snapshot_id": snapshot_id,
                              "captured": captured, "reason": reason})


def test_the_tail_folds_lines_while_the_run_is_still_going(tmp_path):
    """The point of tailing instead of folding at the end: a pair lands in the
    journal when it happens. On a marathon-length run, "the map exists only
    after a clean finish" is the 300-snapshots-no-record failure wearing a
    different hat."""
    import time

    from harness.orchestrator.run import CheckpointTail

    log = tmp_path / "map.jsonl"
    bridge = _TailBridge()
    tail = CheckpointTail(log, bridge, poll_s=0.05).start()
    try:
        # The file does not even exist yet -- the subprocess creates it on the
        # first capture. The tailer must cope, then pick the line up.
        time.sleep(0.1)
        assert bridge.recorded == []
        with open(log, "a") as fh:
            fh.write('{"step": 1, "snapshot_id": "snap-a", "captured": true, '
                     '"reason": "mutated"}\n')
        deadline = time.time() + 2
        while not bridge.recorded and time.time() < deadline:
            time.sleep(0.02)
        assert [r["snapshot_id"] for r in bridge.recorded] == ["snap-a"], \
            "the pair must land during the run, not at stop()"
    finally:
        tail.stop()


def test_stop_drains_what_landed_after_the_last_poll(tmp_path):
    """A capture younger than one poll interval must not be lost: the
    missing-checkpoint report reads the ledger right after stop() returns."""
    from harness.orchestrator.run import CheckpointTail

    log = tmp_path / "map.jsonl"
    bridge = _TailBridge()
    tail = CheckpointTail(log, bridge, poll_s=3600).start()   # never polls
    log.write_text('{"step": 1, "snapshot_id": "snap-a"}\n')
    tail.stop()
    assert [r["snapshot_id"] for r in bridge.recorded] == ["snap-a"]


def test_a_torn_last_line_from_a_killed_subprocess_is_skipped(tmp_path):
    """The whole lines before it fold; the line the kill cut mid-write does not
    -- and a line that is merely *late* (no newline yet) is not misread as torn:
    it stays buffered until its newline arrives."""
    from harness.orchestrator.run import CheckpointTail

    log = tmp_path / "map.jsonl"
    bridge = _TailBridge()
    tail = CheckpointTail(log, bridge, poll_s=3600)
    log.write_text('{"step": 1, "snapshot_id": "snap-a"}\n{"step": 2, "snap')
    tail.drain()
    assert [r["step"] for r in bridge.recorded] == [1]
    # the writer finishes the line: it must fold now, not be lost
    with open(log, "a") as fh:
        fh.write('shot_id": "snap-b"}\n')
    tail.drain()
    assert [r["snapshot_id"] for r in bridge.recorded] == ["snap-a", "snap-b"]


def test_no_file_and_stop_before_start_are_harmless(tmp_path):
    from harness.orchestrator.run import CheckpointTail

    tail = CheckpointTail(tmp_path / "never-written.jsonl", _TailBridge())
    tail.drain()        # no file: nothing
    tail.stop()         # never started: no crash


def test_a_pair_tailed_before_the_session_ref_is_backfilled(tmp_path):
    """The tailer may fold a pair before the slot has disclosed its session id.
    record_pair reuses the bridge's backfill: when session.ref arrives, the
    half-pair is corrected retroactively instead of staying half forever."""
    from harness.checkpointing import SnapshotBridge
    from harness.core.journal import JournalWriter, read_journal

    class NoSnapSession:
        def supports_snapshot(self):
            return False

    journal_path = tmp_path / "j.jsonl"
    with JournalWriter(journal_path, run_id="r1", agent_id="a") as journal:
        bridge = SnapshotBridge.install(journal, NoSnapSession())
        bridge.record_pair(1, "snap-a", captured=True, reason="mutated")
        assert bridge.ledger.checkpoints[0].session_ckpt is None
        journal.emit("session.ref", native_session_id="conv-42")

    records = [r for r in read_journal(journal_path)
               if r["type"] == "checkpoint.captured"]
    assert records[0]["session_ckpt"] is None, "recorded before the ref existed"
    backfills = [r for r in records if r.get("reason") == "session_ref_backfill"]
    assert backfills and backfills[0]["session_ckpt"] == "conv-42"
    assert backfills[0]["snapshot_id"] == "snap-a"


# --- checkpoints at the tool boundary: one tracker, one trigger ---------------
def test_http_mounts_the_tracker_on_the_serving_pipeline(monkeypatch):
    """The interceptor half of checkpointing must watch the pipeline that serves
    the calls. The bridge used to build its own tracker, which nothing fed --
    dirty stayed False, and every step after the first was recorded as clean
    reuse of snapshot 1 while the agent was writing files."""
    from harness.execution.interceptors import MutationTracker

    session = _FakeSession()
    _, owned, _ = _own(monkeypatch, session, transport="http")
    try:
        assert isinstance(owned.tracker, MutationTracker)
        mounted = owned.server.pipeline.interceptors
        assert owned.tracker in mounted, "the tracker must sit on the serving pipeline"
    finally:
        owned.release()


def test_the_bridge_reads_the_same_tracker_the_pipeline_feeds(monkeypatch):
    from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec

    seen = {}

    class FakeBridge:
        ledger = type("L", (), {"checkpoints": []})()

        def on_tool_boundary(self, index):
            pass

    def fake_install(cls, journal, session, always=False, tracker=None):
        seen["tracker"] = tracker
        return FakeBridge()

    monkeypatch.setattr("harness.checkpointing.SnapshotBridge.install",
                        classmethod(fake_install))
    tracker = object()
    owned = OwnedSandbox(session=_FakeSession(), tracker=tracker, sandbox_id="sb")
    Orchestrator()._wire_checkpoints(RunSpec(prompt="x"), object(), owned)
    assert seen["tracker"] is tracker, \
        "two trackers is the unfed-tracker bug in miniature"


def test_the_boundary_lands_on_the_server_after_the_bridge_exists(monkeypatch):
    from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec

    class FakeBridge:
        ledger = type("L", (), {"checkpoints": []})()

        def on_tool_boundary(self, index):
            self.fired = index

    bridge = FakeBridge()
    monkeypatch.setattr("harness.checkpointing.SnapshotBridge.install",
                        classmethod(lambda cls, journal, session, always=False,
                                    tracker=None: bridge))

    class FakeServer:
        boundary = None

    server = FakeServer()
    owned = OwnedSandbox(session=_FakeSession(), server=server, sandbox_id="sb")
    Orchestrator()._wire_checkpoints(RunSpec(prompt="x"), object(), owned)
    assert server.boundary is not None
    import asyncio

    asyncio.run(server.boundary.after_call())
    assert bridge.fired == 1, "the trigger must reach the bridge's checkpointer"


# --- saying so when checkpointing produced nothing --------------------------
class _RecordingJournal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **payload):
        self.events.append((kind, payload))


def test_a_run_that_recorded_no_checkpoints_says_so():
    """Silence about a capability that did not happen is worse than the absence:
    it is discovered later, from the outside, by someone trying to branch a run
    that cannot be branched."""
    from harness.orchestrator.run import Orchestrator, RunSpec

    class Bridge:
        skipped_on_loop = 4
        ledger = type("L", (), {"checkpoints": []})()

    seen = []
    journal = _RecordingJournal()
    orch = Orchestrator(on_event=lambda k, p: seen.append((k, p)))
    orch._report_missing_checkpoints(
        RunSpec(prompt="x", transport="stdio"), journal, Bridge())

    kinds = [k for k, _ in journal.events]
    assert "checkpoint.unavailable" in kinds
    payload = dict(journal.events[0][1])
    assert payload["skipped"] == 4 and payload["transport"] == "stdio"
    assert "http" in payload["reason"], "the message must name the remedy"
    assert seen and seen[0][0] == "checkpoint.unavailable"


def test_a_run_that_did_checkpoint_stays_quiet():
    from harness.orchestrator.run import Orchestrator, RunSpec

    class Bridge:
        skipped_on_loop = 3          # some skips, but snapshots were taken anyway
        ledger = type("L", (), {"checkpoints": [object()]})()

    journal = _RecordingJournal()
    Orchestrator()._report_missing_checkpoints(RunSpec(prompt="x"), journal, Bridge())
    assert journal.events == []


def test_no_checkpointing_requested_is_not_a_warning():
    """Nothing was asked for, so nothing is missing."""
    from harness.orchestrator.run import Orchestrator, RunSpec

    journal = _RecordingJournal()
    Orchestrator()._report_missing_checkpoints(RunSpec(prompt="x"), journal, None)
    assert journal.events == []


# --- stdio: the shutter is pressed in the subprocess -------------------------
#
# The checkpoint machinery sits at the tool path, in whichever process serves the
# calls. Over stdio that is the server subprocess: it sees every tool boundary
# (its loop is strictly sequential, so the hook always runs quiesced) and it has
# its own handle to the sandbox (attach). What travels back to this process is
# only the step->snapshot map, one JSON line per capture.

def test_the_stdio_command_asks_the_subprocess_to_checkpoint(monkeypatch, tmp_path):
    session = _FakeSession()
    session.supports_snapshot = lambda: True
    import harness.execution.session as session_module

    from harness.orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    orch = Orchestrator(out_dir=tmp_path)
    owned = orch._own_sandbox(
        RunSpec(prompt="x", sandbox_image="img", transport="stdio"), None)
    command = owned.mcp.command
    assert "--checkpoint-log" in command
    assert owned.checkpoint_log is not None
    assert str(owned.checkpoint_log) == command[command.index("--checkpoint-log") + 1]
    assert "--checkpoint-always" not in command


def test_snapshot_every_step_reaches_the_subprocess(monkeypatch, tmp_path):
    session = _FakeSession()
    session.supports_snapshot = lambda: True
    import harness.execution.session as session_module

    from harness.orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    owned = Orchestrator(out_dir=tmp_path)._own_sandbox(
        RunSpec(prompt="x", sandbox_image="img", transport="stdio",
                snapshot_every_step=True), None)
    assert "--checkpoint-always" in owned.mcp.command


def test_a_backend_that_cannot_snapshot_is_not_asked_to(monkeypatch, tmp_path):
    """Docker: the flag would only produce a one-line complaint per run."""
    session = _FakeSession()          # supports_snapshot() is False
    import harness.execution.session as session_module

    from harness.orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    owned = Orchestrator(out_dir=tmp_path)._own_sandbox(
        RunSpec(prompt="x", sandbox_image="img", transport="stdio"), None)
    assert "--checkpoint-log" not in owned.mcp.command
    assert owned.checkpoint_log is None


# --- lineage -----------------------------------------------------------------
def test_a_forked_runs_journal_opens_with_its_origin(tmp_path):
    """First event, before anything else: whatever else the journal ends up
    holding -- including nothing, for a run killed immediately -- it says where
    the run came from."""
    Orchestrator(out_dir=tmp_path).run(spec(
        tmp_path,
        origin={"parent_run_id": "p1", "branch_step": 2,
                "snapshot_id": "snap-x", "direction": "try Y"}))
    records = list(read_journal(tmp_path / "j.jsonl"))
    assert records[0]["type"] == "fork.origin"
    assert records[0]["parent_run_id"] == "p1"
    assert records[0]["branch_step"] == 2


def test_a_run_with_no_origin_emits_none(tmp_path):
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path))
    assert all(r["type"] != "fork.origin"
               for r in read_journal(tmp_path / "j.jsonl"))


# --- journals must survive a reboot ------------------------------------------
def test_volatile_paths_are_named_and_persistent_ones_are_not():
    """A journal is the ONLY record a run leaves -- snapshot ids, every step, the
    grading evidence. A 32-instance batch's journals lived in /tmp when the host
    rebooted mid-regrade; hours of agent time now exist only as prose. The guard
    is a resolved-prefix check at the ENTRY POINTS only: the library layer stays
    unguarded because tests legitimately journal into pytest's /tmp fixtures."""
    from harness.core.journal import volatile_reason

    for doomed in ("/tmp/v32", "/tmp", "/var/tmp/x/deep/run.jsonl",
                   "/dev/shm/j.jsonl", "/tmp/../tmp/sneaky"):
        assert volatile_reason(doomed), doomed
    for safe in ("runs/fork-eval", "/home/user/runs", "/tmpdir/runs",
                 "/data/tmp-results"):
        assert volatile_reason(safe) is None, safe


def test_the_run_entrypoint_refuses_a_volatile_journal():
    import subprocess, sys

    proc = subprocess.run(
        [sys.executable, "-m", "harness", "run", "--slot", "claude-code",
         "--journal", "/tmp/doomed.jsonl", "prompt"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "refusing" in proc.stderr and "--volatile-ok" in proc.stderr


def test_fork_eval_refuses_a_volatile_out_dir_but_not_for_regrade():
    """Regrade only READS journals; they are wherever the original run put them,
    and refusing to read from /tmp would strand exactly the data most in need of
    rescue."""
    import subprocess, sys

    proc = subprocess.run(
        [sys.executable, "-m", "swebench.fork_eval", "--instance", "x",
         "-o", "/tmp/doomed"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "refusing" in proc.stderr

    # --regrade on an empty /tmp dir: must get PAST the guard (it fails later,
    # on the empty directory, or succeeds trivially -- either way, no "refusing").
    proc = subprocess.run(
        [sys.executable, "-m", "swebench.fork_eval", "--regrade",
         "-o", "/tmp/definitely-empty-regrade-dir"],
        capture_output=True, text=True)
    assert "refusing" not in proc.stderr
