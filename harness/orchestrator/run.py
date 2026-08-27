"""One run, start to finish.

This is the code the CLI used to hold inline: provision a sandbox, mount a
gateway, install the checkpoint bridge, drive the slot, and tear all of it down
whichever way the run ends. It lives here because it is the *shape* of a run
rather than a way of invoking one -- ``harness run``, ``harness batch``, and
anything that later spawns a subagent all need the same sequence, and a copy in
each is a copy that drifts.

What the orchestrator owns (and a slot deliberately does not):

**Ordering.** The sandbox exists before the agent can name it, the gateway exists
before the agent reads its ``base_url``, and the checkpoint bridge is subscribed
before the first event it must pair. Each of those is a "too late is silently
wrong" rather than an error.

**Teardown.** Every resource acquired here is released here, in reverse, in a
``finally`` -- including when the slot raises. This is best-effort by nature: a
killed *process* releases nothing, which is why acquisitions are also written to
the resource ledger for ``harness reap`` to reclaim. Guaranteed cleanup lives in
two places on purpose, and neither is sufficient alone.

**Accounting.** The run's status, usage and resources go to the journal, not to a
return value only: a run that dies still leaves a record.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from harness.core.journal import JournalWriter, new_run_id
from harness.core.slot import McpWiring, SlotResult, TaskSpec
from harness.orchestrator.resources import ResourceLedger


@dataclass
class RunSpec:
    """Everything one run needs. Mirrors ``harness run``'s flags on purpose."""

    prompt: str
    slot: str = "claude-code"
    cwd: str = "."
    model: Optional[str] = None
    timeout_s: float = 3600.0
    agent_id: str = "agent"
    run_id: Optional[str] = None
    journal_path: Optional[Union[str, Path]] = None

    # --- sandbox ---
    #: A running Execution Server. Required for any sandbox wiring.
    mcp_url: Optional[str] = None
    #: Provision a sandbox from this image and bind the slot to it.
    sandbox_image: Optional[str] = None
    #: Bind to a sandbox that already exists (a restored snapshot, say).
    sandbox_id: Optional[str] = None
    #: Keep the sandbox after the run -- grading and extraction need it alive.
    keep_sandbox: bool = False
    #: stdio wiring instead: args for `python -m harness.execution.server`.
    mcp_stdio_args: Optional[List[str]] = None

    # --- gateway ---
    use_gateway: bool = False
    routes_file: Optional[str] = None
    gateway_port: int = 0
    budget_usd: Optional[float] = None

    # --- rollback ---
    #: An AshSession-like object to snapshot. Given one, the checkpoint bridge is
    #: installed and every quiesce point records a rollback pair.
    session: Any = None
    snapshot_every_step: bool = False

    # --- resume / fork ---
    resume_session_id: Optional[str] = None
    fork: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunOutcome:
    run_id: str
    journal_path: Path
    status: str
    final_text: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    native_session_id: Optional[str] = None
    sandbox_id: Optional[str] = None
    gateway_url: Optional[str] = None
    checkpoints: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class Orchestrator:
    """Runs one task at a time; ``batch`` puts many of these in flight.

    Deliberately not a registry of live agents: nothing here needs to ask "what
    else is running", and inventing that before a subagent exists would be a
    state machine with one state. When spawning arrives, this is where it goes --
    the sequence below is already what a child would need.
    """

    def __init__(
        self,
        *,
        out_dir: Union[str, Path] = "runs",
        ledger: Optional[ResourceLedger] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.ledger = ledger
        self.on_event = on_event or (lambda kind, payload: None)

    # --- the sequence ------------------------------------------------------
    def run(self, spec: RunSpec) -> RunOutcome:
        from harness.slots import load_slot

        run_id = spec.run_id or new_run_id()
        journal_path = Path(spec.journal_path or self.out_dir / ("%s.jsonl" % run_id))
        slot = load_slot(spec.slot)()
        extra = self._slot_extra(spec)

        # Acquired in order, released in reverse.
        provisioned = None
        gateway = None
        bridge = None
        result: Optional[SlotResult] = None
        error: Optional[str] = None

        # ExitStack so the ledger's run context (which marks the run done even on
        # exception) is optional without duplicating the body under an if.
        with ExitStack() as stack:
            journal = stack.enter_context(
                JournalWriter(journal_path, run_id=run_id, agent_id=spec.agent_id))
            claim = (stack.enter_context(self.ledger.run(run_id))
                     if self.ledger is not None else None)
            try:
                provisioned, mcp = self._wire_sandbox(spec, claim)
                task = TaskSpec(
                    prompt=spec.prompt, cwd=spec.cwd, model=spec.model,
                    timeout_s=spec.timeout_s, extra=extra,
                )
                gateway = self._wire_gateway(spec, journal, task, run_id)

                # Both of these subscribe to the journal, and a subscriber only
                # sees events emitted after it attaches -- so both go on before
                # the slot runs, and the claim goes on before the bridge: the
                # bridge *emits* the checkpoint events the claim records. The
                # other order silently loses every snapshot claim, which a killed
                # process then has no way to reclaim.
                if claim is not None:
                    claim.attach(journal)
                bridge = self._wire_checkpoints(spec, journal)

                result = slot.run(task, journal, mcp)
            except Exception as exc:  # noqa: BLE001 - a run reports, it does not raise
                error = "%s: %s" % (type(exc).__name__, exc)
                journal.emit("run.finished", status="error", error=error)
            finally:
                self._teardown(spec, gateway, provisioned, claim)

        return RunOutcome(
            run_id=run_id,
            journal_path=journal_path,
            status=(result.status if result else "error"),
            final_text=(result.final_text if result else ""),
            usage=(result.usage if result else {}),
            native_session_id=(result.native_session_id if result else None),
            sandbox_id=(provisioned.sandbox_id if provisioned else spec.sandbox_id),
            gateway_url=(gateway.base_url if gateway else None),
            checkpoints=len(bridge.ledger.checkpoints) if bridge else 0,
            error=error or (result.error if result else None),
        )

    # --- steps -------------------------------------------------------------
    def _slot_extra(self, spec: RunSpec) -> dict:
        extra = dict(spec.extra)
        if spec.slot.startswith("claude-code"):
            # Eval hygiene: ignore the developer's local CLAUDE.md / .claude
            # config unless asked for, or a run's behaviour depends on the host.
            extra.setdefault("setting_sources", [])
        if spec.slot.startswith("opencode") and "data_home" not in extra:
            # opencode keeps sessions in SQLite; concurrent lanes sharing it fail
            # with "database is locked".
            extra["data_home"] = str(self.out_dir / "state" / (spec.run_id or "single"))
        if spec.resume_session_id:
            extra["resume_session_id"] = spec.resume_session_id
            if spec.fork:
                extra["fork"] = True
        return extra

    def _wire_sandbox(self, spec: RunSpec, claim):
        """Provision or bind a sandbox, returning (provisioned, wiring)."""
        from harness.execution.provision import provision_http
        from harness.execution.wiring import http_wiring, stdio_wiring

        if spec.mcp_url and spec.sandbox_image:
            provisioned = provision_http(
                spec.mcp_url, image=spec.sandbox_image, agent_id=spec.agent_id,
            )
            if claim is not None:
                # Claimed before the run does anything with it: a process killed
                # mid-run releases nothing, and `harness reap` reads this.
                claim.sandbox(provisioned.sandbox_id, keep=spec.keep_sandbox)
            self.on_event("sandbox", {"sandbox_id": provisioned.sandbox_id})
            return provisioned, provisioned.mcp

        if spec.mcp_url:
            return None, http_wiring(
                spec.mcp_url, agent_id=spec.agent_id, sandbox_id=spec.sandbox_id,
            )
        if spec.mcp_stdio_args is not None:
            return None, stdio_wiring(args=list(spec.mcp_stdio_args))
        return None, None

    def _wire_gateway(self, spec: RunSpec, journal, task: TaskSpec, run_id: str):
        """Start a gateway and point the agent at it, when this run needs one."""
        if not (spec.use_gateway or spec.routes_file or spec.budget_usd):
            return None
        from harness.gateway import GatewayServer, RoutingTable

        table = (RoutingTable.from_file(spec.routes_file) if spec.routes_file
                 else RoutingTable())
        gateway = GatewayServer(table, journal=journal, port=spec.gateway_port).start()
        token = table.mint(spec.agent_id, run_id=run_id, budget_usd=spec.budget_usd)
        # Env, not config: this is the one wiring every agent understands.
        task.env.update(gateway.env_for(token))
        self.on_event("gateway", {"url": gateway.base_url, "budget_usd": spec.budget_usd})
        return gateway

    def _wire_checkpoints(self, spec: RunSpec, journal):
        if spec.session is None:
            return None
        from harness.checkpointing import SnapshotBridge

        return SnapshotBridge.install(journal, spec.session,
                                     always=spec.snapshot_every_step)

    def _teardown(self, spec: RunSpec, gateway, provisioned, claim) -> None:
        """Release in reverse. Each step is independent: one failure must not
        strand the others, which is why they are separately guarded."""
        if gateway is not None:
            try:
                gateway.stop()
            except Exception:  # noqa: BLE001
                pass
        if provisioned is not None and not spec.keep_sandbox:
            try:
                provisioned.destroy()
                if claim is not None:
                    claim.released("sandbox", provisioned.sandbox_id)
            except Exception:  # noqa: BLE001 - `harness reap` is the backstop
                pass
