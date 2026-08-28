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
    #: How the agent reaches the sandbox when *this* process runs the server:
    #: ``"stdio"`` spawns it as a subprocess, ``"http"`` runs it in-process on an
    #: ephemeral port. Both end up with the orchestrator owning the sandbox, which
    #: is the point -- the choice is only about how the agent talks to it. Ignored
    #: when ``mcp_url`` names a server somebody else is already running.
    transport: str = "stdio"
    #: Where sandboxes come from, when this orchestrator creates one:
    #: ``{"backend": "microvm", "microvm": {...}}``. Empty means local Docker.
    backend: Dict[str, Any] = field(default_factory=dict)
    #: ash-runtime binary to provision into a bare image (microvm templates).
    runtime_bin: Optional[str] = None
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
    #: Which tool panel the agent is offered -- a shipped name (default, full,
    #: bash_only, no_web) or a path to a manifest. None leaves the server on its
    #: built-in four.
    #:
    #: Honoured whenever *this* process runs the server, which is both transports:
    #: stdio passes `--tools`, http compiles the panel and hands it to the
    #: in-process server. It is refused, not ignored, only when ``mcp_url`` names a
    #: server somebody else started -- that process already has whatever panel it
    #: was given, and accepting the argument would report a tool surface the run
    #: did not have.
    tools: Optional[str] = None

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
class OwnedSandbox:
    """A sandbox this orchestrator created, plus whatever serves it.

    Deliberately the same shape as
    :class:`~harness.execution.provision.Provisioned` where the run sequence
    touches it (``sandbox_id``, ``destroy``), so teardown does not branch on which
    kind it got. The difference is what it holds: ``Provisioned`` is a sandbox on
    somebody else's server, this is a session in *this* process -- which is what
    lets the checkpoint bridge snapshot the environment the agent is working in.
    """

    session: Any
    mcp: Optional[McpWiring] = None
    #: An in-process HTTP server, when the transport is http. None for stdio, where
    #: the server is the slot's own subprocess and dies with it.
    server: Any = None
    #: The MutationTracker mounted on the in-process server's pipeline (http).
    #: Created with the server -- the pipeline must exist before the first call --
    #: and handed to the bridge later, so both read ONE tracker: the bridge's
    #: Checkpointer consults exactly the interceptor that watched the calls.
    #: Two trackers here is the bug this replaced, in miniature: the bridge built
    #: its own, nothing fed it, and every step after the first was recorded as
    #: "clean" reuse of the first snapshot -- while the agent was writing files.
    tracker: Any = None
    #: stdio only: where the server subprocess appends its step->snapshot map.
    #: The orchestrator folds this into the journal after the run -- the snapshots
    #: are taken in that subprocess (it is the one that sees the tool boundary),
    #: but the journal lives here.
    checkpoint_log: Optional[Path] = None
    keep: bool = False
    #: Recorded at creation, NOT read from the session on demand. Teardown runs
    #: before the outcome is assembled, and a destroyed session answers "unknown"
    #: -- so a lazy property left every completed run unable to say which sandbox
    #: produced it, which is the reproducibility gap `environment()` exists to
    #: close.
    sandbox_id: str = ""

    def stop_server(self) -> None:
        """Stop serving. Always safe, and always correct to do at teardown: the
        sandbox may outlive the run (``--keep``), the server serving it must not."""
        if self.server is not None:
            self.server.stop()
            self.server = None

    def destroy(self) -> None:
        self.session.destroy()

    def release(self) -> None:
        """Give everything back. Used when wiring fails halfway: the sandbox
        exists but nothing can reach it, so leaking it buys nobody anything."""
        try:
            self.stop_server()
        finally:
            if not self.keep:
                self.destroy()


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
                bridge = self._wire_checkpoints(spec, journal, provisioned)

                result = slot.run(task, journal, mcp)
            except Exception as exc:  # noqa: BLE001 - a run reports, it does not raise
                error = "%s: %s" % (type(exc).__name__, exc)
                journal.emit("run.finished", status="error", error=error)
            finally:
                # Before teardown, so the journal says what happened while the
                # journal is still open; and in `finally`, because a run that died
                # is exactly one whose missing snapshots matter most.
                self._fold_checkpoint_log(provisioned, bridge, result)
                self._report_missing_checkpoints(spec, journal, bridge)
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

        if spec.tools and spec.mcp_url:
            # The panel is compiled by the server process, and over HTTP that
            # process is already running with whatever `--tools` it was given.
            # Accepting this silently would report a surface the run did not have.
            raise ValueError(
                "RunSpec.tools is stdio-only: over HTTP the tool panel belongs to "
                "the server at %s -- start it with `--tools %s`"
                % (spec.mcp_url, spec.tools))

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

        # No remote server named. If an image is, this orchestrator creates the
        # sandbox and runs the server itself -- see _own_sandbox.
        if spec.sandbox_image and spec.session is None:
            owned = self._own_sandbox(spec, claim)
            return owned, owned.mcp

        if spec.mcp_stdio_args is not None:
            args = list(spec.mcp_stdio_args)
            if spec.tools and "--tools" not in args:
                # This process starts the server, so it can say what to serve.
                args += ["--tools", spec.tools]
            return None, stdio_wiring(args=args)
        return None, None

    def _own_sandbox(self, spec: RunSpec, claim) -> "OwnedSandbox":
        """Create the sandbox, then serve it over the requested transport.

        This is what makes the orchestrator an entry point rather than a function
        you hand an environment to. One owner, both transports:

            create session -> stdio: subprocess server --attach <id>
                           -> http:  in-process server, pool.adopt(sandbox)

        Either way the session stays here, which is what the checkpoint bridge
        needs (it snapshots through this handle) and what teardown needs (the
        sandbox dies after the last snapshot, not when a stream closes).

        The transports differ in exactly one thing that matters: stdio hands the
        sandbox over *by id*, so the subprocess re-derives its own handle and the
        backend must support ``attach`` -- only microvm does. http hands over the
        handle itself, so any backend works. A stdio request on a backend that
        cannot attach is therefore refused here, with the alternative named,
        rather than failing later as an unexplained tool error.
        """
        from harness.execution.session import SandboxSession
        from harness.execution.wiring import stdio_wiring

        if spec.transport not in ("stdio", "http"):
            raise ValueError("unknown transport %r; expected 'stdio' or 'http'"
                             % spec.transport)

        session = SandboxSession(runtime_bin=spec.runtime_bin,
                                 backend=dict(spec.backend), quiet=True)
        if not session.create(spec.sandbox_image):
            # The session is quiet here (its progress lines are not this run's
            # output), so the reason has to travel in the exception or it is lost.
            # getattr: any session-shaped object may be substituted here, and not
            # every one records a reason.
            reason = getattr(session, "create_error", "")
            raise RuntimeError("could not create a sandbox from %s%s" % (
                spec.sandbox_image, ": %s" % reason if reason else ""))
        owned = OwnedSandbox(session=session, keep=spec.keep_sandbox,
                             sandbox_id=session.sandbox_id)
        try:
            if claim is not None:
                # Claimed before the run does anything with it: a process killed
                # mid-run releases nothing, and `harness reap` reads this.
                claim.sandbox(session.sandbox_id, keep=spec.keep_sandbox)
            self.on_event("sandbox", {"sandbox_id": session.sandbox_id})

            if spec.transport == "stdio":
                args = ["--attach", session.sandbox_id]
                if spec.runtime_bin:
                    args += ["--runtime-bin", spec.runtime_bin]
                name = spec.backend.get("backend")
                if name:
                    args += ["--backend", name]
                if spec.tools:
                    args += ["--tools", spec.tools]
                if session.supports_snapshot():
                    # The tool boundary happens in the server subprocess, so the
                    # snapshot is taken there too -- the checkpoint machinery sits
                    # at the tool path, in whichever process serves the calls.
                    # What comes back is the step->snapshot map, folded into the
                    # journal by _fold_checkpoint_log after the run.
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    owned.checkpoint_log = self.out_dir / (
                        "%s.ckpt.jsonl" % session.sandbox_id[:12])
                    args += ["--checkpoint-log", str(owned.checkpoint_log)]
                    if spec.snapshot_every_step:
                        args += ["--checkpoint-always"]
                owned.mcp = stdio_wiring(args=args)
            else:
                owned.mcp = self._serve_in_process(spec, owned)
            return owned
        except Exception:
            # The sandbox exists but nothing can reach it; do not leak it while
            # the caller sees only an exception.
            owned.release()
            raise

    def _serve_in_process(self, spec: RunSpec, owned: "OwnedSandbox") -> McpWiring:
        """An MCP server in this process, serving the sandbox we already hold.

        ``adopt``, not ``attach``: attach re-derives a handle from an id, which
        needs a backend that can and yields a second handle to the same sandbox.
        We have the handle -- so any backend works here, Docker included, and the
        pool is told not to destroy what it did not create.
        """
        from harness.execution.interceptors import MutationTracker
        from harness.execution.panel import load_panel
        from harness.execution.pipeline import ToolPipeline
        from harness.execution.server import (ExecSurface, HttpMcpServer,
                                              SandboxPool)
        from harness.execution.wiring import http_wiring

        surface = (ExecSurface(load_panel(spec.tools, format="raw"))
                   if spec.tools else None)
        pool = SandboxPool(runtime_bin=spec.runtime_bin,
                           backend=dict(spec.backend))
        entry = pool.adopt(owned.session.sandbox, [f"owner:{spec.agent_id}"],
                           sandbox_id=owned.session.sandbox_id,
                           image=spec.sandbox_image or "")
        # The tracker is the interceptor half of checkpointing, and it must sit
        # on THIS pipeline -- the one that serves the calls. It is created here
        # (the pipeline cannot be retrofitted once the server is running) and
        # handed to the bridge in _wire_checkpoints (the bridge cannot exist yet:
        # the journal opens later). Nothing flows between the two moments.
        owned.tracker = MutationTracker()
        server = HttpMcpServer(pool, host="127.0.0.1", port=0, surface=surface,
                               pipeline=ToolPipeline([owned.tracker])).start()
        owned.server = server
        self.on_event("mcp", {"url": server.base_url, "transport": "http"})
        # X-Session-Sandbox binds the slot's session to this sandbox, so the model
        # is served the single-sandbox schema and never sees a sandbox_id at all.
        return http_wiring(server.base_url, agent_id=spec.agent_id,
                           sandbox_id=entry.id)

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

    def _wire_checkpoints(self, spec: RunSpec, journal, owned=None):
        """Install the checkpoint bridge on whichever session this run has.

        The caller may hand one in (``spec.session``); otherwise it is the one this
        orchestrator created. That second case is the point of owning the sandbox:
        rollback pairs used to require the caller to build a session and pass it
        down, so a run that merely named an image got no environment half at all.

        A backend that cannot snapshot needs no special case -- ``session.snapshot``
        returns None and the bridge records nothing.
        """
        # getattr: `owned` is whatever _wire_sandbox returned, and only one of the
        # two kinds holds a session. A `Provisioned` sandbox lives on somebody
        # else's server, so there is no handle here to snapshot through -- which is
        # exactly the limitation owning the sandbox removes, and not an error.
        session = spec.session or getattr(owned, "session", None)
        if session is None:
            return None
        from harness.checkpointing import SnapshotBridge

        bridge = SnapshotBridge.install(journal, session,
                                        always=spec.snapshot_every_step,
                                        tracker=getattr(owned, "tracker", None))
        # The in-process server now has something to fire at each tool boundary.
        # Attached here rather than at server construction because the bridge
        # cannot exist before the journal, and the journal opens after the
        # sandbox; no tool call flows until the slot runs, later still.
        server = getattr(owned, "server", None)
        if server is not None:
            from harness.execution.server import ToolBoundary

            server.boundary = ToolBoundary(bridge.on_tool_boundary)
        return bridge

    def _fold_checkpoint_log(self, owned, bridge, result) -> None:
        """Fold the stdio server's step->snapshot map into the journal.

        The subprocess took the snapshots (it serves the tool calls, so it is the
        one standing at the boundary); it wrote one JSON line per capture. This
        pairs each with the conversation ref the slot disclosed and records it
        through the bridge's ledger, so ``load_checkpoints`` and ``fork_plan``
        read a stdio run exactly as they read an http one.

        Runs in the ``finally`` on purpose: a run that was killed mid-task is
        precisely the one whose surviving snapshots must not be orphaned -- a map
        that only lands on clean exits repeats the 300-snapshots-no-map failure.
        """
        import json as _json

        path = getattr(owned, "checkpoint_log", None)
        if not path or bridge is None or not Path(path).exists():
            return
        session_ckpt = result.native_session_id if result is not None else None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = _json.loads(line)
            except ValueError:
                continue          # a torn last line from a killed subprocess
            if not entry.get("snapshot_id"):
                continue
            bridge.ledger.record(
                int(entry.get("step", 0)),
                entry["snapshot_id"],
                session_ckpt=session_ckpt,
                reason=entry.get("reason") or "tool_boundary_stdio",
                captured=bool(entry.get("captured", True)),
            )
        self.on_event("checkpoints",
                      {"count": len(bridge.ledger.checkpoints),
                       "source": str(path)})

    def _report_missing_checkpoints(self, spec: RunSpec, journal, bridge) -> None:
        """Say so when checkpointing was on and produced nothing.

        The case this exists for: an SDK slot journals its turn from inside its
        event loop, the bridge cannot enter the session's own loop from there, and
        every opportunity is skipped. That used to be invisible -- the run reported
        success, the count sat in a private field, and the snapshots a fork needs
        simply were not there. Silence about a capability that did not happen is
        worse than the absence itself, because it is discovered later, from the
        outside, by someone trying to branch a run that cannot be branched.

        Both owned transports checkpoint at the tool boundary now -- http notifies
        the bridge in-process, stdio's subprocess snapshots itself and the map is
        folded back -- so this fires only on the wirings that genuinely have no
        boundary to stand at: a hand-rolled ``mcp_stdio_args`` without
        ``--checkpoint-log``, or a caller-supplied session with no sandbox wiring
        at all.
        """
        if bridge is None:
            return
        skipped = getattr(bridge, "skipped_on_loop", 0)
        recorded = len(getattr(getattr(bridge, "ledger", None), "checkpoints", ()))
        if recorded or not skipped:
            return
        advice = ("this slot journals from inside its event loop, so the turn "
                  "boundary cannot be used; give the orchestrator the sandbox "
                  "(sandbox_image + transport stdio|http) and it checkpoints at "
                  "each tool call instead -- or add --checkpoint-log to your own "
                  "mcp_stdio_args")
        journal.emit("checkpoint.unavailable", skipped=skipped,
                     transport=spec.transport, slot=spec.slot, reason=advice)
        self.on_event("checkpoint.unavailable",
                      {"skipped": skipped, "transport": spec.transport,
                       "reason": advice})

    def _teardown(self, spec: RunSpec, gateway, provisioned, claim) -> None:
        """Release in reverse. Each step is independent: one failure must not
        strand the others, which is why they are separately guarded."""
        if gateway is not None:
            try:
                gateway.stop()
            except Exception:  # noqa: BLE001
                pass
        # Stops regardless of --keep, and before the sandbox goes: a server left
        # serving a destroyed sandbox answers every call with a transport error.
        stop_server = getattr(provisioned, "stop_server", None)
        if stop_server is not None:
            try:
                stop_server()
            except Exception:  # noqa: BLE001
                pass
        if provisioned is not None and not spec.keep_sandbox:
            try:
                provisioned.destroy()
                if claim is not None:
                    claim.released("sandbox", provisioned.sandbox_id)
            except Exception:  # noqa: BLE001 - `harness reap` is the backstop
                pass
