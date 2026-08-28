"""One sandbox, driven synchronously: lifecycle, snapshots, and the tool seam.

This is the object the orchestrator needs in order to be an entry point rather
than a function you hand a pre-built environment to. Three things have to be held
by the same owner, and this is that owner:

- **the sandbox's lifetime** -- it outlives the agent, because grading and patch
  extraction happen after the agent stops, and its teardown must be guaranteed
  rather than left to whatever the agent did;
- **the snapshot handle** -- ``snapshot`` / ``squash_snapshot`` / ``swap_sandbox``
  is the contract ``harness.checkpointing`` calls, so whoever cannot call these
  cannot record the environment half of a rollback pair;
- **the executor seam** -- ``(tool_name, args) -> ToolResult``, which is the only
  channel anything has into the environment.

It used to live in ``swebench/sandbox.py`` as ``AshSession``, where the execution
plane could not reach it: the orchestrator therefore required its caller to build
a session and pass it in, which is the opposite of owning the run. ``AshSession``
now subclasses this and keeps only what is genuinely SWE-bench's -- the git
baseline it needs to compute a patch, and the patch itself.

What stays out of here, deliberately: anything that decides *what counts as the
answer*. A session runs tools and takes snapshots. Extracting a diff, grading a
test, deciding an episode succeeded -- those belong to the layer that knows what
the benchmark is.

Reporting goes through :meth:`_note` and :meth:`_warn` so a subclass can render
it however its CLI does, and so this module needs no opinion about presentation.
Both respect ``quiet``.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable, Optional

from ash_sandbox import Pool, Sandbox, Snapshot
from ash_sandbox.result import ToolResult as SdkToolResult
from ash_sandbox.toolset import ToolRegistry

from harness.core.result import ToolResult
from harness.execution.backends import build_pool
from harness.execution.templates import builder_from_backend

#: Identity for the owner's own traffic (patch extraction, resets, test runs).
#: Distinct from any agent's id so that bookkeeping does not appear in an agent's
#: event stream, and does not consume its cursor over the log.
OWNER_AGENT_ID = "harness"


class SandboxSession:
    """Manages one sandbox for one run.

    ``execute`` is the owner's own channel, attributed to :data:`OWNER_AGENT_ID`.
    An agent gets its own channel from :meth:`executor_for`, which binds that
    agent's identity -- the identity belongs to the channel rather than to each
    call, so a call site cannot forget it and an agent cannot act as another.
    """

    def __init__(self, runtime_bin: "str | None" = None, timeout: float = 300.0,
                 quiet: bool = False, backend: "dict | None" = None):
        self.runtime_bin = runtime_bin
        self.timeout = timeout
        self.quiet = quiet
        #: Where sandboxes come from (``harness/execution/backends.py``). A
        #: config's execution section, passed through; ``None`` means local
        #: Docker. A session names no concrete pool, so the same code runs on
        #: containers, microVMs, or a k8s fleet.
        self.backend = backend or {}
        self._pool: Optional[Pool] = None
        self._sandbox: Optional[Sandbox] = None
        # This session's manifest-defined tools. One sandbox is one tool surface,
        # so that is the scope -- and it exists from construction, before any
        # sandbox is spawned, so a panel can be compiled into it whenever the
        # caller gets there.
        #
        # Not the process-default registry, which is what this used to pass: a
        # manifest loaded for one configuration stayed visible to the next, so two
        # configurations in one process saw each other's tools.
        self.tools = ToolRegistry()
        #: The image or template this session was asked to start from.
        self._requested_image: str = ""
        #: The environment this episode descends from, as resolved when it
        #: started (digest-pinned for a cold start). Unchanged by re-boarding.
        self._base_image: str = ""
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --- reporting ---------------------------------------------------------
    def _note(self, label: str, text: str) -> None:
        """One line of progress. Overridden by a subclass with a styled CLI."""
        if not self.quiet:
            sys.stderr.write("  %-9s %s\n" % (label, text))

    def _warn(self, text: str) -> None:
        """Something went wrong but the run continues."""
        if not self.quiet:
            sys.stderr.write("  ! %s\n" % text)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    # --- lifecycle ---------------------------------------------------------
    @property
    def sandbox_id(self) -> str:
        """Stable identity of the active sandbox, or ``unknown`` before spawn.

        ``getattr``, not attribute access: this is read by progress lines, and a
        progress line must not be able to break the operation it reports. That was
        not hypothetical -- reporting used to sit behind ``if not self.quiet``, so
        the id was never computed for a quiet caller, and moving it into a helper
        made the argument eager and turned a successful re-board into an
        AttributeError. "unknown" is already this property's documented answer
        whenever the active sandbox cannot name itself.
        """
        if not self._sandbox:
            return "unknown"
        return getattr(self._sandbox, "sandbox_id", "") or "unknown"

    @property
    def sandbox(self) -> Optional[Sandbox]:
        """The live sandbox, for a caller that must reach the SDK directly.

        Resolve it per use rather than holding it: :meth:`swap_sandbox` replaces
        it mid-run, and a stale reference would keep addressing a destroyed one.
        """
        return self._sandbox

    def create(self, image: str, resources: "Optional[dict]" = None) -> bool:
        """``resources``: e.g. ``{"cpu": 4, "memory_mb": 16384}``. A task that
        declares its needs must get them: running a build-heavy task in the
        backend's default (2 CPUs, 1 GB) OOMs it rather than failing loudly."""
        return self._get_loop().run_until_complete(
            self._create_async(image, resources))

    async def _create_async(self, image: str,
                            resources: "Optional[dict]" = None) -> bool:
        try:
            self._pool = build_pool(self.backend, runtime_bin=self.runtime_bin)
            self._requested_image = image
            # A caller names its environment with an image; the microVM backend
            # starts from templates. Build one per image on demand (cached across
            # instances) when the config says how to get the runtime into it.
            builder = builder_from_backend(self.backend)
            if builder is not None:
                self._note("template", "resolving for %s" % image)
                image = builder.template_for(image, resources)
            # Two entries, and the config says which one this caller's image names
            # are for. A benchmark names its environment with an OCI image
            # reference, which only the cold-start path accepts; a replay or a
            # re-board hands over a snapshot id, which only the snapshot path
            # accepts. Guessing from the string would eventually mistake a
            # template tag for an image tag. A builder-resolved name is a template
            # by construction, so it must never take the cold-start path even when
            # `from_image` is set.
            backend_section = self.backend.get(
                str(self.backend.get("backend") or ""), {}) or {}
            from_image = (builder is None
                          and bool(backend_section.get("from_image")
                                   or self.backend.get("from_image")))
            if from_image and self._pool.supports_cold_start():
                self._sandbox = await self._pool.spawn_from_image(
                    image, resources=resources)
            else:
                # No resources here on purpose: a template or snapshot has its
                # shape baked in (only a cold start can choose one), and the pool
                # rejects the argument rather than ignoring it. The task's shape
                # reached the template above; a resumed run continues in the
                # machine it was running on.
                self._sandbox = await self._pool.spawn(image=image)
            # The base image is known here, at the start of the task, and every
            # later checkpoint descends from it. Pinned once so it survives
            # re-boarding, where the sandbox's immediate source becomes a snapshot
            # and no longer names the container the chain grew from.
            self._base_image = getattr(self._sandbox, "base_ref", "") or image
            self._note("sandbox", self.sandbox_id[:12])
            await self._after_create(self._sandbox)
            return True
        except Exception as e:  # noqa: BLE001 - create reports, it does not raise
            self._warn("Failed to create sandbox: %s" % e)
            return False

    async def _after_create(self, sandbox: Sandbox) -> None:
        """Hook: probe whatever this run needs to know about a fresh sandbox.

        Runs once, on the sandbox this session started with -- notably NOT after
        :meth:`swap_sandbox`, and that asymmetry is deliberate. A subclass records
        a *baseline* here (SWE-bench: the repository commit and which files the
        image itself left untracked), and re-probing it later would file the
        agent's own new files under that baseline and drop them from the answer.
        """

    def destroy(self) -> None:
        if self._sandbox:
            container_id = getattr(self._sandbox, "_container_id", "") or ""
            try:
                self._get_loop().run_until_complete(self._destroy_async())
            except Exception:  # noqa: BLE001
                # Last resort for a local container the pool could not remove: a
                # leaked container holds its image and ports. Only meaningful for
                # Docker -- other backends have no local handle to reap, and their
                # own teardown (or a TTL) is the only recourse.
                if container_id:
                    import subprocess
                    subprocess.run(["docker", "rm", "-f", container_id],
                                   capture_output=True, timeout=10)
            self._sandbox = None
            self._pool = None

    async def _destroy_async(self) -> None:
        if self._pool and self._sandbox:
            sid = self.sandbox_id
            await self._pool.destroy(self._sandbox)
            self._note("cleanup", "destroyed %s" % sid[:12])
            self._sandbox = None
            self._pool = None

    # --- host files --------------------------------------------------------
    def supports_upload(self) -> bool:
        """Whether this session's backend can place host files in the sandbox."""
        return bool(self._pool) and self._pool.supports_upload()

    def upload_file(self, source, destination: str) -> bool:
        """Copy a host file into the sandbox. False when unavailable/failed.

        Distinct from writing through ``text_editor``, which is text-only and goes
        through the runtime: this is for binary and bulk (grading fixtures,
        datasets).
        """
        if not self._sandbox or not self.supports_upload():
            return False
        try:
            self._get_loop().run_until_complete(
                self._pool.upload_file(self._sandbox, source, destination))
            return True
        except Exception as e:  # noqa: BLE001
            self._warn("upload failed: %s" % e)
            return False

    # --- provenance --------------------------------------------------------
    def environment(self) -> dict:
        """What this session actually ran against.

        A run is only reproducible if it can say which environment produced it,
        and the image *name* is not enough: a benchmark image is usually a mutable
        tag (``…:latest``), so the same name can mean different bits a month later.
        ``base_ref`` is what the backend resolved that name to -- digest-pinned for
        a cold start.

        A subclass extends this with whatever else identifies its environment
        (SWE-bench adds the repository commit).
        """
        return {
            "requested_image": self._requested_image,
            #: The environment this episode descends from, pinned at the start of
            #: the task: digest-pinned when it cold-started from an image.
            "base_image": self._base_image,
            #: What the *current* sandbox was started from. Becomes a snapshot id
            #: after a re-board, which is why it is not the origin.
            "base_ref": getattr(self._sandbox, "base_ref", "") or "",
            "sandbox_id": self.sandbox_id,
        }

    # --- checkpoints -------------------------------------------------------
    #
    # A rollout that snapshots every step needs two operations beyond
    # create/destroy, and both are pool capabilities rather than session logic:
    # publish a checkpoint, and continue on a sandbox started from one.
    # `harness.checkpointing` calls exactly the three methods below.

    def supports_snapshot(self) -> bool:
        """Whether this session's backend can publish snapshots."""
        return bool(self._pool) and self._pool.supports_snapshot()

    def snapshot(self, name: "str | None" = None,
                 disk_only: bool = True) -> Optional["Snapshot"]:
        """Checkpoint the live sandbox; it keeps running.

        Defaults to ``disk_only``: a rollout replays by restoring the disk and
        re-feeding the transcript, so paying for a memory image every step buys
        nothing. Returns ``None`` when the backend cannot snapshot or the capture
        fails -- a checkpoint is an optimisation for later analysis, never a reason
        to fail the episode in progress.
        """
        if not self._sandbox or not self.supports_snapshot():
            return None
        try:
            return self._get_loop().run_until_complete(
                self._pool.snapshot(self._sandbox, name=name,
                                    disk_only=disk_only))
        except Exception as e:  # noqa: BLE001
            self._warn("snapshot failed: %s" % e)
            return None

    def squash_snapshot(self, snapshot, name: "str | None" = None):
        """Flatten a snapshot's chain, returning an equivalent snapshot.

        Returns the input unchanged if the backend cannot squash or the call
        fails: a deep chain still works, it just makes its children's checkpoints
        more expensive.
        """
        if not self.supports_snapshot():
            return snapshot
        try:
            return self._get_loop().run_until_complete(
                self._pool.squash(snapshot, name=name))
        except Exception as e:  # noqa: BLE001
            self._warn("squash failed: %s" % e)
            return snapshot

    async def _reachable(self, sandbox, attempts: int = 8,
                         delay: float = 0.5) -> bool:
        """Whether a sandbox's runtime answers a trivial call.

        Retried briefly: a freshly booted sandbox may still be starting its
        runtime when the API call that created it has already returned.
        """
        for attempt in range(attempts):
            try:
                result = await sandbox.call("shell", command="true", timeout=10)
                if not result.is_error:
                    return True
            except Exception:  # noqa: BLE001
                pass
            if attempt + 1 < attempts:
                await asyncio.sleep(delay)
        return False

    def swap_sandbox(self, snapshot) -> bool:
        """Continue this session on a sandbox started from ``snapshot``.

        Called when a checkpoint shows the server compacted the layer chain: the
        running sandbox's own layer stack is never compacted, so without replacing
        it every later capture would re-compact the whole chain. The swap is
        invisible to an agent mid-run, because executors resolve the active
        sandbox per call.

        Deliberately does not run :meth:`_after_create` -- see that hook.
        """
        if not self._sandbox or not self._pool:
            return False
        snapshot_id = getattr(snapshot, "id", snapshot)
        previous = self._sandbox
        try:
            replacement = self._get_loop().run_until_complete(
                self._pool.spawn(image=snapshot_id,
                                 agent_id=previous.agent_id))
        except Exception as e:  # noqa: BLE001
            self._warn("re-board failed, keeping sandbox: %s" % e)
            return False

        # Probe before adopting. A sandbox created from a disk-only snapshot
        # cold-boots, so its runtime is only there if the template declares a
        # startup command that launches it; adopting an unreachable replacement
        # would turn every later tool call into a transport error and kill the
        # episode. Keeping the old sandbox instead costs only a deeper layer chain.
        if not self._get_loop().run_until_complete(self._reachable(replacement)):
            self._warn("re-board target has no runtime; keeping sandbox (does "
                       "the template declare a startup command?)")
            try:
                self._get_loop().run_until_complete(
                    self._pool.destroy(replacement))
            except Exception:  # noqa: BLE001
                pass
            return False

        self._sandbox = replacement
        try:
            self._get_loop().run_until_complete(self._pool.destroy(previous))
        except Exception as e:  # noqa: BLE001
            # The replacement is live and serving calls; a stranded old sandbox is
            # a leak for the TTL to reap, not a run failure.
            self._note("note", "old sandbox not destroyed: %s" % e)
        self._note("re-board", self.sandbox_id[:12])
        return True

    # --- the tool seam -----------------------------------------------------
    def executor_for(self, agent_id: str,
                     pipeline=None) -> Callable[[str, dict], ToolResult]:
        """An executor for one agent, with that agent's identity bound in.

        Hand this to an agent instead of :meth:`execute`: its calls are then
        attributed to it, it keeps its own cursor over the event log, and it has
        no way to act under another agent's name. The signature stays
        ``(tool_name, args) -> ToolResult``, so it drops into anything taking an
        executor -- including the interceptor pipeline.

        ``pipeline`` (a :class:`~harness.execution.pipeline.ToolPipeline`) mounts
        governance around this agent's calls -- the in-process equivalent of the
        MCP proxy's ``--plugins``. Identity and the raw executor are wired in here
        so a caller cannot forget either; agents that must be arbitrated together
        share one pipeline instance. The sandbox id is resolved per call, because
        this session may not have spawned its sandbox yet when the executor is
        handed out.
        """
        def run(tool_name: str, args: dict,
                timeout: "float | None" = None) -> ToolResult:
            return self._run(tool_name, args,
                             timeout if timeout is not None else self.timeout,
                             agent_id)
        if pipeline is None:
            return run
        from harness.execution.pipeline import piped_executor
        return piped_executor(pipeline, run, agent_id,
                              sandbox_id=lambda: self.sandbox_id)

    def execute(self, tool_name: str, args: dict,
                timeout: float = 300.0) -> ToolResult:
        """Run a tool as the owner itself (resets, extraction, tests).

        Agent traffic should go through :meth:`executor_for` so it is not
        attributed to the harness.
        """
        return self._run(tool_name, args, timeout, OWNER_AGENT_ID)

    def _run(self, tool_name: str, args: dict, timeout: float,
             agent_id: str) -> ToolResult:
        if not self._sandbox:
            return ToolResult(success=False, output="", error="No active sandbox")
        try:
            return self._get_loop().run_until_complete(
                self._execute_async(tool_name, args, timeout, agent_id)
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, output="", error=str(e))

    async def _execute_async(self, tool_name: str, args: dict, timeout: float,
                             agent_id: str = OWNER_AGENT_ID) -> ToolResult:
        if "timeout" not in args and tool_name == "shell":
            args["timeout"] = int(timeout)
        # Never splat an agent_id out of args: a model that emits one would
        # otherwise collide with the bound identity (TypeError, surfaced as a
        # baffling tool failure). The channel's identity is the only one that
        # counts, so drop any that arrived in the arguments.
        call_args = {k: v for k, v in args.items() if k != "agent_id"}
        # call_agent_tool, not call: it owns the agent-facing tool surface --
        # builtin routing plus manifest-defined tools, whose artifact->shell
        # expansion it also memoises, so a repeat call skips the download
        # round-trip. This session's registry is passed explicitly: the Sandbox has
        # one of its own, but the panel compiled for this run loaded into ours.
        sdk_result: SdkToolResult = await self._sandbox.call_agent_tool(
            tool_name, call_args, registry=self.tools, agent_id=agent_id)
        return ToolResult.from_sdk(sdk_result)
