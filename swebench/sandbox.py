"""Ash sandbox session — thin wrapper around ash_sandbox SDK.

Provides synchronous interface for spawning containers and executing tools.
"""

import asyncio
from typing import Callable, Optional

from ash_sandbox import Pool, Sandbox, Snapshot
from ash_sandbox.result import ToolResult as SdkToolResult

from . import style as S
from ash_sandbox.toolset import ToolRegistry
from .backends import build_pool
from .models import ToolResult
from .patch import UNTRACKED_LIST, WORKDIR, extract_patch


#: Identity for the harness's own traffic (patch extraction, resets, test
#: runs). Distinct from any agent's id so that bookkeeping does not appear in
#: an agent's event stream, and does not consume its cursor over the log.
HARNESS_AGENT_ID = "harness"


class AshSession:
    """Manages an ash sandbox for SWE-bench evaluation.

    ``execute`` is the harness's own channel, attributed to
    ``HARNESS_AGENT_ID``. An agent gets its own channel from
    :meth:`executor_for`, which binds that agent's identity -- the identity
    belongs to the channel rather than to each call, so a call site cannot
    forget it and an agent cannot act as another.
    """

    def __init__(self, runtime_bin: str | None = None, timeout: float = 300.0,
                 quiet: bool = False, backend: dict | None = None):
        self.runtime_bin = runtime_bin
        self.timeout = timeout
        self.quiet = quiet
        #: Where sandboxes come from (``swebench/backends.py``). The config's
        #: execution section, passed through; ``None`` means local Docker. A
        #: session names no concrete pool, so the same harness code runs on
        #: containers, microVMs, or a k8s fleet.
        self.backend = backend or {}
        self._pool: Optional[Pool] = None
        self._sandbox: Optional[Sandbox] = None
        # This session's manifest-defined tools. One sandbox is one tool surface, so
        # that is the scope -- and it exists from construction, before any sandbox is
        # spawned, so a panel can be compiled into it whenever the harness gets there.
        #
        # Not the process-default registry, which is what this used to pass: a manifest
        # loaded for one configuration stayed visible to the next, so two configurations
        # in one process saw each other's tools. Hand this to `build_panel(registry=…)`
        # and the loop and the executor are looking at the same set.
        self.tools = ToolRegistry()
        self._base_commit: str = ""
        #: The image or template this session was asked to start from.
        self._requested_image: str = ""
        #: The environment this episode descends from, as resolved when it
        #: started (digest-pinned for a cold start). Unchanged by re-boarding.
        self._base_image: str = ""
        #: Untracked paths present before the agent ran (see get_patch).
        self._baseline_untracked: set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    @property
    def sandbox_id(self) -> str:
        """Stable identity of the active sandbox, or ``unknown`` before spawn."""
        if not self._sandbox:
            return "unknown"
        return self._sandbox.sandbox_id or "unknown"

    def create(self, image: str) -> bool:
        return self._get_loop().run_until_complete(self._create_async(image))

    async def _create_async(self, image: str) -> bool:
        try:
            self._pool = build_pool(self.backend, runtime_bin=self.runtime_bin)
            self._requested_image = image
            # Two entries, and the config says which one this harness's image
            # names are for. A benchmark names its environment with an OCI
            # image reference, which only the cold-start path accepts; a
            # replay or a re-board hands over a snapshot id, which only the
            # snapshot path accepts. Guessing from the string would eventually
            # mistake a template tag for an image tag.
            backend_section = self.backend.get(
                str(self.backend.get("backend") or ""), {}) or {}
            from_image = bool(backend_section.get("from_image")
                              or self.backend.get("from_image"))
            if from_image and self._pool.supports_cold_start():
                self._sandbox = await self._pool.spawn_from_image(image)
            else:
                self._sandbox = await self._pool.spawn(image=image)
            # The base image is known here, at the start of the task, and every
            # later checkpoint descends from it. Pinned once so it survives
            # re-boarding, where the sandbox's immediate source becomes a
            # snapshot and no longer names the container the chain grew from.
            self._base_image = getattr(self._sandbox, "base_ref", "") or image
            if not self.quiet:
                print(S.kv("sandbox ", S.cyan(self.sandbox_id[:12])))
            r = await self._sandbox.call("shell", command=f"git -C {WORKDIR} rev-parse HEAD")
            if not r.is_error:
                self._base_commit = r.output.strip()
            # What is already untracked before the agent starts. A SWE-bench
            # image can ship a `build/` tree or a stray artifact; those are the
            # image's, and after the run there is no way to tell them from the
            # agent's own new files.
            probe = await self._sandbox.call(
                "shell", command=f"cd {WORKDIR} && {UNTRACKED_LIST}")
            self._baseline_untracked = set(
                line.strip() for line in (probe.output or "").splitlines()
                if line.strip()) if not probe.is_error else set()
            return True
        except Exception as e:
            print(f"  {S.bright_red('!')} Failed to create sandbox: {e}")
            return False

    def destroy(self):
        if self._sandbox:
            container_id = self._sandbox._container_id or ""
            try:
                self._get_loop().run_until_complete(self._destroy_async())
            except Exception:
                # Last resort for a local container the pool could not remove:
                # a leaked container holds its image and ports. Only meaningful
                # for Docker -- other backends have no local handle to reap,
                # and their own teardown (or a TTL) is the only recourse.
                if container_id:
                    import subprocess
                    subprocess.run(["docker", "rm", "-f", container_id],
                                   capture_output=True, timeout=10)
            self._sandbox = None
            self._pool = None

    async def _destroy_async(self):
        if self._pool and self._sandbox:
            sid = self.sandbox_id
            await self._pool.destroy(self._sandbox)
            if not self.quiet:
                print(S.kv("cleanup ", S.dim(f"destroyed {sid[:12]}")))
            self._sandbox = None
            self._pool = None

    def environment(self) -> dict:
        """What this session actually ran against.

        A run is only reproducible if it can say which environment produced it,
        and the image *name* is not enough: a SWE-bench image is usually a
        mutable tag (``…:latest``), so the same name can mean different bits a
        month later. ``base_ref`` is what the backend resolved that name to --
        digest-pinned for a cold start -- and ``base_commit`` is the repository
        state inside it, which is what a replay is ultimately about.
        """
        return {
            "requested_image": self._requested_image,
            #: The environment this episode descends from, pinned at the start
            #: of the task: digest-pinned when it cold-started from an image.
            "base_image": self._base_image,
            #: What the *current* sandbox was started from. Becomes a snapshot
            #: id after a re-board, which is why it is not the origin.
            "base_ref": getattr(self._sandbox, "base_ref", "") or "",
            "base_commit": self._base_commit,
            "sandbox_id": self.sandbox_id,
        }

    # --- Checkpoints ---
    #
    # A rollout that snapshots every step needs two operations beyond
    # create/destroy, and both are pool capabilities rather than session
    # logic: publish a checkpoint, and continue on a sandbox started from one.

    def supports_snapshot(self) -> bool:
        """Whether this session's backend can publish snapshots."""
        return bool(self._pool) and self._pool.supports_snapshot()

    def snapshot(self, name: str | None = None,
                 disk_only: bool = True) -> Optional["Snapshot"]:
        """Checkpoint the live sandbox; it keeps running.

        Defaults to ``disk_only``: a rollout replays by restoring the disk and
        re-feeding the transcript, so paying for a memory image every step
        buys nothing. Returns ``None`` when the backend cannot snapshot or the
        capture fails -- a checkpoint is an optimisation for later analysis,
        never a reason to fail the episode in progress.
        """
        if not self._sandbox or not self.supports_snapshot():
            return None
        try:
            return self._get_loop().run_until_complete(
                self._pool.snapshot(self._sandbox, name=name,
                                    disk_only=disk_only))
        except Exception as e:
            if not self.quiet:
                print(f"  {S.bright_red('!')} snapshot failed: {e}")
            return None

    def squash_snapshot(self, snapshot, name: str | None = None):
        """Flatten a snapshot's chain, returning an equivalent snapshot.

        Returns the input unchanged if the backend cannot squash or the call
        fails: a deep chain still works, it just makes its children's
        checkpoints more expensive.
        """
        if not self.supports_snapshot():
            return snapshot
        try:
            return self._get_loop().run_until_complete(
                self._pool.squash(snapshot, name=name))
        except Exception as e:
            if not self.quiet:
                print(f"  {S.bright_red('!')} squash failed: {e}")
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
            except Exception:
                pass
            if attempt + 1 < attempts:
                await asyncio.sleep(delay)
        return False

    def swap_sandbox(self, snapshot) -> bool:
        """Continue this session on a sandbox started from ``snapshot``.

        Called when a checkpoint shows the server compacted the layer chain:
        the running sandbox's own layer stack is never compacted, so without
        replacing it every later capture would re-compact the whole chain.
        The swap is invisible to an agent mid-run, because executors resolve
        the active sandbox per call.

        Deliberately does not re-probe the repository baseline: ``create``
        recorded which files the image itself left untracked, and re-probing
        now would file the agent's own new files under that baseline and drop
        them from the patch.
        """
        if not self._sandbox or not self._pool:
            return False
        snapshot_id = getattr(snapshot, "id", snapshot)
        previous = self._sandbox
        try:
            replacement = self._get_loop().run_until_complete(
                self._pool.spawn(image=snapshot_id,
                                 agent_id=previous.agent_id))
        except Exception as e:
            if not self.quiet:
                print(f"  {S.bright_red('!')} re-board failed, keeping sandbox: {e}")
            return False

        # Probe before adopting. A sandbox created from a disk-only snapshot
        # cold-boots, so its runtime is only there if the template declares a
        # startup command that launches it; adopting an unreachable
        # replacement would turn every later tool call into a transport error
        # and kill the episode. Keeping the old sandbox instead costs only a
        # deeper layer chain.
        if not self._get_loop().run_until_complete(self._reachable(replacement)):
            if not self.quiet:
                print(f"  {S.bright_red('!')} re-board target has no runtime; "
                      "keeping sandbox (does the template declare a startup "
                      "command?)")
            try:
                self._get_loop().run_until_complete(
                    self._pool.destroy(replacement))
            except Exception:
                pass
            return False

        self._sandbox = replacement
        try:
            self._get_loop().run_until_complete(self._pool.destroy(previous))
        except Exception as e:
            # The replacement is live and serving calls; a stranded old
            # sandbox is a leak for the TTL to reap, not a run failure.
            if not self.quiet:
                print(f"  {S.dim(f'note: old sandbox not destroyed: {e}')}")
        if not self.quiet:
            print(S.kv("re-board ", S.cyan(self.sandbox_id[:12])))
        return True

    def executor_for(self, agent_id: str,
                     pipeline=None) -> Callable[[str, dict], ToolResult]:
        """An executor for one agent, with that agent's identity bound in.

        Hand this to an agent instead of :meth:`execute`: its calls are then
        attributed to it, it keeps its own cursor over the event log, and it
        has no way to act under another agent's name. The signature stays
        ``(tool_name, args) -> ToolResult``, so it drops into anything taking
        an executor -- including the interceptor pipeline.

        ``pipeline`` (a :class:`~swebench.agent.pipeline.ToolPipeline`) mounts
        L2 governance around this agent's calls -- the harness-side equivalent
        of the MCP proxy's ``--coordinate``/``--plugins``. Identity and the
        raw executor are wired in here so a harness cannot forget either;
        agents that must be arbitrated together share one pipeline instance.
        The sandbox id is resolved per call, because this session may not
        have spawned its sandbox yet when the executor is handed out.
        """
        def run(tool_name: str, args: dict, timeout: float | None = None) -> ToolResult:
            return self._run(tool_name, args,
                             timeout if timeout is not None else self.timeout,
                             agent_id)
        if pipeline is None:
            return run
        from .agent.pipeline import piped_executor
        return piped_executor(pipeline, run, agent_id,
                              sandbox_id=lambda: self.sandbox_id)

    def execute(self, tool_name: str, args: dict, timeout: float = 300.0) -> ToolResult:
        """Run a tool as the harness itself (resets, patch extraction, tests).

        Agent traffic should go through :meth:`executor_for` so it is not
        attributed to the harness.
        """
        return self._run(tool_name, args, timeout, HARNESS_AGENT_ID)

    def _run(self, tool_name: str, args: dict, timeout: float,
             agent_id: str) -> ToolResult:
        if not self._sandbox:
            return ToolResult(success=False, output="", error="No active sandbox")
        try:
            return self._get_loop().run_until_complete(
                self._execute_async(tool_name, args, timeout, agent_id)
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def _execute_async(self, tool_name: str, args: dict, timeout: float,
                             agent_id: str = HARNESS_AGENT_ID) -> ToolResult:
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
        # round-trip. This session's registry is passed explicitly: the Sandbox has one
        # of its own, but the panel compiled for this run loaded into ours.
        sdk_result: SdkToolResult = await self._sandbox.call_agent_tool(
            tool_name, call_args, registry=self.tools, agent_id=agent_id)
        return ToolResult.from_sdk(sdk_result)

    def get_patch(self) -> str:
        """Everything the agent changed, as a diff.

        Includes files it created -- only the agent knows whether a new file is
        part of the answer -- while excluding what the image already had and
        caches nobody means to submit. See ``swebench/patch.py``.
        """
        def shell(command: str) -> ToolResult:
            return self.execute("shell", {"command": command,
                                          "working_dir": WORKDIR})

        patch, added = extract_patch(shell, self._base_commit,
                                     self._baseline_untracked)
        if added and not self.quiet:
            shown = ", ".join(added[:4])
            more = f" (+{len(added) - 4})" if len(added) > 4 else ""
            print(S.kv("added   ", S.dim(f"new files in patch: {shown}{more}")))
        return patch
