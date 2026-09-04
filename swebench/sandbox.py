"""Ash sandbox session — thin wrapper around ash_sandbox SDK.

Provides synchronous interface for spawning containers and executing tools.
"""

import asyncio
from typing import Callable, Optional

from ash_sandbox import Pool, Sandbox
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

    @property
    def supports_snapshot(self) -> bool:
        return bool(self._pool and self._pool.supports_snapshot())

    @property
    def supports_fork(self) -> bool:
        return bool(self._pool and self._pool.supports_fork())

    def create(self, image: str) -> bool:
        return self._get_loop().run_until_complete(self._create_async(image))

    def snapshot(self, name: str | None = None) -> str:
        """Persist the active sandbox and return a durable snapshot id."""
        if not self._pool or not self._sandbox:
            raise RuntimeError("No active sandbox")
        if not self._pool.supports_snapshot():
            raise NotImplementedError(
                f"{type(self._pool).__name__} does not support durable snapshots"
            )
        return self._get_loop().run_until_complete(
            self._pool.snapshot(self._sandbox, name=name)
        )

    async def _create_async(self, image: str) -> bool:
        try:
            self._pool = build_pool(self.backend, runtime_bin=self.runtime_bin)
            self._sandbox = await self._pool.spawn(image=image)
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

    def restore(self, snapshot_id: str, agent_id: str = "") -> bool:
        """Attach this session to a new sandbox restored from a durable snapshot.

        A restored branch gets its own Pool/client and event loop. This matters
        when several branches run in parallel threads: sharing the source pool's
        async HTTP client across loops is unsafe even though the underlying
        AgentENV snapshot is perfectly shareable.
        """
        return self._get_loop().run_until_complete(
            self._restore_async(snapshot_id, agent_id=agent_id)
        )

    async def _restore_async(self, snapshot_id: str, agent_id: str = "") -> bool:
        if not snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        try:
            self._pool = build_pool(self.backend, runtime_bin=self.runtime_bin)
            if not self._pool.supports_restore():
                raise NotImplementedError(
                    f"{type(self._pool).__name__} does not support snapshot restore"
                )
            self._sandbox = await self._pool.restore(snapshot_id, agent_id=agent_id)
            if not self.quiet:
                print(S.kv("restore ", S.cyan(self.sandbox_id[:12])))
            return True
        except Exception as e:
            print(f"  {S.bright_red('!')} Failed to restore sandbox: {e}")
            if self._pool is not None:
                try:
                    await self._pool.close()
                except Exception:
                    pass
            self._pool = None
            self._sandbox = None
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
