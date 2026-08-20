"""Ash sandbox session — thin wrapper around ash_sandbox SDK.

Provides synchronous interface for spawning containers and executing tools.
"""

import asyncio
from typing import Callable, Optional

from ash_sandbox import Pool, Sandbox
from ash_sandbox.result import ToolResult as SdkToolResult

from . import style as S
from .agent.custom_tools import DEFAULT_REGISTRY
from .backends import build_pool
from .models import ToolResult
from .patch import WORKDIR, extract_patch, untracked_paths


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
        self._base_commit: str = ""
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
            self._sandbox = await self._pool.spawn(image=image)
            if not self.quiet:
                print(S.kv("sandbox ", S.cyan(self.sandbox_id[:12])))
            r = await self._sandbox.call("shell", command="git -C /testbed rev-parse HEAD")
            if not r.is_error:
                self._base_commit = r.output.strip()
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
        # round-trip. The registry is passed explicitly because manifests load
        # into the process-default one while a Sandbox starts with its own.
        sdk_result: SdkToolResult = await self._sandbox.call_agent_tool(
            tool_name, call_args, registry=DEFAULT_REGISTRY, agent_id=agent_id)
        return ToolResult.from_sdk(sdk_result)

    def get_patch(self) -> str:
        """The agent's changes to the repository's own files, as a diff.

        Files the agent created are left out; see ``swebench/patch.py``. Unless
        quiet, they are named on the way past — a reproduce script is expected, a
        `build/` tree means a command ran that the task did not need.
        """
        def shell(command: str) -> ToolResult:
            return self.execute("shell", {"command": command,
                                          "working_dir": WORKDIR})

        patch = extract_patch(shell, self._base_commit)
        if not self.quiet:
            left_out = untracked_paths(shell)
            if left_out:
                shown = ", ".join(left_out[:4])
                more = f" (+{len(left_out) - 4})" if len(left_out) > 4 else ""
                print(S.kv("untracked", S.dim(f"not in patch: {shown}{more}")))
        return patch
