"""Ash sandbox session — thin wrapper around ash_sandbox SDK.

Provides synchronous interface for spawning containers and executing tools.
"""

import asyncio
from typing import Optional

from ash_sandbox import DockerPool, Sandbox
from ash_sandbox.result import ToolResult as SdkToolResult

from . import style as S
from .models import ToolResult


class AshSession:
    """Manages an ash sandbox for SWE-bench evaluation."""

    def __init__(self, runtime_bin: str | None = None, timeout: float = 300.0, quiet: bool = False):
        self.runtime_bin = runtime_bin
        self.timeout = timeout
        self.quiet = quiet
        self._pool: Optional[DockerPool] = None
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
            self._pool = DockerPool(runtime_bin=self.runtime_bin)
            self._sandbox = await self._pool.spawn(image=image)
            cid = self._sandbox._container_id or "unknown"
            if not self.quiet:
                print(S.kv("sandbox ", S.cyan(cid[:12])))
            r = await self._sandbox.call("shell", command="git -C /testbed rev-parse HEAD")
            if not r.is_error:
                self._base_commit = r.output.strip()
            return True
        except Exception as e:
            print(f"  {S.bright_red('!')} Failed to create sandbox: {e}")
            return False

    def destroy(self):
        if self._sandbox:
            cid = self._sandbox._container_id or ""
            try:
                self._get_loop().run_until_complete(self._destroy_async())
            except Exception:
                if cid:
                    import subprocess
                    subprocess.run(["docker", "rm", "-f", cid],
                                   capture_output=True, timeout=10)
            self._sandbox = None
            self._pool = None

    async def _destroy_async(self):
        if self._pool and self._sandbox:
            cid = self._sandbox._container_id or ""
            await self._pool.destroy(self._sandbox)
            if not self.quiet:
                print(S.kv("cleanup ", S.dim(f"destroyed {cid[:12]}")))
            self._sandbox = None
            self._pool = None

    def execute(self, tool_name: str, args: dict, timeout: float = 300.0) -> ToolResult:
        if not self._sandbox:
            return ToolResult(success=False, output="", error="No active sandbox")
        try:
            return self._get_loop().run_until_complete(
                self._execute_async(tool_name, args, timeout)
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def _execute_async(self, tool_name: str, args: dict, timeout: float) -> ToolResult:
        if "timeout" not in args and tool_name == "shell":
            args["timeout"] = int(timeout)
        sdk_result: SdkToolResult = await self._sandbox.call(tool_name, **args)
        if sdk_result.is_error:
            return ToolResult(success=False, output=sdk_result.output, error=sdk_result.output)
        return ToolResult(success=True, output=sdk_result.output)

    def get_patch(self) -> str:
        """Get the full diff of all changes vs initial state."""
        self.execute("shell", {"command": "git add -A", "working_dir": "/testbed"})
        base = self._base_commit or "HEAD"
        result = self.execute("shell", {
            "command": f"git diff {base}",
            "working_dir": "/testbed",
        })
        patch = result.output if result.success else ""
        patch = patch.rstrip("\r\n")
        if patch:
            patch += "\n"
        return patch
