"""Ash SDK session for executing commands in sandboxes.

Replaces the old CLI-based session with direct SDK calls via DockerPool.
"""

import asyncio
from typing import Optional

from ash_sandbox import DockerPool, Sandbox
from ash_sandbox.result import ToolResult as SdkToolResult

from . import style as S
from .models import ToolResult


class AshSession:
    """Manages an ash sandbox for SWE-bench evaluation.

    Uses the Python SDK (DockerPool) to spawn containers and call tools
    directly via HTTP JSON-RPC — no ash CLI binary needed.
    """

    def __init__(self, runtime_bin: str | None = None, timeout: float = 300.0):
        self.runtime_bin = runtime_bin
        self.timeout = timeout
        self._pool: Optional[DockerPool] = None
        self._sandbox: Optional[Sandbox] = None

    def create(self, image: str) -> bool:
        """Spawn a sandbox container with the given image."""
        return asyncio.get_event_loop().run_until_complete(self._create_async(image))

    async def _create_async(self, image: str) -> bool:
        try:
            self._pool = DockerPool(runtime_bin=self.runtime_bin)
            self._sandbox = await self._pool.spawn(image=image)
            cid = self._sandbox._container_id or "unknown"
            print(S.kv("sandbox ", S.cyan(cid[:12])))
            return True
        except Exception as e:
            print(f"  {S.bright_red('!')} Failed to create sandbox: {e}")
            return False

    def destroy(self):
        """Destroy the sandbox container."""
        if self._sandbox:
            asyncio.get_event_loop().run_until_complete(self._destroy_async())

    async def _destroy_async(self):
        if self._pool and self._sandbox:
            cid = self._sandbox._container_id or ""
            await self._pool.destroy(self._sandbox)
            print(S.kv("cleanup ", S.dim(f"destroyed {cid[:12]}")))
            self._sandbox = None
            self._pool = None

    def execute(self, tool_name: str, args: dict, timeout: float = 300.0) -> ToolResult:
        """Execute a tool call in the sandbox."""
        if not self._sandbox:
            return ToolResult(success=False, output="", error="No active sandbox")

        try:
            result = asyncio.get_event_loop().run_until_complete(
                self._execute_async(tool_name, args, timeout)
            )
            return result
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
        """Get the git diff (patch) from the sandbox."""
        result = self.execute("shell", {"command": "git diff", "working_dir": "/testbed"})
        if result.success and result.output.strip():
            return result.output.strip()

        # Also check untracked files
        self.execute("shell", {"command": "git add -N .", "working_dir": "/testbed"})
        result = self.execute("shell", {"command": "git diff", "working_dir": "/testbed"})
        return result.output.strip() if result.success else ""
