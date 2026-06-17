from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
from dataclasses import dataclass, field

import httpx

from .backends import Backend, CLIBackend, HTTPBackend, MCPBackend
from .result import ToolResult


@dataclass
class Sandbox:
    """Manages lifecycle + delegates execution to a Backend."""

    backend: Backend
    _container_id: str | None = field(default=None, repr=False)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _tools_cache: list[dict] | None = field(default=None, repr=False)

    # --- Constructors ---

    @classmethod
    def connect(cls, url: str) -> Sandbox:
        """Connect to a running ash-runtime via HTTP."""
        return cls(backend=HTTPBackend(url))

    @classmethod
    def mcp(cls, url: str) -> Sandbox:
        """Connect via MCP protocol."""
        return cls(backend=MCPBackend(url))

    @classmethod
    def local(cls, bin_path: str | None = None) -> Sandbox:
        """Use CLI backend — no server needed, calls binary directly."""
        return cls(backend=CLIBackend(bin_path))

    @classmethod
    async def spawn(
        cls,
        image: str = "ubuntu:24.04",
        port: int = 3000,
        runtime_bin: str | None = None,
        docker_args: list[str] | None = None,
    ) -> Sandbox:
        """Spawn a Docker container with ash-runtime injected."""
        if runtime_bin is None:
            runtime_bin = shutil.which("ash-runtime")
            if runtime_bin is None:
                raise RuntimeError("ash-runtime binary not found in PATH")

        host_port = _find_free_port()
        cmd = [
            "docker", "run", "-d",
            "-p", f"{host_port}:{port}",
            "-v", f"{runtime_bin}:/usr/local/bin/ash-runtime:ro",
            *(docker_args or []),
            image,
            "ash-runtime", "--port", str(port),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr}")

        container_id = result.stdout.strip()
        url = f"http://localhost:{host_port}"
        sb = cls(backend=HTTPBackend(url))
        sb._container_id = container_id

        await sb._wait_ready()
        return sb

    # --- Lifecycle ---

    async def destroy(self):
        if self._container_id:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", self._container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._container_id = None

    async def close(self):
        await self.backend.close()
        await self.destroy()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    # --- Execution ---

    async def call(self, tool_name: str, **kwargs) -> ToolResult:
        """Call a tool by name."""
        return await self.backend.call(tool_name, kwargs)

    async def execute_tool_call(self, tool_call: dict) -> ToolResult:
        """Execute an LLM tool_call (OpenAI or Anthropic format)."""
        if "function" in tool_call:
            name = tool_call["function"]["name"]
            args_raw = tool_call["function"].get("arguments", "{}")
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        else:
            name = tool_call["name"]
            args = tool_call.get("input", {})
        return await self.backend.call(name, args)

    async def execute_tool_calls(self, tool_calls: list[dict]) -> list[ToolResult]:
        """Execute multiple tool_calls sequentially."""
        return [await self.execute_tool_call(tc) for tc in tool_calls]

    # --- Schemas ---

    async def tool_schemas(self, format: str = "openai") -> list[dict]:
        """Get tool schemas for LLM function calling.

        format: "openai" | "anthropic" | "raw"
        """
        tools = await self._get_tools()

        if format == "anthropic":
            return [{"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]} for t in tools]
        if format == "raw":
            return tools

        # OpenAI format
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}} for t in tools]

    # --- Internal ---

    async def _get_tools(self) -> list[dict]:
        if self._tools_cache is None:
            self._tools_cache = await self.backend.list_tools()
        return self._tools_cache

    async def _wait_ready(self, timeout: float = 30):
        deadline = asyncio.get_event_loop().time() + timeout
        client = httpx.AsyncClient(timeout=5)
        try:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(self.backend.url)
                    if resp.status_code == 200:
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(0.3)
        finally:
            await client.aclose()
        raise TimeoutError(f"ash-runtime not ready after {timeout}s")


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]
