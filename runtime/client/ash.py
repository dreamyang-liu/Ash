"""
Ash Sandbox Client

Separates lifecycle management from tool execution:
  - Sandbox: manages container/process lifecycle
  - Backend: handles tool execution (HTTP, CLI, or MCP)

Usage:

    # Remote runtime (HTTP JSON-RPC)
    async with Sandbox.connect("http://localhost:3000") as sb:
        result = await sb.call("shell", command="ls")

    # Local CLI (no server needed)
    async with Sandbox.local() as sb:
        result = await sb.call("shell", command="ls")

    # Spawn container with runtime
    async with await Sandbox.spawn(image="python:3.11") as sb:
        result = await sb.call("shell", command="pytest")

    # MCP protocol (for FastMCP/Claude Desktop)
    async with Sandbox.mcp("http://localhost:3000/mcp") as sb:
        result = await sb.call("shell", command="ls")

    # Get schemas for LLM
    tools = await sb.tool_schemas(format="openai")

    # Execute model tool_call directly
    result = await sb.execute_tool_call(tool_call)
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx


# ==================== Result ====================

@dataclass
class ToolResult:
    output: str
    is_error: bool
    notifications: list[dict[str, Any]] = field(default_factory=list)


# ==================== Backend (execution layer) ====================

class Backend(ABC):
    """Abstract execution backend."""

    @abstractmethod
    async def call(self, tool_name: str, args: dict) -> ToolResult:
        ...

    @abstractmethod
    async def list_tools(self) -> list[dict]:
        ...

    async def close(self):
        pass


class HTTPBackend(Backend):
    """Calls ash-runtime via HTTP JSON-RPC (POST /)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=360)

    async def call(self, tool_name: str, args: dict) -> ToolResult:
        resp = await self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        })
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return ToolResult(output="", is_error=True, notifications=[])
        result = data["result"]
        text = result["content"][0]["text"] if result.get("content") else ""
        return ToolResult(
            output=text,
            is_error=result.get("isError", False),
            notifications=result.get("notifications", []),
        )

    async def list_tools(self) -> list[dict]:
        resp = await self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        resp.raise_for_status()
        result = resp.json()["result"]
        return result if isinstance(result, list) else result.get("tools", [])

    async def close(self):
        await self._client.aclose()


class MCPBackend(Backend):
    """Calls ash-runtime via MCP Streamable HTTP (POST /mcp)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=360)
        self._initialized = False

    async def _ensure_init(self):
        if self._initialized:
            return
        await self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "ash-client", "version": "0.1.0"}},
        })
        await self._client.post(self.url, json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        self._initialized = True

    async def call(self, tool_name: str, args: dict) -> ToolResult:
        await self._ensure_init()
        resp = await self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        })
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return ToolResult(output="", is_error=True, notifications=[])
        result = data["result"]
        text = result["content"][0]["text"] if result.get("content") else ""
        return ToolResult(
            output=text,
            is_error=result.get("isError", False),
            notifications=result.get("notifications", []),
        )

    async def list_tools(self) -> list[dict]:
        await self._ensure_init()
        resp = await self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        resp.raise_for_status()
        result = resp.json()["result"]
        if isinstance(result, list):
            return result
        return result.get("tools", [])

    async def close(self):
        await self._client.aclose()


class CLIBackend(Backend):
    """Calls ash-runtime directly via subprocess (no HTTP server needed)."""

    def __init__(self, bin_path: str | None = None):
        self.bin_path = bin_path or shutil.which("ash-runtime")
        if not self.bin_path:
            raise RuntimeError("ash-runtime not found in PATH")

    async def call(self, tool_name: str, args: dict) -> ToolResult:
        request = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        })
        proc = await asyncio.create_subprocess_exec(
            self.bin_path, "--mode", "stdio",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate(input=(request + "\n").encode())
        data = json.loads(stdout.decode().strip())
        if data.get("error"):
            return ToolResult(output=data["error"]["message"], is_error=True, notifications=[])
        result = data["result"]
        text = result["content"][0]["text"] if result.get("content") else ""
        return ToolResult(
            output=text,
            is_error=result.get("isError", False),
            notifications=result.get("notifications", []),
        )

    async def list_tools(self) -> list[dict]:
        request = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        proc = await asyncio.create_subprocess_exec(
            self.bin_path, "--mode", "stdio",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate(input=(request + "\n").encode())
        data = json.loads(stdout.decode().strip())
        result = data["result"]
        return result if isinstance(result, list) else result.get("tools", [])


class GatewayBackend(Backend):
    """Calls tools via a gateway that routes by session_id (X-Session-ID header).
    Used in K8s deployments where a shared gateway proxies to many sandbox pods."""

    def __init__(self, gateway_url: str, session_id: str):
        self.gateway_url = gateway_url.rstrip("/")
        self.session_id = session_id
        self._client = httpx.AsyncClient(timeout=360)

    async def call(self, tool_name: str, args: dict) -> ToolResult:
        resp = await self._client.post(
            self.gateway_url,
            headers={"X-Session-ID": self.session_id},
            json={
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return ToolResult(output="", is_error=True, notifications=[])
        result = data["result"]
        text = result["content"][0]["text"] if result.get("content") else ""
        return ToolResult(
            output=text,
            is_error=result.get("isError", False),
            notifications=result.get("notifications", []),
        )

    async def list_tools(self) -> list[dict]:
        resp = await self._client.post(
            self.gateway_url,
            headers={"X-Session-ID": self.session_id},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        return result if isinstance(result, list) else result.get("tools", [])

    async def close(self):
        await self._client.aclose()


# ==================== Sandbox (lifecycle layer) ====================

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


# ==================== SandboxPool (K8s multi-sandbox) ====================

class DockerPool:
    """Manages multiple sandboxes locally via Docker.

    Usage:
        pool = DockerPool(runtime_bin="./ash-runtime")

        sb1 = await pool.spawn(image="python:3.11")
        sb2 = await pool.spawn(image="ubuntu:24.04")

        await sb1.call("shell", command="pytest")
        await sb2.call("shell", command="make build")

        await pool.destroy_all()
    """

    def __init__(self, runtime_bin: str | None = None, port: int = 3000):
        self.runtime_bin = runtime_bin or shutil.which("ash-runtime")
        if not self.runtime_bin:
            raise RuntimeError("ash-runtime binary not found in PATH")
        self.port = port
        self._sandboxes: dict[str, Sandbox] = {}

    async def spawn(
        self,
        image: str = "ubuntu:24.04",
        docker_args: list[str] | None = None,
    ) -> Sandbox:
        """Spawn a new container with ash-runtime injected."""
        host_port = _find_free_port()
        cmd = [
            "docker", "run", "-d",
            "-p", f"{host_port}:{self.port}",
            "-v", f"{self.runtime_bin}:/usr/local/bin/ash-runtime:ro",
            *(docker_args or []),
            image,
            "ash-runtime", "--port", str(self.port),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr}")

        container_id = result.stdout.strip()
        url = f"http://localhost:{host_port}"
        sb = Sandbox(backend=HTTPBackend(url))
        sb._container_id = container_id
        self._sandboxes[container_id] = sb

        await sb._wait_ready()
        return sb

    async def destroy(self, sandbox: Sandbox):
        """Destroy a specific sandbox."""
        cid = sandbox._container_id
        if cid and cid in self._sandboxes:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            del self._sandboxes[cid]
            sandbox._container_id = None

    async def destroy_all(self):
        """Destroy all sandboxes in this pool."""
        for cid in list(self._sandboxes.keys()):
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        self._sandboxes.clear()

    def list(self) -> list[Sandbox]:
        return list(self._sandboxes.values())

    async def close(self):
        await self.destroy_all()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


class SandboxPool:
    """Manages multiple sandboxes via control-plane + gateway (K8s deployment).

    Usage:
        pool = SandboxPool(
            control_plane_url="http://control-plane:80",
            gateway_url="http://gateway:80",
        )

        # Spawn sandboxes
        sb1 = await pool.spawn(image="python:3.11")
        sb2 = await pool.spawn(image="ubuntu:24.04")

        # Use them
        await sb1.call("shell", command="pytest")
        await sb2.call("shell", command="make build")

        # Destroy
        await pool.destroy_all()
    """

    def __init__(self, control_plane_url: str, gateway_url: str, default_image: str = "ubuntu:24.04"):
        self.control_plane_url = control_plane_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.default_image = default_image
        self._client = httpx.AsyncClient(timeout=60)
        self._sandboxes: dict[str, Sandbox] = {}

    async def spawn(
        self,
        image: str | None = None,
        ports: list[int] | None = None,
        env: dict[str, str] | None = None,
        resources: dict | None = None,
    ) -> Sandbox:
        """Spawn a new sandbox via the control plane, return a connected Sandbox."""
        body: dict[str, Any] = {
            "image": image or self.default_image,
            "ports": [{"container_port": p} for p in (ports or [3000])],
        }
        if env:
            body["env"] = env
        if resources:
            body["resources"] = resources

        resp = await self._client.post(f"{self.control_plane_url}/spawn", json=body)
        resp.raise_for_status()
        data = resp.json()

        session_id = data["uuid"]
        sb = Sandbox(backend=GatewayBackend(self.gateway_url, session_id))
        sb._container_id = session_id
        self._sandboxes[session_id] = sb
        return sb

    async def destroy(self, sandbox: Sandbox):
        """Destroy a specific sandbox."""
        if sandbox._container_id and sandbox._container_id in self._sandboxes:
            await self._client.post(
                f"{self.control_plane_url}/destroy",
                json={"uuid": sandbox._container_id},
            )
            del self._sandboxes[sandbox._container_id]
            sandbox._container_id = None

    async def destroy_all(self):
        """Destroy all sandboxes managed by this pool."""
        for sid in list(self._sandboxes.keys()):
            await self._client.post(
                f"{self.control_plane_url}/destroy",
                json={"uuid": sid},
            )
        self._sandboxes.clear()

    async def close(self):
        await self.destroy_all()
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

