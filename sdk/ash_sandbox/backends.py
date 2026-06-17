from __future__ import annotations

import asyncio
import json
import shutil
from abc import ABC, abstractmethod

import httpx

from .result import ToolResult


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
    """Calls tools via a gateway that routes by sandbox_id (X-Sandbox-ID header).
    Used in K8s deployments where a shared gateway proxies to many sandbox pods."""

    def __init__(self, gateway_url: str, sandbox_id: str):
        self.gateway_url = gateway_url.rstrip("/")
        self.sandbox_id = sandbox_id
        self._client = httpx.AsyncClient(timeout=360)

    async def call(self, tool_name: str, args: dict) -> ToolResult:
        resp = await self._client.post(
            self.gateway_url,
            headers={"X-Sandbox-ID": self.sandbox_id},
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
            headers={"X-Sandbox-ID": self.sandbox_id},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        return result if isinstance(result, list) else result.get("tools", [])

    async def close(self):
        await self._client.aclose()
