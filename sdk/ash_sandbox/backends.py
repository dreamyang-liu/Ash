from __future__ import annotations

import asyncio
import json
import shutil
from abc import ABC, abstractmethod

import httpx

from .result import ToolResult


def call_params(tool_name: str, args: dict, agent_id: str = "") -> dict:
    """Build the JSON-RPC params for a tools/call request.

    agent_id travels beside the arguments rather than inside them: the runtime
    uses it to decide whose event subscriptions a response carries and who
    caused an action, which is transport-level addressing rather than an
    argument to the tool. Omitted when empty, so an anonymous caller sends the
    same request it always did.
    """
    params: dict = {"name": tool_name, "arguments": args}
    if agent_id:
        params["agent_id"] = agent_id
    return params


class Backend(ABC):
    """Abstract execution backend."""

    @abstractmethod
    async def call(self, tool_name: str, args: dict,
                   agent_id: str = "") -> ToolResult:
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

    async def call(self, tool_name: str, args: dict,
                   agent_id: str = "") -> ToolResult:
        resp = await self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": call_params(tool_name, args, agent_id),
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

    async def call(self, tool_name: str, args: dict,
                   agent_id: str = "") -> ToolResult:
        await self._ensure_init()
        resp = await self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": call_params(tool_name, args, agent_id),
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

    async def call(self, tool_name: str, args: dict,
                   agent_id: str = "") -> ToolResult:
        request = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": call_params(tool_name, args, agent_id),
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
    """Calls tools through a gateway that routes requests to one sandbox.

    Every such gateway does the same thing -- read a sandbox id from a header
    and forward the body -- but they disagree on the header's name, and some
    also need the port the runtime listens on inside the sandbox. Both are
    therefore configurable, so a new gateway (a microVM host, another proxy)
    is a different set of header names rather than a different Backend.

    Used in K8s deployments where a shared gateway proxies to many sandbox
    pods, and by provisioners whose sandboxes are only reachable that way.
    """

    def __init__(self, gateway_url: str, sandbox_id: str,
                 sandbox_id_header: str = "X-Sandbox-ID",
                 target_port: int | None = None,
                 target_port_header: str = "X-Target-Port"):
        self.gateway_url = gateway_url.rstrip("/")
        self.sandbox_id = sandbox_id
        self.sandbox_id_header = sandbox_id_header
        self.target_port = target_port
        self.target_port_header = target_port_header
        self._client = httpx.AsyncClient(timeout=360)

    def _routing_headers(self) -> dict[str, str]:
        headers = {self.sandbox_id_header: self.sandbox_id}
        if self.target_port is not None:
            headers[self.target_port_header] = str(self.target_port)
        return headers

    async def call(self, tool_name: str, args: dict,
                   agent_id: str = "") -> ToolResult:
        resp = await self._client.post(
            self.gateway_url,
            headers=self._routing_headers(),
            json={
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": call_params(tool_name, args, agent_id),
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
            headers=self._routing_headers(),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        return result if isinstance(result, list) else result.get("tools", [])

    async def close(self):
        await self._client.aclose()
