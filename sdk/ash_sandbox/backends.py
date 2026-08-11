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
    """Calls ash-runtime directly via subprocess (no HTTP server needed).

    One runtime process serves every call. It used to be one process per call,
    which cost more than startup time: a runtime's event log, its background
    processes and its artifact cache all live inside that process, so every
    call began in an empty sandbox while the filesystem kept its state --
    events simply never arrived, and only through this backend. The runtime's
    stdio mode is a request-per-line loop, so one process answers as many
    requests as we send it.
    """

    def __init__(self, bin_path: str | None = None):
        self.bin_path = bin_path or shutil.which("ash-runtime")
        if not self.bin_path:
            raise RuntimeError("ash-runtime not found in PATH")
        self._proc: asyncio.subprocess.Process | None = None
        # One request in flight at a time: responses are matched by arrival
        # order on a shared pipe, so interleaving would mismatch them.
        self._lock = asyncio.Lock()
        self._next_id = 0

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._proc is None or self._proc.returncode is not None:
            self._proc = await asyncio.create_subprocess_exec(
                self.bin_path, "--mode", "stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        return self._proc

    async def _request(self, method: str, params: dict) -> dict:
        """Send one JSON-RPC line and read its response line."""
        async with self._lock:
            proc = await self._ensure_process()
            self._next_id += 1
            line = json.dumps({
                "jsonrpc": "2.0", "id": self._next_id,
                "method": method, "params": params,
            }) + "\n"
            proc.stdin.write(line.encode())
            await proc.stdin.drain()

            raw = await proc.stdout.readline()
            if not raw:
                # The runtime died; drop it so the next call starts a fresh one
                # instead of writing into a closed pipe forever.
                self._proc = None
                raise RuntimeError("ash-runtime exited while handling a request")
            return json.loads(raw.decode())

    async def call(self, tool_name: str, args: dict,
                   agent_id: str = "") -> ToolResult:
        data = await self._request(
            "tools/call", call_params(tool_name, args, agent_id))
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
        data = await self._request("tools/list", {})
        result = data["result"]
        return result if isinstance(result, list) else result.get("tools", [])

    async def close(self) -> None:
        """Stop the runtime process, ending its sandbox state with it."""
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            proc.kill()
            await proc.wait()


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
