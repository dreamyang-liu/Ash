"""Identity-preserving app-server dynamic-tool facade for an owned Ash MCP."""

from __future__ import annotations

import json
from uuid import uuid4

import httpx

from harness.core.checkpoint_identity import CALL_IDENTITY_KEY


class CodexTools:
    def __init__(self, wiring, journal, timeout_s: float) -> None:
        if not wiring.url:
            raise ValueError("Codex exact checkpoints require owned HTTP MCP")
        self.wiring = wiring
        self.journal = journal
        self.timeout_s = timeout_s
        self.headers = dict(wiring.headers or {})
        self.namespace = wiring.name
        self._called: set[str] = set()
        self._request("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "ash-codex-tools", "version": "1"},
        })
        listed = self._request("tools/list", {}).get("tools", [])
        self.tools = {tool["name"]: tool for tool in listed}
        if not self.tools:
            raise ValueError("Owned MCP returned no tools")

    def _request(self, method: str, params: dict) -> dict:
        request_id = uuid4().hex
        headers = {"Accept": "application/json, text/event-stream", **self.headers}
        response = httpx.post(self.wiring.url, headers=headers, timeout=self.timeout_s,
                              json={"jsonrpc": "2.0", "id": request_id,
                                    "method": method, "params": params})
        response.raise_for_status()
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self.headers["mcp-session-id"] = session_id
        if "text/event-stream" in response.headers.get("content-type", ""):
            messages = [json.loads(line[5:].strip()) for line in response.text.splitlines()
                        if line.startswith("data:")]
            body = next((message for message in messages if message.get("id") == request_id), {})
        else:
            body = response.json()
        if body.get("id") != request_id or "error" in body:
            raise ValueError("Invalid owned MCP response")
        return body["result"]

    def definitions(self) -> list[dict]:
        return [{"type": "namespace", "name": self.namespace,
                 "description": "Tools executing in the checkpointed Ash sandbox",
                 "tools": [{"type": "function", "name": tool["name"],
                            "description": tool.get("description", ""),
                            "inputSchema": tool["inputSchema"]}
                           for tool in self.tools.values()]}]

    def call(self, params: dict) -> dict:
        name = params.get("tool")
        call_id = params.get("callId")
        arguments = params.get("arguments")
        if (params.get("namespace") != self.namespace or name not in self.tools
                or not isinstance(arguments, dict) or not isinstance(call_id, str)
                or not call_id or call_id in self._called or CALL_IDENTITY_KEY in arguments):
            raise ValueError("Unrecognized or repeated native tool call")
        self._called.add(call_id)
        event = self.journal.emit("tool.started", call_id=call_id,
                                  name=self.namespace + "__" + name, args=arguments)
        result = self._request("tools/call", {"name": name, "arguments": {
            **arguments, CALL_IDENTITY_KEY: {"call_id": call_id, "step": event["step"]}}})
        content = result.get("content", [])
        if any(item.get("type") != "text" for item in content):
            raise ValueError("Exact Codex facade currently requires text MCP output")
        self.journal.emit("tool.finished", call_id=call_id,
                          name=self.namespace + "__" + name,
                          status="error" if result.get("isError") else "ok", output=result)
        return {"success": not result.get("isError", False),
                "contentItems": [{"type": "inputText", "text": item["text"]} for item in content]}
