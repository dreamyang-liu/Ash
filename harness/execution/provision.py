"""Sandbox provisioning: the orchestrator's half of the tool path.

Division of responsibility, and why it is this way round:

- **The orchestrator creates the sandbox** and holds its id. It has to: the
  sandbox outlives the agent (grading and patch extraction happen after the
  agent stops), it is what gets snapshotted for rollback, and its teardown must
  be guaranteed rather than left to whatever the agent did.
- **The slot only carries the id** into the agent's MCP configuration as a
  header. It never interprets it.
- **The agent never sees it at all.** A bound session is served the
  single-sandbox tool schema, so there is no ``sandbox_id`` argument for the
  model to fill in, forget, or point somewhere else.

That last point is the reason binding is done server-side rather than by having
the slot inject an argument into each call: only claude-code could inject
(``PreToolUse`` → ``updatedInput``); codex and opencode have no such hook. A
header is uniform across all three, and it removes the parameter instead of
fighting over its value.

Usage::

    provisioned = provision_http(
        "http://127.0.0.1:8400/mcp", image="python:3.11-slim", agent_id="slot-1"
    )
    try:
        slot.run(task, journal, provisioned.mcp)
    finally:
        provisioned.destroy()
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from harness.core.slot import McpWiring
from harness.execution.wiring import http_wiring


class ProvisionError(RuntimeError):
    """Sandbox could not be created or destroyed."""


@dataclass
class Provisioned:
    """A sandbox owned by the orchestrator, plus the wiring a slot needs."""

    sandbox_id: str
    mcp: McpWiring
    url: str
    agent_id: str
    _destroy: Optional[Callable[[], None]] = None

    def destroy(self) -> None:
        if self._destroy is not None:
            self._destroy()


def _rpc(url: str, method: str, params: Optional[dict], headers: Dict[str, str],
         timeout: float) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                       "params": params or {}}).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read() or b"{}")
    except urllib.error.URLError as exc:
        raise ProvisionError("MCP server unreachable at %s: %s" % (url, exc)) from exc
    if payload.get("error"):
        raise ProvisionError("MCP error on %s: %s" % (method, payload["error"]))
    return payload.get("result") or {}


def _tool_text(result: dict) -> str:
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text") or ""
    return ""


def provision_http(
    url: str,
    *,
    image: str,
    agent_id: str,
    groups: Optional[List[str]] = None,
    name: str = "ash",
    timeout_s: float = 600.0,
) -> Provisioned:
    """Create a sandbox on a running Execution Server and bind a slot to it.

    The lifecycle calls are made under the orchestrator's *own* session identity
    (``<agent_id>`` as owner), which is what makes the sandbox visible to the
    slot that will use the same identity.
    """
    headers = {"X-Session-Owner": agent_id}
    if groups:
        headers["X-Session-Groups"] = ",".join(groups)

    result = _rpc(
        url,
        "tools/call",
        {"name": "sandbox_create", "arguments": {"image": image, "groups": groups or []}},
        headers,
        timeout_s,
    )
    text = _tool_text(result)
    if result.get("isError") or not text.strip().startswith("{"):
        raise ProvisionError("sandbox_create failed: %s" % (text or result))
    try:
        sandbox_id = json.loads(text)["id"]
    except (ValueError, KeyError) as exc:
        raise ProvisionError("sandbox_create returned no id: %s" % text) from exc

    def destroy() -> None:
        try:
            _rpc(
                url,
                "tools/call",
                {"name": "sandbox_destroy", "arguments": {"sandbox_id": sandbox_id}},
                headers,
                timeout_s,
            )
        except ProvisionError:
            # Teardown is best-effort here; `harness reap` is the backstop that
            # reclaims sandboxes whose owner died before getting this far.
            pass

    return Provisioned(
        sandbox_id=sandbox_id,
        mcp=http_wiring(url, name=name, agent_id=agent_id,
                        sandbox_id=sandbox_id, groups=groups),
        url=url,
        agent_id=agent_id,
        _destroy=destroy,
    )
