"""MCP wiring constructors.

The execution plane is ``swebench.mcp_server`` (stdio or HTTP). This module only
builds the *addressing* -- how a slot is told to reach it -- so harness/slots
never grows knowledge of tool semantics or of benchmark code.

Two modes, matching mcp_server's own CLI:

- stdio: one server process per slot, spawned by the agent CLI itself.
- http:  a long-lived multi-session server; slots get a URL. Preferred when one
         Execution Server fronts many slots (agent_id passthrough via headers).
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

from harness.core.slot import McpWiring

DEFAULT_SERVER_MODULE = "swebench.mcp_server"


def stdio_wiring(
    *,
    name: str = "ash",
    module: str = DEFAULT_SERVER_MODULE,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    python: Optional[str] = None,
) -> McpWiring:
    """MCP server as a stdio subprocess: ``python -m swebench.mcp_server ...``."""
    command = [python or sys.executable, "-m", module]
    command.extend(args or [])
    return McpWiring(name=name, command=command, env=dict(env or {}))


def http_wiring(
    url: str,
    *,
    name: str = "ash",
    agent_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> McpWiring:
    """Remote MCP server.

    ``agent_id`` / ``sandbox_id`` are passed as headers so the Execution Server
    can attribute pipeline decisions and taps to the right slot (the
    agent_id-passthrough requirement in the architecture diagram).
    """
    hdrs = dict(headers or {})
    if agent_id:
        hdrs.setdefault("X-Ash-Agent-Id", agent_id)
    if sandbox_id:
        hdrs.setdefault("X-Ash-Sandbox-Id", sandbox_id)
    return McpWiring(name=name, url=url, headers=hdrs)
