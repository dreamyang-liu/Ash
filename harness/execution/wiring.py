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
    groups: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> McpWiring:
    """Remote MCP server.

    Header names are the ones ``swebench.mcp_server`` actually reads -- they are
    a wire contract, not decoration:

    ``X-Session-Owner``
        Session identity. Without it every request is a fresh anonymous session,
        so a sandbox created by one call is invisible to the next (its owner
        group changed) and per-agent interceptor state can never accumulate.
        Passing ``agent_id`` also makes pipeline decisions attributable to the
        right slot.
    ``X-Session-Sandbox``
        Binds this session to a pre-provisioned sandbox: the orchestrator
        created it and states which one this slot owns. The server then serves
        the single-sandbox tool schema, so the model never sees a ``sandbox_id``
        argument -- it cannot omit it and cannot name someone else's sandbox.
    ``X-Session-Groups``
        Extra visibility groups, for deliberately shared sandboxes.
    """
    hdrs = dict(headers or {})
    if agent_id:
        hdrs.setdefault("X-Session-Owner", agent_id)
    if sandbox_id:
        hdrs.setdefault("X-Session-Sandbox", sandbox_id)
    if groups:
        hdrs.setdefault("X-Session-Groups", ",".join(groups))
    return McpWiring(name=name, url=url, headers=hdrs)
