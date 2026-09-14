"""TOML serialization and isolated tool configuration for the Codex SDK."""

from __future__ import annotations

import json
import os
import subprocess
from typing import List, Optional

from harness.core.slot import McpWiring, TaskSpec


def _toml_str(value: str) -> str:
    return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_str_array(values: List[str]) -> str:
    return "[%s]" % ",".join(_toml_str(value) for value in values)


def _mcp_config_pairs(mcp: Optional[McpWiring]) -> List[str]:
    """Serialize MCP configuration into the SDK's TOML override entries."""
    if mcp is None:
        return []
    prefix = "mcp_servers.%s" % mcp.name
    out: List[str] = []
    if mcp.command:
        out.append("%s.command=%s" % (prefix, _toml_str(mcp.command[0])))
        if len(mcp.command) > 1:
            out.append("%s.args=%s" % (prefix, _toml_str_array(mcp.command[1:])))
        for key, value in (mcp.env or {}).items():
            out.append("%s.env.%s=%s" % (prefix, key, _toml_str(value)))
    elif mcp.url:
        out.append("%s.url=%s" % (prefix, _toml_str(mcp.url)))
        for key, value in (mcp.headers or {}).items():
            out.append("%s.http_headers.%s=%s" % (prefix, key, _toml_str(value)))
    return out


def _mcp_tool_policy() -> dict:
    """Native MCP resource helpers remain available; task tools belong to MCP."""
    disabled = (
        "shell_tool", "shell_snapshot", "unified_exec", "view_image", "multi_agent", "multi_agent_v2",
        "apps", "plugins", "code_mode", "code_mode_host", "code_mode_only",
        "computer_use", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
        "in_app_browser", "image_generation", "goals", "memories", "skill_search",
        "tool_suggest", "request_permissions_tool", "default_mode_request_user_input",
        "sleep_tool", "artifact",
    )
    return {
        "web_search": "disabled",
        "features": {feature: False for feature in disabled},
        "tools": {"experimental_request_user_input": {"enabled": False},
                  "update_plan": {"enabled": False}},
    }


def _config_pairs(config: dict, prefix: str = "") -> List[str]:
    pairs = []
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            pairs.extend(_config_pairs(value, path))
        else:
            pairs.append(f"{path}={json.dumps(value)}")
    return pairs


def _mcp_only_overrides(mcp: Optional[McpWiring]) -> List[str]:
    if mcp is None:
        return []
    return [*_mcp_config_pairs(mcp), *_config_pairs(_mcp_tool_policy())]


def _configured_mcp_servers(binary: str, task: TaskSpec, overrides: List[str]) -> List[str]:
    """Inspect resolved configuration without starting MCP servers or inference."""
    command = [binary]
    for pair in overrides:
        command.extend(["-c", pair])
    command.extend(["mcp", "list", "--json"])
    result = subprocess.run(command, cwd=task.cwd, env={**os.environ, **task.env},
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError("Cannot inspect Codex MCP configuration; refusing unverified tool exposure")
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or any(not isinstance(row.get("name"), str) for row in rows):
        raise RuntimeError("Invalid Codex MCP configuration inventory")
    return [row["name"] for row in rows]


def _disabled_mcp_overrides(names: List[str]) -> List[str]:
    if not names:
        return []
    entries = ",".join(f"{json.dumps(name)}={{enabled=false}}" for name in names)
    return [f"mcp_servers={{{entries}}}"]
