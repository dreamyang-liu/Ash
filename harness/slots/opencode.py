"""opencode slot (``opencode run --format json``).

Verified against opencode 1.18.5.

Wiring notes:
- opencode has *native session fork* (``--session <id> --fork``), the only one of
  the three slots that does. The orchestrator can therefore branch conversation
  state without touching the environment snapshot -- still pair the two for a
  complete rollback (env side effects are not in the session).
- MCP servers are configured in a JSON config file. opencode reads
  ``$OPENCODE_CONFIG`` (a path to a config JSON), so we materialize a per-slot
  file instead of mutating the user's global config -- concurrent slots stay
  isolated.
- ``--pure`` skips external plugins so a developer's local plugins cannot leak
  into eval runs (same intent as ``claude --bare``).
- Prompt is passed as a positional arg; opencode has no stdin prompt mode.
- **Concurrency needs a per-lane data dir.** opencode keeps sessions in a SQLite
  database under ``$XDG_DATA_HOME/opencode``; several processes sharing it fail
  with "database is locked" (observed immediately in a 3-worker batch). Set
  ``extra["data_home"]`` to give a lane its own directory. Scope it per *task*,
  not per attempt: a fork or resume must find the session the earlier run wrote,
  and that lives in this database.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List, Optional

from harness.core.slot import McpWiring, SlotCapabilities, TaskSpec
from harness.normalize import opencode as opencode_normalize
from harness.slots.cli_base import JsonlCliSlot


class OpenCodeSlot(JsonlCliSlot):
    name = "opencode"
    binary = "opencode"
    normalizer = staticmethod(opencode_normalize.normalize)
    capabilities = SlotCapabilities(
        resume=True,
        fork=True,            # --session <id> --fork
        mcp_stdio=True,
        mcp_remote=True,
        emits_usage=True,
        deny_builtin_tools=False,
    )

    def __init__(self) -> None:
        super().__init__()
        self._config_path: Optional[str] = None

    def build_command(self, task: TaskSpec, mcp: Optional[McpWiring]) -> List[str]:
        extra = task.extra or {}
        command = ["opencode", "run", "--format", "json", "--dir", task.cwd]

        if not extra.get("allow_plugins"):
            command.append("--pure")
        if task.model:
            command += ["-m", task.model]
        if extra.get("variant"):
            command += ["--variant", str(extra["variant"])]
        if extra.get("agent"):
            command += ["--agent", str(extra["agent"])]
        if extra.get("thinking"):
            command.append("--thinking")

        session_id = extra.get("resume_session_id")
        if session_id:
            command += ["--session", str(session_id)]
            if extra.get("fork"):
                command.append("--fork")

        command.append(task.prompt)
        return command

    def build_env(self, task: TaskSpec, mcp: Optional[McpWiring]) -> dict:
        env = super().build_env(task, mcp)
        config = self._render_config(mcp, task)
        if config is not None:
            env["OPENCODE_CONFIG"] = config
        data_home = (task.extra or {}).get("data_home")
        if data_home:
            # Isolates the session SQLite db so concurrent lanes do not contend.
            os.makedirs(str(data_home), exist_ok=True)
            env["XDG_DATA_HOME"] = str(data_home)
        return env

    def _render_config(self, mcp: Optional[McpWiring], task: TaskSpec) -> Optional[str]:
        """Write a per-slot opencode config exposing the MCP server."""
        if mcp is None:
            return None
        if mcp.command:
            entry = {
                "type": "local",
                "command": list(mcp.command),
                "enabled": True,
            }
            if mcp.env:
                entry["environment"] = dict(mcp.env)
        elif mcp.url:
            entry = {"type": "remote", "url": mcp.url, "enabled": True}
            if mcp.headers:
                entry["headers"] = dict(mcp.headers)
        else:
            return None

        payload = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {mcp.name: entry},
        }
        base = (task.extra or {}).get("config_base")
        if isinstance(base, dict):
            merged = dict(base)
            merged.setdefault("mcp", {}).update(payload["mcp"])
            merged.setdefault("$schema", payload["$schema"])
            payload = merged

        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", prefix="opencode-slot-", delete=False, encoding="utf-8"
        )
        with handle as fh:
            json.dump(payload, fh)
        self._config_path = handle.name
        return handle.name

    def kill(self) -> None:
        super().kill()
        if self._config_path and os.path.exists(self._config_path):
            try:
                os.unlink(self._config_path)
            except OSError:  # pragma: no cover
                pass
            self._config_path = None
