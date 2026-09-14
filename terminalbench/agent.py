"""Terminal-Bench's container-side Claude Code, with file editing through shell."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment

from harness.slots.claude_code import DENIED_BUILTINS


class AshClaudeCode(ClaudeCode):
    def __init__(self, logs_dir: Path, *args, **kwargs):
        denied = set(DENIED_BUILTINS) - {"Bash", "BashOutput", "KillShell"}
        denied.update({"text_editor", "mcp__ash__text_editor"})
        denied.update(shlex.split(kwargs.get("disallowed_tools") or ""))
        for server in kwargs.get("mcp_servers") or []:
            denied.add(f"mcp__{server.name}__text_editor")
        kwargs["disallowed_tools"] = " ".join(sorted(denied))
        kwargs.setdefault("append_system_prompt",
                          "Use Bash for reading, editing files, and running commands. "
                          "Task-provided MCP tools remain available.")
        super().__init__(logs_dir, *args, **kwargs)
        self._ash_denied_tools = sorted(denied)

    @staticmethod
    def name() -> str:
        return "ash-claude-code"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "ash-agent.json").write_text(json.dumps({
            "driver": "harbor.agents.installed.claude_code.ClaudeCode",
            "execution_location": "task_environment",
            "disallowed_tools": self._ash_denied_tools,
            "mcp_servers": [server.name for server in self.mcp_servers],
            "checkpoint_support": False,
        }, indent=2))
        await super().setup(environment)
