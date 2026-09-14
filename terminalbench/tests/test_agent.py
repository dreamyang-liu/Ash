from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.models.task.config import MCPServerConfig
from terminalbench.agent import AshClaudeCode


def test_agent_keeps_task_mcp_and_disables_editor_in_real_cli_flags(tmp_path):
    server = MCPServerConfig(name="playwright", transport="sse", url="http://playwright-mcp:3080/sse")
    agent = AshClaudeCode(logs_dir=tmp_path, model_name="test-model", mcp_servers=[server])
    flags = agent.build_cli_flags()
    assert "--disallowedTools" in flags
    assert "mcp__playwright__text_editor" in flags
    assert "text_editor" in agent._ash_denied_tools
    assert "Bash" not in agent._ash_denied_tools
    assert "Read" in agent._ash_denied_tools and "Edit" in agent._ash_denied_tools
    assert "mcp__playwright__browser_navigate" not in flags
    registration = agent._build_register_mcp_servers_command()
    assert "http://playwright-mcp:3080/sse" in registration
    assert agent.SUPPORTS_ATIF
    assert agent.run.__func__ is ClaudeCode.run


def test_additional_denials_are_preserved(tmp_path):
    agent = AshClaudeCode(logs_dir=tmp_path, disallowed_tools="mcp__custom__delete")
    assert "mcp__custom__delete" in agent.build_cli_flags()


def test_setup_uses_official_installer_and_records_tool_policy(tmp_path, monkeypatch):
    seen = []

    async def setup(self, environment):
        seen.append(environment)

    monkeypatch.setattr(ClaudeCode, "setup", setup)
    agent = AshClaudeCode(logs_dir=tmp_path, model_name="test-model")
    environment = object()
    asyncio.run(agent.setup(environment))
    assert seen == [environment]
    metadata = json.loads((tmp_path / "ash-agent.json").read_text())
    assert metadata["execution_location"] == "task_environment"
    assert metadata["checkpoint_support"] is False
