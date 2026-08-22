"""SWE-bench evaluation framework with pluggable agent harnesses.

Harnesses:
    - litellm: Custom agent loop supporting any model via litellm
    - claude-code: Claude Code CLI via MCP sandbox
    - codex: (future) OpenAI Codex CLI

Usage:
    python -m swebench -c swebench/configs/claude-opus.yaml --harness claude-code
    python -m swebench -c swebench/configs/bedrock-opus46.yaml
    python -m swebench --harness litellm --model openai/gpt-4o --api-base http://...
"""

from .models import AgentConfig, CostTracker, ToolResult, Trajectory
from .sandbox import AshSession
from .agent import AshAgent
from .agent.tools import build_panel, load_panel
from .dataset import load_instances, resolve_image, format_task_prompt

__all__ = [
    "AgentConfig",
    "AshAgent",
    "AshSession",
    "CostTracker",
    "build_panel",
    "load_panel",
    "ToolResult",
    "Trajectory",
    "load_instances",
    "resolve_image",
    "format_task_prompt",
]
