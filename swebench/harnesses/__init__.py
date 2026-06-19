"""Pluggable agent harnesses for SWE-bench evaluation.

Each harness implements a different agent backend:
- litellm: Custom agent loop using litellm (any model)
- claude_code: Claude Code CLI via MCP
- codex: OpenAI Codex CLI (future)
"""

from .base import BaseHarness
from .litellm import LiteLLMHarness
from .claude_code import ClaudeCodeHarness

HARNESSES = {
    "litellm": LiteLLMHarness,
    "claude-code": ClaudeCodeHarness,
    # "codex": CodexHarness,
}


def get_harness(name: str) -> type:
    """Get harness class by name."""
    if name not in HARNESSES:
        available = ", ".join(HARNESSES.keys())
        raise ValueError(f"Unknown harness: {name}. Available: {available}")
    return HARNESSES[name]
