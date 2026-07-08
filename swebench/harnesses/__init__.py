"""Pluggable agent harnesses for SWE-bench evaluation.

Each harness implements a different agent backend:
- litellm: Custom agent loop using litellm (any model)
- claude_code: Claude Code CLI via MCP
- manager_worker: explore -> decompose -> parallel workers on one shared sandbox
- best_of_n: N isolated parallel candidates, one patch selected
- codex: OpenAI Codex CLI (future)
"""

from .base import BaseHarness
from .litellm import LiteLLMHarness
from .claude_code import ClaudeCodeHarness
from .manager_worker import ManagerWorkerHarness
from .best_of_n import BestOfNHarness

HARNESSES = {
    "litellm": LiteLLMHarness,
    "claude-code": ClaudeCodeHarness,
    "manager-worker": ManagerWorkerHarness,
    "best-of-n": BestOfNHarness,
    # "codex": CodexHarness,
}


def get_harness(name: str) -> type:
    """Get harness class by name."""
    if name not in HARNESSES:
        available = ", ".join(HARNESSES.keys())
        raise ValueError(f"Unknown harness: {name}. Available: {available}")
    return HARNESSES[name]
