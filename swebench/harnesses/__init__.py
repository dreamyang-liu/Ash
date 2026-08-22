"""Pluggable agent harnesses for SWE-bench evaluation.

A harness is a topology: how many agents, how many worktrees, who reports the
answer. Two today:

- litellm: one agent, one sandbox — the custom loop, any litellm model
- claude_code: the Claude Code CLI over MCP

``manager-worker`` (a manager decomposing work across workers sharing one sandbox)
and ``best-of-n`` (N isolated candidates, one patch selected) lived here too, and
were removed while the single-agent path is being settled. Both worked; what they
exercised in L2 — Waggle's optimistic concurrency, several agents sharing one
chain through ``executor_for(pipeline=)`` — is still here and still tested, so
bringing them back is a revert rather than a rewrite.
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
