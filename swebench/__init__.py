"""SWE-bench benchmark for ash-cli.

Runs an LLM agent with ash tools (via CLI) on SWE-bench instances
in isolated Docker containers managed by ash sessions.

Usage:
    python -m swebench.runner --instance sympy__sympy-15599
    python -m swebench.runner --subset verified --split test --workers 4 -o results/
"""

from .models import AgentConfig, CostTracker, ToolResult, Trajectory
from .ash_cli import AshSession
from .agent import AshAgent
from .tools import TOOLS_SCHEMA

__all__ = [
    "AgentConfig",
    "AshAgent",
    "AshSession",
    "CostTracker",
    "TOOLS_SCHEMA",
    "ToolResult",
    "Trajectory",
]
