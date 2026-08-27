"""Ash harness core: multi-agent slot abstraction over the sandbox stack.

Layering (docs/ARCHITECTURE.md L2/L3):

- ``harness.core``       event model v2, journal store, slot contract
- ``harness.normalize``  per-agent native event -> unified journal events
- ``harness.slots``      drivers: claude-code (Agent SDK), codex, opencode
- ``harness.execution``  MCP wiring helpers (tool injection via swebench.mcp_server)
- ``harness.atif``       journal -> ATIF v1.8 export

Design notes live in harness/README.md. This package must not import
benchmark-specific code (swebench.*) at module import time; slots reference
swebench.mcp_server only as a subprocess command string.
"""

from harness.core.events import JOURNAL_SCHEMA_VERSION
from harness.core.journal import JournalWriter, read_journal
from harness.core.slot import (
    AgentSlot,
    McpWiring,
    SlotCapabilities,
    SlotResult,
    TaskSpec,
)

__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "JournalWriter",
    "read_journal",
    "AgentSlot",
    "McpWiring",
    "SlotCapabilities",
    "SlotResult",
    "TaskSpec",
]
