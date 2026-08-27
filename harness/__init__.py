"""Ash harness core: multi-agent slot abstraction over the sandbox stack.

Layering (docs/ARCHITECTURE.md L2/L3):

- ``harness.core``       event model v2, journal store, slot contract
- ``harness.normalize``  per-agent native event -> unified journal events
- ``harness.slots``      drivers: claude-code (Agent SDK), codex, opencode
- ``harness.execution``  the MCP execution plane: server, pipeline,
                         interceptors, backends, checkpoints, wiring
- ``harness.extract``    a run's answer, read back from any step's snapshot
- ``harness.atif``       journal -> ATIF v1.8 export

Design notes live in harness/README.md. This package must not import
benchmark-specific code (``swebench.*``). What a run's *answer* is stays out of
here too, and needs no hook to: the harness owns the sandbox, so it restores a
snapshot and runs the benchmark's extractor against it (``harness/extract.py``) --
after the fact, at any step, and re-runnable when the extractor changes.
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
