"""Agent slot contract.

A *slot* runs one agent against one task and streams normalized events into a
journal. Slots are black boxes to the orchestrator: the contract below is the
whole interface. Concrete drivers live in harness/slots/.

Design constraints (see harness/README.md):
- ``run()`` is synchronous from the caller's perspective; async drivers own
  their own event loop internally.
- Everything a slot learns goes into the journal *as it happens* -- a killed
  run must still leave a usable trajectory (same rule as checkpoints.py).
- Tool injection is expressed as :class:`McpWiring`; the slot only wires it
  into the agent's native config. It never interprets tool semantics.
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from harness.core.journal import JournalWriter


@dataclass
class TaskSpec:
    """What to run. ``extra`` carries slot-specific knobs (documented per slot)."""

    prompt: str
    cwd: str
    model: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout_s: float = 3600.0
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class McpWiring:
    """One MCP server the agent should be given access to.

    Exactly one of ``command`` (stdio subprocess) or ``url`` (remote HTTP)
    should be set. ``swebench.mcp_server`` supports both modes; see
    harness/execution/wiring.py for the standard constructors.
    """

    name: str = "ash"
    command: Optional[List[str]] = None
    url: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class SlotCapabilities:
    resume: bool = False          # can continue a native session by id
    fork: bool = False            # can branch a native session
    mcp_stdio: bool = True        # accepts stdio MCP servers
    mcp_remote: bool = False      # accepts remote (HTTP) MCP servers
    emits_usage: bool = True      # native events carry token usage
    deny_builtin_tools: bool = False  # driver enforces builtin-tool denial


@dataclass
class SlotResult:
    status: str                   # completed | error | timeout | killed
    final_text: str = ""
    usage: Dict[str, object] = field(default_factory=dict)
    native_session_id: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None


class AgentSlot(ABC):
    """Contract every driver implements."""

    name: str = "abstract"
    capabilities: SlotCapabilities = SlotCapabilities()

    #: CLI binary for version probing; None for pure-SDK slots.
    binary: Optional[str] = None

    @abstractmethod
    def run(
        self,
        task: TaskSpec,
        journal: JournalWriter,
        mcp: Optional[McpWiring] = None,
    ) -> SlotResult:
        """Execute the task, streaming normalized events into ``journal``."""

    def kill(self) -> None:  # pragma: no cover - overridden by CLI slots
        """Best-effort termination; orchestrator guarantees env teardown."""

    def version(self) -> Optional[str]:
        """Upstream version string, for run.started provenance + contracts CI."""
        if not self.binary or not shutil.which(self.binary):
            return None
        try:
            out = subprocess.run(
                [self.binary, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return (out.stdout or out.stderr).strip().splitlines()[0]
        except Exception:  # noqa: BLE001 - version probing is best-effort
            return None
