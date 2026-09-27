"""Optional mini-swe-agent slot. All effects go through owned HTTP MCP."""

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import threading
from uuid import uuid4

from harness.core.journal import JournalWriter
from harness.core.slot import AgentSlot, McpWiring, SlotCapabilities, SlotResult, TaskSpec

MINI_VERSION = "2.4.6"
_IMPORT_LOCK = threading.Lock()


class MiniSweSlot(AgentSlot):
    name = "mini-swe-agent"
    capabilities = SlotCapabilities(resume=True, fork=True, mcp_stdio=False,
                                    mcp_remote=True, deny_builtin_tools=True)

    def version(self) -> str | None:
        try:
            return version("mini-swe-agent")
        except PackageNotFoundError:
            return None

    def run(self, task: TaskSpec, journal: JournalWriter, mcp: McpWiring | None = None) -> SlotResult:
        installed = self.version()
        if installed != MINI_VERSION:
            raise ValueError(f"mini-swe-agent {MINI_VERSION} required (installed: {installed}); "
                             "install harness/requirements-mini-swe-agent.txt")
        if not mcp or not mcp.url:
            raise ValueError("mini-swe-agent requires HTTP MCP; use --transport http")
        if not task.model:
            raise ValueError("mini-swe-agent requires an explicit model")
        home = Path(task.extra.get("native_home") or journal.path.parent / "mini-native")
        home.mkdir(parents=True, exist_ok=True)
        # Upstream loads a user-global .env on first import. Import against an
        # empty attempt-owned directory so a developer's config cannot set RL
        # credentials, model selection or process-global limits.
        with _IMPORT_LOCK:
            settings = {"MSWEA_GLOBAL_CONFIG_DIR": str(home / "config"),
                        "MSWEA_SILENT_STARTUP": "1",
                        "MSWEA_GLOBAL_COST_LIMIT": "0", "MSWEA_GLOBAL_CALL_LIMIT": "0"}
            previous = {key: os.environ.get(key) for key in settings}
            try:
                os.environ.update(settings)
                from harness.slots.mini_runtime import run
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        return run(task, journal, mcp, home, uuid4().hex)
