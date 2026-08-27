"""SWE-bench entry point for the ash MCP execution plane.

The server itself is generic and lives in :mod:`harness.execution.server`; it
runs tools in sandboxes and has no notion of what a run's *answer* is. This
module supplies the SWE-bench half -- a git diff per sandbox, via
:class:`swebench.patch.PatchObserver` -- and keeps the CLI that the harnesses and
docs already use:

    python -m swebench.mcp_server --http --port 8400
    python -m swebench.mcp_server --image <docker-image> --patch-dir /tmp/patches/

``--patch-dir`` mounts the observer; every other flag is passed through. The tool
schemas and server classes are re-exported (the same objects) because
``harnesses/marathon_claude_code.py`` imports ``EXEC_TOOLS_SINGLE`` from here to
keep the in-process MCP surface identical to this one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from harness.execution.server import (  # noqa: F401  (re-exported)
    ALL_TOOLS,
    EXEC_TOOLS,
    EXEC_TOOLS_MULTI,
    EXEC_TOOLS_SINGLE,
    LIFECYCLE_TOOLS,
    HttpMcpServer,
    SandboxEntry,
    SandboxPool,
    Session,
    SessionHandler,
    StdioMcpServer,
)
from harness.execution.server import main as _server_main

__all__ = [
    "ALL_TOOLS", "EXEC_TOOLS", "EXEC_TOOLS_MULTI", "EXEC_TOOLS_SINGLE",
    "LIFECYCLE_TOOLS", "HttpMcpServer", "SandboxEntry", "SandboxPool", "Session",
    "SessionHandler", "StdioMcpServer", "main",
]

PATCH_OBSERVER = "swebench.patch:patch_observer"


def main() -> None:
    """Translate ``--patch-dir`` into the observer the generic server mounts."""
    argv = sys.argv[1:]
    patch_dir = _take_value(argv, "--patch-dir")
    patch_file = _take_value(argv, "--patch-file")   # deprecated alias
    if not patch_dir and patch_file:
        patch_dir = str(Path(patch_file).parent)

    if patch_dir:
        # The observer spec carries no arguments, so the directory travels in the
        # environment (see swebench.patch.patch_observer).
        os.environ["ASH_PATCH_DIR"] = patch_dir
        argv += ["--observer", PATCH_OBSERVER]

    sys.argv = [sys.argv[0]] + argv
    _server_main()


def _take_value(argv: list[str], flag: str) -> str | None:
    """Remove ``--flag value`` / ``--flag=value`` from argv, returning the value."""
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if token.startswith(flag + "="):
            return argv.pop(i).split("=", 1)[1]
    return None


if __name__ == "__main__":
    main()
