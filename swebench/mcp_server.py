"""SWE-bench's entry point to the ash MCP execution plane.

The proxy itself is generic and lives in :mod:`harness.execution.server`. This
module is a thin alias: it keeps the documented command line working and
re-exports the tool schemas, so ``harnesses/claude_code.py`` and
``harnesses/marathon_claude_code.py`` build their in-process MCP tools from the
same definitions the proxy serves and the two cannot drift.

    python -m swebench.mcp_server --http --port 8400

It used to carry ``--image`` (create a sandbox at startup) and ``--patch-dir``
(write a git diff beside it), for one topology: the claude-code harness spawned
this as a stdio subprocess, so the *server* owned the sandbox and had to extract
the answer because the harness could not. That inverted ownership cost more than
it saved -- the sandbox died when the stream closed, which is when grading needs
it; the patch was written during teardown, so the harness polled a file and could
not distinguish an extraction failure from an empty diff; and nothing outside the
subprocess could snapshot the session, so checkpoints and post-hoc extraction were
unavailable on that path.

Both harnesses now own their session and extract directly, so both flags and the
``PatchWritingPool`` that implemented them are gone. A caller that needs a sandbox
creates one (``harness.execution.provision``, or an ``AshSession``) and hands the
agent a wiring bound to it.
"""

from __future__ import annotations

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
from harness.execution.server import main

__all__ = [
    "ALL_TOOLS", "EXEC_TOOLS", "EXEC_TOOLS_MULTI", "EXEC_TOOLS_SINGLE",
    "LIFECYCLE_TOOLS", "HttpMcpServer", "SandboxEntry", "SandboxPool", "Session",
    "SessionHandler", "StdioMcpServer", "main",
]


if __name__ == "__main__":
    main()
