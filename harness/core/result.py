"""Generic tool-result types — the core seam of the whole repo.

``executor(tool_name, args) -> ToolResult`` is the one interface every layer
agrees on, so these types belong to the execution plane rather than to any
benchmark. ``swebench.models`` re-exports them for backwards compatibility;
both names are the *same class object*, so isinstance checks are unaffected.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolResult:
    """Result of one tool call.

    ``output`` and ``error`` are alternatives, not two views of one message:
    ``output`` is what the tool produced, ``error`` is the tool refusing to
    produce anything (bad arguments, no sandbox, a transport failure). A command
    that ran and exited non-zero has real output and no error -- ``success``
    already says it failed.

    Storing one message in both fields costs twice, because the agent loop
    renders a failure as ``f"Error: {error}\\n{output}"``: the model reads the
    same bytes twice and pays for both.
    """
    success: bool
    output: str
    error: Optional[str] = None
    #: What running a command produced, when the tool ran one and had something
    #: to report beyond its stdout (``ash_sandbox.CommandOutcome`` fields:
    #: ``exit_code``, ``stdout``, ``stderr``, ``timed_out``, byte counts,
    #: truncation flags). ``None`` for a plain success, a refusal, or a tool that
    #: runs no command. Interceptors read it to compose what the model sees; see
    #: ``agent/interceptors/``.
    outcome: Optional["CommandOutcome"] = None

    @classmethod
    def from_sdk(cls, result) -> "ToolResult":
        """Convert an ``ash_sandbox.ToolResult``.

        The text slot is one string plus an ``is_error`` flag, so a failed call
        gives us no way to know whether that string is output or a refusal. Treat
        it as output: it is what the runtime chose to show, and ``success=False``
        carries the failure without duplicating the text.

        A command's structured outcome rides along unrendered — turning it into
        prose is presentation, and presentation belongs to the interceptors
        (docs/ARCHITECTURE.md, ADR-2), not to a type conversion.
        """
        return cls(success=not result.is_error, output=result.output,
                   outcome=CommandOutcome.from_sdk(result))


@dataclass
class CommandOutcome:
    """What running a command produced — the runtime's shared schema.

    Reported the same way by ``shell`` in the foreground and ``process read`` on
    a background pid, so code that reasons about a command works with either.

    ``exit_code`` is ``None`` when unknown: still running, or never run. Test it
    with ``is None``; ``0`` is a real answer.
    """
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    running: bool = False
    timed_out: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    @classmethod
    def from_sdk(cls, result) -> "Optional[CommandOutcome]":
        """Lift an ``ash_sandbox.ToolResult``'s outcome fields, or None."""
        if getattr(result, "stdout", None) is None:
            return None          # no command outcome in this response
        return cls(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            running=result.running,
            timed_out=result.timed_out,
            stdout_bytes=result.stdout_bytes or 0,
            stderr_bytes=result.stderr_bytes or 0,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )


