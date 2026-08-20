"""Tool-level guardrails: nudge the agent away from common failure patterns.

One kernel, one mounting — the same split as Waggle (docs/ARCHITECTURE.md,
ADR-2):

``GuardrailState``
    The kernel. Tracks, per ``(agent_id, sandbox_id)``, which files have been
    read and how many times each has been edited without running tests.
    Thread-safe: one instance is shared by every agent on a pipeline.

``GuardrailInterceptor``
    An L2 seat on the tool-call path. ``before`` decides (warn or reject),
    ``after`` records successful reads and appends warnings to the result.
    Mounted in the MCP proxy, on a harness executor via
    ``AshSession.executor_for(pipeline=)``, or via ``--plugins``.

Enforcement is a parameter, not a fork. ``enforcement="warn"`` appends advisory
text (the agent loop's historical behavior, fail-open); ``enforcement="reject"``
refuses the call outright. Waggle's ``require_read`` is the same rule with
teeth — when both are mounted, give this one ``"warn"`` and let Waggle reject,
or set ``require_read=False`` on Waggle and reject here.
"""

from __future__ import annotations

import threading
from typing import Literal, Optional

from ..models import ToolResult
from .pipeline import CallContext, Continue, Reject, ToolInterceptor, Verdict
from .tools import EDIT_COMMANDS

__all__ = ["GuardrailState", "GuardrailInterceptor",
           "TEST_MARKERS", "EDIT_STREAK_LIMIT"]

TEST_MARKERS = ("pytest", "test_", "assert")
EDIT_STREAK_LIMIT = 3


def _read_before_edit_warning(path: str) -> str:
    return (f"[Warning] You are editing {path} without reading it first. "
            f"Use text_editor(view) first to see the current content.")


def _edit_streak_warning(path: str, count: int) -> str:
    return (f"[Warning] This is edit #{count} to {path} "
            f"without running tests. Consider testing before making more changes.")


class GuardrailState:
    """Read/edit bookkeeping, keyed by ``(agent_id, sandbox_id)``.

    One instance per interceptor, shared across agents — hence the keying:
    files A read must never excuse B's blind edit. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._files_read: dict[tuple[str, str], set[str]] = {}
        self._edit_streak: dict[tuple[str, str], dict[str, int]] = {}

    # -- reads --------------------------------------------------------------- #

    def record_read(self, agent_id: str, sandbox_id: str, path: str) -> None:
        with self._lock:
            self._files_read.setdefault((agent_id, sandbox_id), set()).add(path)

    def has_read(self, agent_id: str, sandbox_id: str, path: str) -> bool:
        with self._lock:
            return path in self._files_read.get((agent_id, sandbox_id), ())

    # -- edit streaks -------------------------------------------------------- #

    def record_edit(self, agent_id: str, sandbox_id: str, path: str) -> int:
        """Count this edit and return the streak length since the last test run."""
        with self._lock:
            streaks = self._edit_streak.setdefault((agent_id, sandbox_id), {})
            streaks[path] = streaks.get(path, 0) + 1
            return streaks[path]

    def reset_edits(self, agent_id: str, sandbox_id: str) -> None:
        with self._lock:
            self._edit_streak.pop((agent_id, sandbox_id), None)

    def dump(self) -> dict:
        """JSON-friendly snapshot, for audit symmetry with ``Waggle.dump()``."""
        with self._lock:
            return {
                f"{agent}:{sbx}": {
                    "files_read": sorted(paths),
                    "edit_streak": dict(self._edit_streak.get((agent, sbx), {})),
                }
                for (agent, sbx), paths in self._files_read.items()
            }


def _is_read(tool_name: str, args: dict) -> bool:
    return tool_name == "text_editor" and args.get("command") == "view"


def _is_edit(tool_name: str, args: dict) -> bool:
    return tool_name == "text_editor" and args.get("command") in EDIT_COMMANDS


def _is_test_run(tool_name: str, args: dict) -> bool:
    if tool_name != "shell":
        return False
    command = args.get("command", "")
    return any(marker in command for marker in TEST_MARKERS)


class GuardrailInterceptor(ToolInterceptor):
    """Read-before-edit and edit-streak guardrails as an L2 interceptor.

    Advisory by default (``fail_mode="open"``): warnings ride out on
    ``ToolResult.output`` and a broken guardrail never blocks work. With
    ``enforcement="reject"`` the read-before-edit rule refuses the call
    instead, and ``fail_mode`` becomes closed so a crash cannot silently
    disable the rule.

    Reads are recorded in ``after``, and only when the view succeeded —
    viewing a nonexistent file does not unlock editing it. (The agent loop's
    inline ``Guardrails.check`` recorded reads before execution, so a failed
    view counted; this mounting fixes that.)

    ``read_before_edit=False`` drops that rule and keeps only edit-streak
    nudges. Use it when ``WaggleInterceptor`` is also mounted: its
    ``require_read`` enforces the same rule with a better message (it names the
    version the agent is stale against), and two seats stating one rule tells
    the model the same thing twice.
    """

    tools = {"text_editor", "shell"}

    def __init__(self, state: Optional[GuardrailState] = None,
                 enforcement: Literal["warn", "reject"] = "warn",
                 read_before_edit: bool = True,
                 edit_streak_limit: int = EDIT_STREAK_LIMIT) -> None:
        if enforcement not in ("warn", "reject"):
            raise ValueError(f"enforcement must be 'warn' or 'reject', got {enforcement!r}")
        self.state = state or GuardrailState()
        self.enforcement = enforcement
        self.read_before_edit = read_before_edit
        self.edit_streak_limit = edit_streak_limit
        self.fail_mode = "closed" if enforcement == "reject" else "open"

    def before(self, ctx: CallContext) -> Verdict:
        if _is_test_run(ctx.tool_name, ctx.args):
            self.state.reset_edits(ctx.agent_id, ctx.sandbox_id)
            return Continue()
        if not _is_edit(ctx.tool_name, ctx.args):
            return Continue()

        path = ctx.args.get("path", "")
        warnings: list[str] = []
        if self.read_before_edit and path and \
                not self.state.has_read(ctx.agent_id, ctx.sandbox_id, path):
            if self.enforcement == "reject":
                return Reject(_read_before_edit_warning(path))
            warnings.append(_read_before_edit_warning(path))

        count = self.state.record_edit(ctx.agent_id, ctx.sandbox_id, path)
        if count >= self.edit_streak_limit:
            warnings.append(_edit_streak_warning(path, count))
        if warnings:
            # Stashed for `after`: the warning belongs on the result the model
            # sees, not on this call's arguments.
            ctx.metadata.setdefault("guardrail_warnings", []).extend(warnings)
        return Continue()

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        if result.success and _is_read(ctx.tool_name, ctx.args):
            path = ctx.args.get("path")
            if path:
                self.state.record_read(ctx.agent_id, ctx.sandbox_id, path)
        warnings = ctx.metadata.pop("guardrail_warnings", None)
        if not warnings:
            return result
        return ToolResult(success=result.success,
                          output=_append_warnings(result.output, warnings),
                          error=result.error)

    def dump(self) -> dict:
        return self.state.dump()


def _append_warnings(output: str, warnings: list[str]) -> str:
    text = "\n".join(warnings)
    return f"{output}\n\n{text}" if output else text
