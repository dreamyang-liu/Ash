"""Tool-level guardrails: nudge the agent away from common failure patterns.

One kernel, one mounting (docs/ARCHITECTURE.md,
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
refuses the call outright. A coordination seat mounted below may enforce the same
rule with a better message (it can name the version the agent is stale against);
when one is, give this seat ``read_before_edit=False`` so the model is not told
the same thing twice.
"""

from __future__ import annotations

from typing import Literal, Optional

from ....models import ToolResult
from ...pipeline import (RAW_OUTPUT, CallContext, Continue, Reject,
                         ToolInterceptor, Verdict)
from .classify import is_content_edit, is_edit, is_read, is_test_run
from .state import GuardrailState

__all__ = ["GuardrailInterceptor", "EDIT_STREAK_LIMIT"]

EDIT_STREAK_LIMIT = 3
_WARNINGS = "guardrail_warnings"


def _read_before_edit_warning(path: str) -> str:
    return (f"[Warning] You are editing {path} without reading it first. "
            f"Use text_editor(view) first to see the current content.")


def _edit_streak_warning(path: str, count: int) -> str:
    return (f"[Warning] This is edit #{count} to {path} "
            f"without running tests. Consider testing before making more changes.")


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
    nudges. Use it when a coordination seat is also mounted: its
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
        if is_test_run(ctx.tool_name, ctx.args):
            self.state.reset_edits(ctx.agent_id, ctx.sandbox_id)
            return Continue()
        if not is_edit(ctx.tool_name, ctx.args):
            return Continue()

        path = ctx.args.get("path", "")
        warnings: list[str] = []
        # `write` is an edit for streak purposes but not for read-before-edit:
        # it also creates files, and creation has nothing to have read.
        if self.read_before_edit and path \
                and is_content_edit(ctx.tool_name, ctx.args) \
                and not self.state.has_read(ctx.agent_id, ctx.sandbox_id, path):
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
        if result.success and is_read(ctx.tool_name, ctx.args):
            path = ctx.args.get("path")
            if path:
                self.state.record_read(ctx.agent_id, ctx.sandbox_id, path)
        warnings = ctx.metadata.pop(_WARNINGS, None)
        if not warnings:
            return result
        # Record what the runtime returned before annotating it. A nudge is for
        # the model; the trace must still be able to report the real output
        # (and not count the warning's bytes as the tool's).
        ctx.metadata.setdefault(RAW_OUTPUT, result.output)
        # `outcome` is carried through: a seat that annotates text must not
        # destroy the structured report a seat further out still wants to read.
        # Dropping it was invisible while this was the outermost seat -- nothing
        # downstream looked -- and became reachable the moment `extra=` let a
        # caller mount their own seat outside it.
        return ToolResult(success=result.success,
                          output=_append_warnings(result.output, warnings),
                          error=result.error,
                          outcome=result.outcome)

    def dump(self) -> dict:
        return self.state.dump()


def _append_warnings(output: str, warnings: list[str]) -> str:
    """Append warnings, always behind a blank line.

    The separator is unconditional even when ``output`` is empty: the loop
    prefixes failures with ``Error: …\\n``, so a bare join would glue the
    warning to the error text.
    """
    return f"{output}\n\n" + "\n".join(warnings)
