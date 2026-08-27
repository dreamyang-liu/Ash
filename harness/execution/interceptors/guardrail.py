"""Read-before-edit and edit-streak nudges.

Three parts, kept together because they are one rule: what counts as a read or an
edit, the per-agent state that remembers it, and the interceptor that acts on it.

Enforcement is a parameter, not a fork. ``enforcement="warn"`` appends advisory
text (fail-open); ``enforcement="reject"`` refuses the call. State is keyed by
``(agent_id, sandbox_id)``: two agents in one sandbox must not satisfy each
other's read-before-edit, and one agent across two sandboxes has not read a path
in the second just because it read it in the first.
"""

from __future__ import annotations

import threading

from harness.execution.pipeline import (RAW_OUTPUT, CallContext, Continue,
                                        Reject, ToolInterceptor, Verdict)
from harness.core.result import ToolResult
from harness.execution.tool_constants import (CONTENT_EDIT_COMMANDS,
                                              EDIT_COMMANDS)

__all__ = ["GuardrailInterceptor", "GuardrailState", "EDIT_STREAK_LIMIT",
           "TEST_MARKERS", "is_read", "is_edit", "is_content_edit",
           "is_test_run"]


# --- what counts as what ---------------------------------------------------
#: What a command has to mention to count as running the tests. Crude on purpose:
#: the streak counter only needs to know the agent has stopped editing and gone to
#: check, and a missed match costs one extra nudge rather than a wrong decision.
TEST_MARKERS = ("pytest", "test_", "assert")


def _command(tool_name: str, args: dict) -> str:
    """This call's text_editor command, or ``''``.

    A model can put anything in ``args`` — a list, a dict, a number. Membership
    tests against a frozenset raise ``TypeError`` on unhashable values, and in
    reject mode a fail-closed crash would turn malformed model output into a
    policy refusal. Normalize once, here, so junk simply does not match.
    """
    if tool_name != "text_editor":
        return ""
    command = args.get("command")
    return command if isinstance(command, str) else ""


def is_read(tool_name: str, args: dict) -> bool:
    return _command(tool_name, args) == "view"


def is_edit(tool_name: str, args: dict) -> bool:
    return _command(tool_name, args) in EDIT_COMMANDS


def is_content_edit(tool_name: str, args: dict) -> bool:
    """An edit to existing content — see ``tools.CONTENT_EDIT_COMMANDS``."""
    return _command(tool_name, args) in CONTENT_EDIT_COMMANDS


def is_test_run(tool_name: str, args: dict) -> bool:
    if tool_name != "shell":
        return False
    command = args.get("command", "")
    return any(marker in command for marker in TEST_MARKERS)


# --- the state a rule needs ------------------------------------------------
import threading



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
        """JSON-friendly snapshot, so an audit can read this interceptor's state.

        Keyed over reads *and* streaks: an agent that only ever edited blindly
        has no read entry, and it is exactly the behavior this audit exists to
        surface.
        """
        with self._lock:
            keys = set(self._files_read) | set(self._edit_streak)
            return {
                f"{agent}:{sbx}": {
                    "files_read": sorted(self._files_read.get((agent, sbx), ())),
                    "edit_streak": dict(self._edit_streak.get((agent, sbx), {})),
                }
                for agent, sbx in sorted(keys)
            }


# --- the interceptor -------------------------------------------------------
from typing import Literal, Optional

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
    nudges. Use it when a coordination interceptor is also mounted: its
    ``require_read`` enforces the same rule with a better message (it names the
    version the agent is stale against), and two interceptors stating one rule tells
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
        # `outcome` is carried through: an interceptor that annotates text must not
        # destroy the structured report an interceptor further out still wants to read.
        # Dropping it was invisible while this was the outermost interceptor -- nothing
        # downstream looked -- and became reachable the moment `extra=` let a
        # caller mount their own interceptor outside it.
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
