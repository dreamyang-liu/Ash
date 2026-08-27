"""Which tool calls could have changed the environment.

The interceptor half of checkpointing: it answers "did anything since the last
snapshot possibly mutate", which is what lets a read-only step reuse the previous
capture instead of paying for an identical one. *When* to snapshot is a separate
decision and lives in ``harness/execution/checkpoints.py``.

Mounted outermost, so it also sees calls the inner interceptors reject: a
rejected call changes nothing, but over-counting is cheaper than reasoning about
which rejections are total.
"""

from __future__ import annotations

from typing import Optional

from harness.execution.pipeline import CallContext, Continue, ToolInterceptor, Verdict

__all__ = ["MutationTracker", "call_mutates", "READ_ONLY_TOOLS"]


#: Tools that cannot change the environment: they read, search, or fetch.
#: Anything not listed is assumed to mutate, because a wrong "clean" verdict
#: silently loses a step's state while a wrong "dirty" one only costs a cheap
#: capture.
READ_ONLY_TOOLS = frozenset({
    "grep_files",
    "web_fetch",
    "web_search",
    "wait_for_events",
})

#: ``text_editor`` sub-commands that only read.
READ_ONLY_EDITOR_COMMANDS = frozenset({"view"})

#: ``process`` sub-commands that only read. ``kill`` changes what is running,
#: which is environment state a later step can observe.
READ_ONLY_PROCESS_COMMANDS = frozenset({"read", "peek", "list", "status"})


def call_mutates(tool_name: str, args: dict) -> bool:
    """Whether a tool call could have changed the environment.

    Conservative by construction: ``shell`` is always treated as mutating
    because a command's effects cannot be read off its text (``ls`` reads,
    ``ls > out`` writes), and a bash-only panel would otherwise have every
    step misjudged.
    """
    if tool_name in READ_ONLY_TOOLS:
        return False
    if tool_name == "text_editor":
        return str(args.get("command", "")) not in READ_ONLY_EDITOR_COMMANDS
    if tool_name == "process":
        return str(args.get("command", "")) not in READ_ONLY_PROCESS_COMMANDS
    return True


class MutationTracker(ToolInterceptor):
    """Flags whether any call since the last checkpoint could have mutated.

    Mounted outermost so it also sees calls the inner guardrails reject: a
    rejected call changes nothing, but it is cheaper to over-count than to
    reason about which rejections are total.

    Also tracks whether the episode may have live background processes: a
    disk-only checkpoint captures the filesystem but not processes, so a
    replay of a step taken while a background process ran diverges -- the
    process is gone, its pids answer errors, its unflushed output never made
    it to disk. `may_have_background` turns from best-effort bookkeeping into
    a per-record flag the replay tooling can warn about. It latches on
    `shell(background=true)` and clears only on `process(kill)` with no way
    to know *which* process died, so it over-reports -- the flag means
    "replay may diverge", not "will".
    """

    fail_mode = "open"

    def __init__(self) -> None:
        self._dirty = False
        self._background_starts = 0

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def may_have_background(self) -> bool:
        return self._background_starts > 0

    def clear(self) -> None:
        self._dirty = False

    def before(self, ctx: CallContext) -> Verdict:
        if call_mutates(ctx.tool_name, ctx.args):
            self._dirty = True
        if ctx.tool_name == "shell" and bool(ctx.args.get("background")):
            self._background_starts += 1
        elif (ctx.tool_name == "process"
              and str(ctx.args.get("command", "")) == "kill"
              and self._background_starts > 0):
            self._background_starts -= 1
        return Continue()
