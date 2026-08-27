"""Tool-level constants and helpers shared by the execution plane.

These describe the *runtime's* tool protocol (which `text_editor` commands
mutate, how to bound one result), not any benchmark, so they belong beside the
interceptors that reason about them. ``swebench.agent.tools`` re-exports them.
"""

from __future__ import annotations

#: text_editor commands that modify a file. The single source of truth for
#: "this call is an edit", shared by everything that has to reason about it:
#: the guardrails, and any coordination interceptor mounted as a plugin.
EDIT_COMMANDS = frozenset({"str_replace", "insert", "write"})

#: Edits that rewrite *existing* content, so "you did not read this file first"
#: is unambiguous. `write` is excluded on purpose: it also creates files, and
#: telling creation from overwrite needs a filesystem probe. An interceptor that
#: can afford that probe may refuse blind overwrites; one that cannot pays for
#: that probe and refuses only blind overwrites (`_write_unregistered`); a rule
#: that cannot afford the probe must not claim to cover `write`, or creating a
#: new file becomes an unsatisfiable warning — or, when enforced, impossible.
CONTENT_EDIT_COMMANDS = frozenset({"str_replace", "insert"})


def truncate_output(content: str, max_len: int = 12000) -> str:
    """Elide the middle of overly long tool output."""
    if len(content) <= max_len:
        return content
    head, tail = max_len * 2 // 3, max_len // 3  # ~8000 / ~4000 chars
    elided = len(content) - head - tail
    return (
        content[:head] +
        f"\n\n... [{elided} characters truncated — output too long. Use `tail` on shell "
        f"commands, `limit` on grep, or pipe through `grep` for targeted output] ...\n\n" +
        content[-tail:]
    )
