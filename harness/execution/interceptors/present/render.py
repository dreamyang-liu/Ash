"""The default rendering of a command's outcome.

Replaceable by construction: `OutcomePresenter(my_renderer)` takes any
`(CommandOutcome) -> str | None`, and returning None keeps the runtime's own text.
"""

from __future__ import annotations

from harness.core.result import CommandOutcome

__all__ = ["render_outcome"]


def render_outcome(outcome: CommandOutcome) -> str:
    """Turn a command's outcome into text for a model — the default rendering.

    Sections come from the separate streams, so a command printing something that
    looks like a divider cannot fake one. Exit status is stated from the number
    rather than implied, because a silent failure has nothing else to show.
    """
    parts = []
    if outcome.stdout:
        parts.append(outcome.stdout.rstrip("\n"))
    if outcome.stderr:
        parts.append("--- stderr ---\n" + outcome.stderr.rstrip("\n"))
    if outcome.running:
        parts.append("[still running]")
    elif outcome.timed_out:
        parts.append("[timed out]")
    elif outcome.exit_code:
        parts.append(f"[exit {outcome.exit_code}]")
    if outcome.truncated:
        parts.append(f"[output clipped — {outcome.stdout_bytes} bytes on stdout, "
                     f"{outcome.stderr_bytes} on stderr. Narrow the command, or "
                     f"use `tail`/`grep` to select what you need]")
    return "\n".join(parts)
