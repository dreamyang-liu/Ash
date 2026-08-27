"""Compose the model's text from the runtime's structured report.

The runtime returns a ``CommandOutcome`` (exit code, streams, truncation flags);
a model reads prose. This is the only place that turns one into the other, so how
a tool call *looks* to the model is one decision rather than several.
"""

from __future__ import annotations

from harness.core.result import CommandOutcome, ToolResult
from harness.execution.pipeline import RAW_OUTPUT, CallContext, ToolInterceptor

__all__ = ["OutcomePresenter", "render_outcome"]


# --- rendering -------------------------------------------------------------
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


# --- the interceptor -------------------------------------------------------
from typing import Callable




class OutcomePresenter(ToolInterceptor):
    """Compose what the model reads from a command's reported outcome.

    The runtime executes and reports (ADR-1): a command's exit code, its two
    streams unmerged, byte counts, truncation flags. Rendering those for a reader
    is policy, and policy is code (ADR-3) — so this interceptor takes the renderer as a
    plain function::

        renderer(outcome: CommandOutcome) -> str | None

    Returning ``None`` keeps the runtime's own text for that call. Results with
    no outcome (a refusal, a tool that runs no command, a plain success whose
    text is just its stdout) pass through untouched.

    Mount it innermost: it composes the text, the truncation interceptor bounds whatever
    it composed, and the guardrail interceptor annotates last — so a renderer cannot
    hand the model more than the byte budget allows.
    """

    fail_mode = "open"

    def __init__(self, renderer: Callable[[CommandOutcome], "str | None"] = render_outcome) -> None:
        self.renderer = renderer

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        if result.outcome is None:
            return result
        text = self.renderer(result.outcome)
        if text is None or text == result.output:
            return result
        ctx.metadata.setdefault(RAW_OUTPUT, result.output)
        return ToolResult(success=result.success, output=text,
                          error=result.error, outcome=result.outcome)
