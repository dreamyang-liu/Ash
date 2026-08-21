"""The seat that hands a rendered outcome to the model."""

from __future__ import annotations

from typing import Callable

from ....models import CommandOutcome, ToolResult
from ...pipeline import RAW_OUTPUT, CallContext, ToolInterceptor
from .render import render_outcome

__all__ = ["OutcomePresenter"]


class OutcomePresenter(ToolInterceptor):
    """Compose what the model reads from a command's reported outcome.

    The runtime executes and reports (ADR-1): a command's exit code, its two
    streams unmerged, byte counts, truncation flags. Rendering those for a reader
    is policy, and policy is code (ADR-3) — so this seat takes the renderer as a
    plain function::

        renderer(outcome: CommandOutcome) -> str | None

    Returning ``None`` keeps the runtime's own text for that call. Results with
    no outcome (a refusal, a tool that runs no command, a plain success whose
    text is just its stdout) pass through untouched.

    Mount it innermost: it composes the text, the truncation seat bounds whatever
    it composed, and the guardrail seat annotates last — so a renderer cannot
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
