"""Interceptors migrated out of the agent loop, plus the loop's default chain.

Before this module the agent loop applied its cross-cutting concerns inline:
guardrails via a hand-rolled ``Guardrails.check`` call, output truncation via a
``result_processors`` hook. Both only ever protected the litellm loop — an agent
arriving through the MCP proxy (claude-code) got neither. Expressed as
interceptors they work at all three mount points (docs/ARCHITECTURE.md, ADR-2):
the proxy, ``AshSession.executor_for(pipeline=)``, and ``--plugins``.

    GuardrailInterceptor   read-before-edit + edit-streak nudges (guardrails.py)
    TruncateInterceptor    bound one tool result's size
    OutcomePresenter       turn a command's reported outcome into prose

``default_pipeline()`` is what ``AshAgent`` mounts when its caller does not
supply a chain — the loop's historical behavior, now assembled from seats.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..models import CommandOutcome, ToolResult
from .guardrails import GuardrailInterceptor, GuardrailState
from .pipeline import (
    RAW_ERROR,
    RAW_OUTPUT,
    CallContext,
    ToolInterceptor,
    ToolPipeline,
)
from .tools import truncate_output

__all__ = ["TruncateInterceptor", "GuardrailInterceptor", "OutcomePresenter",
           "render_outcome", "default_pipeline", "RAW_OUTPUT", "RAW_ERROR"]


class TruncateInterceptor(ToolInterceptor):
    """Bound the size of one tool result on the way out.

    Stateless and fail-open: a bug here must never block a tool call.

    Both ``output`` and ``error`` are bounded, and against their *combined*
    size. A failing tool sets them independently — every executor in this repo
    reports a failure as ``ToolResult(success=False, output=X, error=X)`` — and
    the agent loop shows the model ``f"Error: {error}\\n{output}"``. Bounding
    each field separately would let a failing command through at twice the
    budget; bounding only ``output`` would let it through whole, since ``error``
    carries the same bytes. A failing `pytest` is the common case here, so this
    is the path that matters most.

    Whatever gets rewritten is preserved in ``ctx.metadata`` (``RAW_OUTPUT`` /
    ``RAW_ERROR``): truncation exists to protect the model's context, not to
    discard evidence, and the trace records ground truth.
    """

    fail_mode = "open"

    def __init__(self, max_len: int = 12000) -> None:
        self.max_len = max_len

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        error = result.error or ""
        if len(result.output) + len(error) <= self.max_len:
            return result

        # Split the budget the way the loop presents the result: the error is
        # the headline on a failure, so it is bounded first and keeps what it
        # needs, and the output lives within the remainder.
        error_budget = min(len(error), self.max_len // 2) if error else 0
        new_error = truncate_output(error, error_budget) if error else result.error
        new_output = truncate_output(result.output,
                                    max(self.max_len - error_budget, 1))

        # setdefault: RAW_* hold what the RUNTIME returned. An inner seat (the
        # presenter) may already have recorded that before rewriting; the text
        # this seat received would then be a rewrite, not ground truth.
        if new_output != result.output:
            ctx.metadata.setdefault(RAW_OUTPUT, result.output)
        if new_error != result.error:
            ctx.metadata.setdefault(RAW_ERROR, result.error)
        return ToolResult(success=result.success, output=new_output,
                          error=new_error, outcome=result.outcome)


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


def default_pipeline(guardrail_state: Optional[GuardrailState] = None,
                     max_output_len: int = 12000,
                     read_before_edit: bool = True,
                     renderer: Callable[[CommandOutcome], "str | None"] = render_outcome,
                     ) -> ToolPipeline:
    """The agent loop's default chain: present, bound, nudge.

    Order is semantics, read innermost-out: the presenter turns a reported
    outcome into prose, truncation bounds what it produced, and guardrails
    annotate last — a warning is appended after truncation and therefore survives
    it (it would otherwise be elided along with the output's middle).

    Guardrails are advisory here; rejection is Waggle's job when coordination is
    mounted. Pass ``read_before_edit=False`` when composing this chain with
    ``WaggleInterceptor``, which enforces that rule itself. Pass ``renderer`` to
    show commands to the model differently.
    """
    return ToolPipeline([
        GuardrailInterceptor(state=guardrail_state, enforcement="warn",
                             read_before_edit=read_before_edit),
        TruncateInterceptor(max_len=max_output_len),
        OutcomePresenter(renderer),
    ])
