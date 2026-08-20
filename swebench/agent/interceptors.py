"""Interceptors migrated out of the agent loop, plus the loop's default chain.

Before this module the agent loop applied its cross-cutting concerns inline:
guardrails via a hand-rolled ``Guardrails.check`` call, output truncation via a
``result_processors`` hook. Both only ever protected the litellm loop — an agent
arriving through the MCP proxy (claude-code) got neither. Expressed as
interceptors they work at all three mount points (docs/ARCHITECTURE.md, ADR-2):
the proxy, ``AshSession.executor_for(pipeline=)``, and ``--plugins``.

    GuardrailInterceptor   read-before-edit + edit-streak nudges (guardrails.py)
    TruncateInterceptor    bound one tool result's size

``default_pipeline()`` is what ``AshAgent`` mounts when its caller does not
supply a chain — the loop's historical behavior, now assembled from seats.
"""

from __future__ import annotations

from typing import Optional

from ..models import ToolResult
from .guardrails import GuardrailInterceptor, GuardrailState
from .pipeline import (
    RAW_ERROR,
    RAW_OUTPUT,
    CallContext,
    ToolInterceptor,
    ToolPipeline,
)
from .tools import truncate_output

__all__ = ["TruncateInterceptor", "GuardrailInterceptor", "default_pipeline",
           "RAW_OUTPUT", "RAW_ERROR"]


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

        if new_output != result.output:
            ctx.metadata[RAW_OUTPUT] = result.output
        if new_error != result.error:
            ctx.metadata[RAW_ERROR] = result.error
        return ToolResult(success=result.success, output=new_output,
                          error=new_error)


def default_pipeline(guardrail_state: Optional[GuardrailState] = None,
                     max_output_len: int = 12000,
                     read_before_edit: bool = True) -> ToolPipeline:
    """The agent loop's default chain: nudge, then bound the result.

    Guardrails sit outermost so their ``before`` sees the call first and their
    ``after`` runs last — a warning is appended after truncation and therefore
    survives it (it would otherwise be elided along with the output's middle).
    Guardrails are advisory here; rejection is Waggle's job when coordination
    is mounted. Pass ``read_before_edit=False`` when composing this chain with
    ``WaggleInterceptor``, which enforces that rule itself.
    """
    return ToolPipeline([
        GuardrailInterceptor(state=guardrail_state, enforcement="warn",
                             read_before_edit=read_before_edit),
        TruncateInterceptor(max_len=max_output_len),
    ])
