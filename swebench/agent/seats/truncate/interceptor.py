"""The default truncation seat: elide the middle, keep both ends."""

from __future__ import annotations

from ....models import ToolResult
from ...pipeline import RAW_ERROR, RAW_OUTPUT, CallContext, ToolInterceptor
from ...tools import truncate_output

__all__ = ["TruncateInterceptor"]


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
