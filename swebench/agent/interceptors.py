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
from .pipeline import CallContext, ToolInterceptor, ToolPipeline
from .tools import truncate_output

__all__ = ["TruncateInterceptor", "GuardrailInterceptor", "default_pipeline",
           "RAW_OUTPUT"]

#: ``ctx.metadata`` key holding the pre-truncation output. The agent loop reads
#: it so its trace keeps recording what the runtime actually returned, with the
#: bounded text reported separately as the model's observation.
RAW_OUTPUT = "raw_output"


class TruncateInterceptor(ToolInterceptor):
    """Elide the middle of oversized tool output on the way out.

    Stateless and fail-open: a bug here must never block a tool call. Runs on
    ``ToolResult.output`` — i.e. before the loop's ``Error:`` prefix — so the
    bound applies to what the tool produced, and error text is never the part
    that gets elided.

    The untruncated output is preserved in ``ctx.metadata[RAW_OUTPUT]``:
    truncation exists to protect the model's context, not to discard evidence,
    and the trace records ground truth.
    """

    fail_mode = "open"

    def __init__(self, max_len: int = 12000) -> None:
        self.max_len = max_len

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        if len(result.output) <= self.max_len:
            return result
        ctx.metadata[RAW_OUTPUT] = result.output
        return ToolResult(success=result.success,
                          output=truncate_output(result.output, self.max_len),
                          error=result.error)


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
