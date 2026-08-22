"""The default chain: which interceptors, in which order.

Assembly is separate from the interceptors because the order is a property of the set,
not of any member. Read it innermost-out -- present, bound, nudge -- and the reason
is in `default_pipeline`'s docstring: a warning appended before truncation would be
elided along with the middle of the output it was appended to.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ...models import CommandOutcome
from ..pipeline import ToolInterceptor, ToolPipeline
from .guardrail import GuardrailInterceptor, GuardrailState
from .present import OutcomePresenter, render_outcome
from .truncate import TruncateInterceptor

__all__ = ["default_pipeline"]


def default_pipeline(guardrail_state: Optional[GuardrailState] = None,
                     max_output_len: int = 12000,
                     read_before_edit: bool = True,
                     renderer: Callable[[CommandOutcome], "str | None"] = render_outcome,
                     extra: "Sequence[ToolInterceptor] | None" = None,
                     ) -> ToolPipeline:
    """The agent loop's default chain: present, bound, nudge.

    Order is semantics, read innermost-out: the presenter turns a reported
    outcome into prose, truncation bounds what it produced, and guardrails
    annotate last — a warning is appended after truncation and therefore survives
    it (it would otherwise be elided along with the output's middle).

    Guardrails are advisory here. Pass ``read_before_edit=False`` when composing
    this chain with a coordination interceptor that enforces the same rule, so the model
    is not told it twice. Pass ``renderer`` to show commands differently.

    ``extra`` mounts your own interceptors *outside* the defaults — import the
    class, put an instance in the list, done::

        default_pipeline(extra=[NoDestructiveShell()])

    Outside, because that is where an interceptor can do things the inner ones cannot: it
    sees the call before truncation spends anything on it, and it
    still sees calls the inner interceptors reject, since a short circuit unwinds the
    onion through everything already entered. An interceptor that must instead
    observe the *final* text the model reads wants to be innermost, which the
    defaults are — build the list by hand for that (``ToolPipeline([...])``); this
    argument is the common case, not every case.
    """
    return ToolPipeline([
        *(extra or ()),
        GuardrailInterceptor(state=guardrail_state, enforcement="warn",
                             read_before_edit=read_before_edit),
        TruncateInterceptor(max_len=max_output_len),
        OutcomePresenter(renderer),
    ])
