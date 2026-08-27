"""The interceptors that ship with Ash, and the chain they form.

One package each, because an interceptor is the unit somebody replaces: a caller
who wants different truncation, a different rendering, or a different rule writes a
class and names it in a chain. Sharing one module made that look like editing Ash's
internals; a directory apiece says what is actually true, which is that these are
three independent implementations of one interface (`pipeline.ToolInterceptor`).

    guardrail/   read-before-edit and edit-streak nudges
    truncate/    bound what one result costs the model's context
    present/     compose the model's text from the runtime's structured report

`default_pipeline()` is the assembly, and lives beside them rather than inside any
one of them -- the order is a fact about the three together.
"""

from .guardrail import (EDIT_STREAK_LIMIT, TEST_MARKERS,
                        GuardrailInterceptor, GuardrailState)
from .pipeline import default_pipeline
from .present import OutcomePresenter, render_outcome
from .truncate import TruncateInterceptor

__all__ = ["GuardrailInterceptor", "GuardrailState", "EDIT_STREAK_LIMIT",
           "TEST_MARKERS", "TruncateInterceptor", "OutcomePresenter",
           "render_outcome", "default_pipeline"]
