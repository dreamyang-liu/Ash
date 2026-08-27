"""The interceptors that ship with Ash, and the chain they form.

One file each, because an interceptor is the unit somebody replaces: a caller who
wants different truncation, a different rendering, or a rule of their own writes a
class and names it in a chain.

    guardrail.py   read-before-edit and edit-streak nudges
    truncate.py    bound what one result costs the model's context
    present.py     compose the model's text from the runtime's structured report
    mutation.py    did anything since the last snapshot possibly change the env

**This list is what we happened to need, not a taxonomy.** A new interceptor does
not have to fit one of these; add a file, export the class, and mount it --
``default_pipeline(extra=[Yours()])``, or a ``--plugins`` file exporting
``PIPELINE``. Only the first three are in the default chain; ``mutation`` is
mounted by whoever turns checkpointing on, because most runs do not.

``default_pipeline()`` is the assembly and lives beside them rather than inside
any one of them -- the order is a fact about them together, not about any single
interceptor.
"""

from .guardrail import (EDIT_STREAK_LIMIT, TEST_MARKERS, GuardrailInterceptor,
                        GuardrailState)
from .mutation import MutationTracker, call_mutates
from .pipeline import default_pipeline
from .present import OutcomePresenter, render_outcome
from .truncate import TruncateInterceptor

__all__ = ["GuardrailInterceptor", "GuardrailState", "EDIT_STREAK_LIMIT",
           "TEST_MARKERS", "TruncateInterceptor", "OutcomePresenter",
           "render_outcome", "MutationTracker", "call_mutates",
           "default_pipeline"]
