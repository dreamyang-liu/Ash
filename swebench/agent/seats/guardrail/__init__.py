"""Nudging the agent away from common failure patterns.

Two rules today -- read a file before editing it, and run the tests after a run of
edits -- but the shape is what matters: the state that remembers what an agent has
read (`state.py`), the predicates that classify a call (`classify.py`), and the seat
that turns the two into a verdict (`interceptor.py`). A different rule reuses the
first two and ships its own seat.
"""

from .classify import TEST_MARKERS
from .interceptor import EDIT_STREAK_LIMIT, GuardrailInterceptor
from .state import GuardrailState

__all__ = ["GuardrailInterceptor", "GuardrailState", "EDIT_STREAK_LIMIT",
           "TEST_MARKERS"]
