"""Bounding what one tool result costs the model's context.

Its own package, because an interceptor is a unit somebody replaces. Swapping the
truncation strategy (keep the tail only, summarise instead of eliding, budget per
tool) means writing a class beside this one and naming it in a chain -- not editing
a file shared with the interceptors you are keeping.
"""

from .interceptor import TruncateInterceptor

__all__ = ["TruncateInterceptor"]
