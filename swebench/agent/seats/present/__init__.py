"""Composing what the model reads from what the runtime reported.

The runtime executes and reports (ADR-1); turning `exit_code`, two unmerged
streams and byte counts into prose is policy, and policy is code (ADR-3). So the
seat takes a plain function and this package ships one default implementation of
it -- `render_outcome` is a starting point to replace, not the only rendering.
"""

from .interceptor import OutcomePresenter
from .render import render_outcome

__all__ = ["OutcomePresenter", "render_outcome"]
