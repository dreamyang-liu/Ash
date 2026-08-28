"""Backwards-compatible re-export.

Moved to :mod:`harness.execution.templates`: turning an image into a microVM
template is execution-plane machinery, not a benchmark concern -- it is what any
caller needs whose environments are raw per-instance images, and the session that
drives it now lives in the execution plane too. The names below are the same
objects, so nothing observable changes.
"""

from harness.execution.templates import (  # noqa: F401
    MAX_TEMPLATE_ATTEMPTS,
    RUNTIME_PATH,
    TemplateBuilder,
    TemplateError,
    builder_from_backend,
    template_name,
)

__all__ = ["MAX_TEMPLATE_ATTEMPTS", "RUNTIME_PATH", "TemplateBuilder",
           "TemplateError", "builder_from_backend", "template_name"]
