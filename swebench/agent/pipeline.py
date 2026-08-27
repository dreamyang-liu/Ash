"""Backwards-compatible re-export.

The interceptor framework moved to :mod:`harness.execution.pipeline` — it is
benchmark-agnostic (an onion around one ``executor(tool_name, args)`` seam) and
is now mounted by the harness slot layer as well as by this package.

Everything is re-exported as the *same* objects, so ``isinstance`` checks and
``PIPELINE = [...]`` plugin files written against the old path keep working.
"""

from harness.execution.pipeline import (  # noqa: F401
    EXECUTOR,
    RAW_ERROR,
    RAW_OUTPUT,
    CallContext,
    Continue,
    Executor,
    Reject,
    Rewrite,
    ShortCircuit,
    ToolInterceptor,
    ToolPipeline,
    Verdict,
    load_pipeline,
    mounted_pipeline,
    piped_executor,
)

__all__ = [
    "EXECUTOR", "RAW_ERROR", "RAW_OUTPUT", "CallContext", "Continue", "Executor",
    "Reject", "Rewrite", "ShortCircuit", "ToolInterceptor", "ToolPipeline",
    "Verdict", "load_pipeline", "mounted_pipeline", "piped_executor",
]
