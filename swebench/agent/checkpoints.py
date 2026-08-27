"""Backwards-compatible re-export.

Per-step environment checkpointing moved to
:mod:`harness.execution.checkpoints`: deciding *when* a sandbox is worth
snapshotting is execution-plane machinery (it reads tool calls, not benchmark
results), and the harness slot layer drives it for external agents too.

Same objects, so ``isinstance`` and existing call sites are unaffected.
"""

from harness.execution.checkpoints import (  # noqa: F401
    CheckpointRecord,
    Checkpointer,
    MutationTracker,
    READ_ONLY_TOOLS,
    call_mutates,
    install,
)

__all__ = ["CheckpointRecord", "Checkpointer", "MutationTracker",
           "READ_ONLY_TOOLS", "call_mutates", "install"]
