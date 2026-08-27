"""Backwards-compatible re-export.

Moved to :mod:`harness.execution.backends`: choosing where sandboxes come from
(docker / microvm / k8s) is execution-plane configuration, not a benchmark
concern. Imported from here by the harnesses, ``AshSession`` and the rollout
server; the names below are the same objects, so nothing observable changes.
"""

from harness.execution.backends import (  # noqa: F401
    BACKENDS,
    DEFAULT_BACKEND,
    BackendError,
    backend_config,
    backend_name,
    build_pool,
)

__all__ = ["BACKENDS", "DEFAULT_BACKEND", "BackendError", "backend_config",
           "backend_name", "build_pool"]
