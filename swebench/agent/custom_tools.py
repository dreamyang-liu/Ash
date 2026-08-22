"""Manifest-defined tools, on top of the SDK's registry (ash_sandbox.toolset).

A thin layer over one process-default ``ToolRegistry``: the SDK owns the tool data
and the manifest schema (configs/custom_tools/README.md), and this module is where
a harness loads them and asks for their schemas.

Two halves, and they were split for a while. Dispatch is wired into the agent loop
(``is_custom_tool`` / ``plan_custom_tool``, agent/__init__.py) and always worked.
*Loading* was only ever called by runner.py -- a stale second CLI, since deleted --
so nothing put manifests into the registry and ``custom_tools_dir`` reached
``AgentConfig`` from nowhere. The feature was unreachable rather than broken, which
is the failure mode a compatibility shim invites: the names all resolve.
"""

from __future__ import annotations

from pathlib import Path

from ash_sandbox.toolset import (  # noqa: F401  (re-exports)
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    CustomToolPlan,
    CustomToolSpec,
    ManifestError,
    ParamSpec,
    ToolRegistry,
    parse_manifest,
)

# Default manifest location (repo-relative), overridable per run.
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "configs" / "custom_tools"

# Process-default registry backing the module-level API below. Harnesses
# that need per-task tool panels should construct their own ToolRegistry
# and use Sandbox.call_agent_tool(..., registry=...) instead.
DEFAULT_REGISTRY = ToolRegistry()

# Backward-compatible view of the default registry's specs.
CUSTOM_TOOL_SPECS = DEFAULT_REGISTRY.custom_specs


def register(spec: CustomToolSpec) -> None:
    """Register a custom tool on the process-default registry."""
    DEFAULT_REGISTRY.register(spec)


def load_manifests(directory: str | Path) -> list[CustomToolSpec]:
    """Load and register all manifests in a directory (default registry)."""
    return DEFAULT_REGISTRY.load_manifests(directory)


def load_custom_tools(directory: str | Path | None = None) -> list[CustomToolSpec]:
    """Load manifests from `directory`, or the default location.

    - explicit directory: must exist (typo of a user-passed path is an error)
    - default location: silently skipped when absent (opt-in feature)
    """
    if directory is not None:
        directory = Path(directory)
        if not directory.is_dir():
            raise ManifestError(f"custom tools dir not found: {directory}")
        return load_manifests(directory)
    if DEFAULT_MANIFEST_DIR.is_dir():
        return load_manifests(DEFAULT_MANIFEST_DIR)
    return []


def custom_agent_schemas() -> list[dict]:
    """Function-calling schemas for all registered custom tools."""
    return DEFAULT_REGISTRY.custom_agent_schemas()


def plan_custom_tool(name: str, args: dict) -> CustomToolPlan:
    """Resolve a custom tool call into an execution plan (default registry)."""
    return DEFAULT_REGISTRY.plan_custom_tool(name, args)
