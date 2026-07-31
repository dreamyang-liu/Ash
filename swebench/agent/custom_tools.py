"""Compatibility shim: custom tools now live in the SDK (ash_sandbox.toolset).

Importers (verified via grep): agent/__init__.py (plan_custom_tool),
agent/tools.py (CUSTOM_TOOL_SPECS), runner.py (load_custom_tools,
custom_agent_schemas), tests/test_custom_tools.py, tests/test_custom_tool_
dispatch.py. Public surface preserved on top of a process-default
ToolRegistry; manifest schema unchanged (configs/custom_tools/README.md).
Moved per user instruction: "tool 之类的可以用DSL 或者data 配置项来做"
(SDK owns tool data; AshAgent is pure policy).
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
