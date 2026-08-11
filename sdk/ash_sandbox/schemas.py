"""Provider-specific rendering of tool schemas.

One converter serves both sources of tools: the builtin ones the runtime
reports through tools/list, and the manifest-defined ones a ToolRegistry
compiles. A single implementation is what lets custom tools appear in every
provider format -- they used to be hardcoded to OpenAI's shape, so an
Anthropic-facing harness silently got the wrong field names.

Callers: sandbox.tool_schemas (the complete panel) and
toolset.CustomToolSpec.agent_schema (one custom tool).
"""

from __future__ import annotations

SUPPORTED_FORMATS = ("openai", "anthropic", "raw")


def render(name: str, description: str, parameters: dict,
           format: str = "openai") -> dict:
    """Render one tool's schema in a provider's shape.

    `parameters` is the JSON-Schema object describing the call arguments;
    providers disagree only on what to call it and how deeply to nest it.
    """
    if format == "raw":
        return {"name": name, "description": description, "inputSchema": parameters}
    if format == "anthropic":
        return {"name": name, "description": description, "input_schema": parameters}
    if format == "openai":
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
    raise ValueError(
        f"unsupported schema format: {format!r} (expected one of {SUPPORTED_FORMATS})"
    )


def render_runtime_tool(tool: dict, format: str = "openai") -> dict:
    """Render a tool as the runtime reported it in tools/list."""
    return render(
        tool["name"],
        tool.get("description", ""),
        tool.get("inputSchema", {"type": "object", "properties": {}}),
        format,
    )
