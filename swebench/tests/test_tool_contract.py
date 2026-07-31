from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from typing import Any

from swebench.agent.tools import AGENT_TOOL_ROUTES, TOOLS_SCHEMA, route_agent_tool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

from ash_sandbox import Sandbox  # noqa: E402
from ash_sandbox.backends import Backend  # noqa: E402
from ash_sandbox.result import ToolResult  # noqa: E402


EXPECTED_TOOL_NAMES = [
    "shell",
    "text_editor",
    "grep_files",
    "process",
    "web_fetch",
    "web_search",
    "wait_for_events",
]

RUNTIME_TOOL_NAMES = {
    "shell",
    "text_editor",
    "grep_files",
    "process",
    "web_fetch",
    "web_search",
    "wait_for_events",
}

EXPECTED_REQUIRED = {
    "shell": ["command"],
    "text_editor": ["command", "path"],
    "grep_files": ["pattern"],
    "process": ["pid", "action"],
    "web_fetch": ["url"],
    "web_search": ["query"],
    "wait_for_events": [],
}

ROOT_SCHEMA_KEYS = {"type", "properties", "required", "description"}
FIELD_SCHEMA_KEYS = {"type", "items", "enum", "default", "description"}
SUPPORTED_SCHEMA_TYPES = {"object", "array", "string", "integer", "boolean"}


def _function_by_name() -> dict[str, dict[str, Any]]:
    return {tool["function"]["name"]: tool["function"] for tool in TOOLS_SCHEMA}


def _runtime_tools_from_agent_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "inputSchema": tool["function"]["parameters"],
        }
        for tool in TOOLS_SCHEMA
    ]


def _assert_supported_root_schema(tool_name: str, schema: dict[str, Any]):
    extra = set(schema) - ROOT_SCHEMA_KEYS
    assert not extra, f"{tool_name} root schema uses unsupported keys: {sorted(extra)}"

    assert schema.get("type") == "object"
    assert isinstance(schema.get("properties"), dict)
    assert set(schema.get("required", [])) <= set(schema["properties"])

    for property_name, field_schema in schema["properties"].items():
        _assert_supported_field_schema(tool_name, property_name, field_schema)


def _assert_supported_field_schema(tool_name: str, path: str, schema: dict[str, Any]):
    extra = set(schema) - FIELD_SCHEMA_KEYS
    assert not extra, f"{tool_name}.{path} uses unsupported schema keys: {sorted(extra)}"

    schema_type = schema.get("type")
    assert schema_type in SUPPORTED_SCHEMA_TYPES, f"{tool_name}.{path} has unsupported type: {schema_type}"

    if "enum" in schema:
        assert isinstance(schema["enum"], list)
        assert schema["enum"], f"{tool_name}.{path} enum must not be empty"

    if schema_type == "array":
        assert "items" in schema, f"{tool_name}.{path} array schema requires items"
        items = schema["items"]
        assert isinstance(items, dict), f"{tool_name}.{path}.items must be an object schema"
        _assert_supported_field_schema(tool_name, f"{path}.items", items)
    else:
        assert "items" not in schema, f"{tool_name}.{path} non-array schema must not define items"


RUNTIME_REQUIRED = {
    "shell": ["command"],
    "text_editor": ["command", "path"],
    "grep_files": ["pattern"],
    "process": ["pid", "action"],
    "web_fetch": ["url"],
    "web_search": ["query"],
    "wait_for_events": [],
}


def test_agent_tool_schema_exposes_current_surface():
    names = [tool["function"]["name"] for tool in TOOLS_SCHEMA]

    assert names == EXPECTED_TOOL_NAMES
    assert "read_file" not in names


def test_agent_tool_routes_are_explicit_and_runtime_routeable():
    functions = _function_by_name()

    assert set(AGENT_TOOL_ROUTES) == set(functions)
    assert set(AGENT_TOOL_ROUTES.values()) <= RUNTIME_TOOL_NAMES

    for agent_tool, runtime_tool in AGENT_TOOL_ROUTES.items():
        agent_required = set(functions[agent_tool]["parameters"].get("required", []))
        runtime_required = set(RUNTIME_REQUIRED[runtime_tool])
        if agent_tool == runtime_tool:
            assert agent_required >= runtime_required


def test_route_agent_tool_identity_routes_return_runtime_call_copy():
    args = {"command": "echo hi"}

    runtime_tool, runtime_args = route_agent_tool("shell", args)

    assert runtime_tool == "shell"
    assert runtime_args == args
    assert runtime_args is not args


def test_route_agent_tool_rejects_unknown_tool():
    try:
        route_agent_tool("missing", {})
    except KeyError as exc:
        assert "unknown agent tool: missing" in str(exc)
    else:
        raise AssertionError("expected unknown agent tool to be rejected")


def test_agent_tool_schemas_use_supported_plain_object_subset():
    for tool in TOOLS_SCHEMA:
        assert tool["type"] == "function"
        function = tool["function"]
        _assert_supported_root_schema(function["name"], function["parameters"])


def test_command_specific_text_editor_contract_is_described_not_encoded_with_oneof():
    text_editor = _function_by_name()["text_editor"]
    parameters = text_editor["parameters"]
    properties = parameters["properties"]

    assert parameters["required"] == ["command", "path"]
    assert properties["command"]["enum"] == ["view", "str_replace", "insert", "write"]
    assert "Required for str_replace" in properties["old_str"]["description"]
    assert "Required for insert" in properties["insert_text"]["description"]
    assert "Required for write" in properties["file_text"]["description"]


def test_tool_required_fields_are_stable():
    functions = _function_by_name()

    assert set(functions) == set(EXPECTED_REQUIRED)
    for name, required in EXPECTED_REQUIRED.items():
        assert functions[name]["parameters"]["required"] == required


def test_process_and_shell_descriptions_match_runtime_log_contract():
    functions = _function_by_name()

    assert "current bounded output snapshot" in functions["process"]["description"]
    assert "max_output_bytes" in functions["shell"]["parameters"]["properties"]
    assert (
        "first 40% and last 60%"
        in functions["shell"]["parameters"]["properties"]["max_output_bytes"]["description"]
    )


class FakeBackend(Backend):
    def __init__(self, tools: list[dict[str, Any]]):
        self.tools = tools
        self.list_calls = 0

    async def call(self, tool_name: str, args: dict) -> ToolResult:
        return ToolResult(output=f"{tool_name}:{args}", is_error=False)

    async def list_tools(self) -> list[dict]:
        self.list_calls += 1
        return copy.deepcopy(self.tools)


async def _collect_sdk_formats():
    backend = FakeBackend(_runtime_tools_from_agent_schema())
    sandbox = Sandbox(backend=backend)

    raw = await sandbox.tool_schemas(format="raw")
    openai = await sandbox.tool_schemas(format="openai")
    anthropic = await sandbox.tool_schemas(format="anthropic")

    return raw, openai, anthropic, backend.list_calls


def test_sdk_tool_schema_formats_preserve_contract_shape():
    raw, openai, anthropic, list_calls = asyncio.run(_collect_sdk_formats())

    assert list_calls == 1
    assert [tool["name"] for tool in raw] == EXPECTED_TOOL_NAMES

    assert [tool["function"]["name"] for tool in openai] == EXPECTED_TOOL_NAMES
    for tool in openai:
        function = tool["function"]
        assert tool["type"] == "function"
        assert function["parameters"]["type"] == "object"
        assert function["parameters"]["required"] == EXPECTED_REQUIRED[function["name"]]

    assert [tool["name"] for tool in anthropic] == EXPECTED_TOOL_NAMES
    for tool in anthropic:
        assert tool["input_schema"]["type"] == "object"
        assert tool["input_schema"]["required"] == EXPECTED_REQUIRED[tool["name"]]


def test_sdk_sandbox_exposes_managed_sandbox_identity():
    sandbox = Sandbox(backend=FakeBackend([]))
    assert sandbox.sandbox_id is None

    sandbox._container_id = "sandbox-123"
    assert sandbox.sandbox_id == "sandbox-123"
