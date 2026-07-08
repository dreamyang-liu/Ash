from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from typing import Any

from swebench.agent.tools import TOOLS_SCHEMA

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
]

EXPECTED_REQUIRED = {
    "shell": ["command"],
    "text_editor": ["command", "path"],
    "grep_files": ["pattern"],
    "process": ["pid", "action"],
    "web_fetch": ["url"],
    "web_search": ["query"],
}

FORBIDDEN_SCHEMA_KEYS = {
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "patternProperties",
    "dependencies",
    "dependentRequired",
}


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


def _walk_schema(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_schema(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_schema(child)


def test_agent_tool_schema_matches_runtime_tool_surface():
    names = [tool["function"]["name"] for tool in TOOLS_SCHEMA]

    assert names == EXPECTED_TOOL_NAMES
    assert "read_file" not in names


def test_agent_tool_schemas_stay_model_compatible_plain_objects():
    for tool in TOOLS_SCHEMA:
        assert tool["type"] == "function"
        function = tool["function"]
        parameters = function["parameters"]

        assert parameters["type"] == "object"
        assert isinstance(parameters.get("properties"), dict)
        assert set(parameters.get("required", [])) <= set(parameters["properties"])

        for node in _walk_schema(parameters):
            forbidden = FORBIDDEN_SCHEMA_KEYS & set(node)
            assert not forbidden, f"{function['name']} uses unsupported schema keys: {sorted(forbidden)}"


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
