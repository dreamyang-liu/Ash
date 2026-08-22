from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from typing import Any

from swebench.agent.tools import load_panel

_PANEL = load_panel()
TOOLS_SCHEMA = _PANEL.schema

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


def test_every_offered_tool_routes_to_a_real_runtime_tool():
    """The panel and the routing come from one compiled set of views, so they cannot
    disagree -- this checks the set itself lands on tools the runtime serves."""
    functions = _function_by_name()
    views = _PANEL.views

    assert set(views) == set(functions)
    assert {v.runtime_tool for v in views.values()} <= RUNTIME_TOOL_NAMES

    for name, view in views.items():
        agent_required = set(functions[name]["parameters"].get("required", []))
        runtime_required = set(RUNTIME_REQUIRED[view.runtime_tool])
        exposed_required = {r for r in runtime_required
                            if r in set(view.arguments.values())}
        assert agent_required >= exposed_required, \
            f"{name} offers {sorted(exposed_required - agent_required)} as optional"


def test_route_agent_tool_identity_routes_return_runtime_call_copy():
    args = {"command": "echo hi"}

    runtime_tool, runtime_args = _PANEL.route("shell", args)

    assert runtime_tool == "shell"
    assert runtime_args == args
    assert runtime_args is not args


def test_route_agent_tool_rejects_unknown_tool():
    try:
        _PANEL.route("missing", {})
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
    # The head/tail split is described by truncate_mode; max_output_bytes
    # only carries the total budget.
    assert "truncate_mode" in functions["shell"]["parameters"]["properties"]
    assert (
        "first 40% and last 60%"
        in functions["shell"]["parameters"]["properties"]["truncate_mode"]["description"]
    )


class FakeBackend(Backend):
    def __init__(self, tools: list[dict[str, Any]]):
        self.tools = tools
        self.list_calls = 0

    async def call(self, tool_name: str, args: dict, agent_id: str = "") -> ToolResult:
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


# --------------------------------------------------------------------------- #
#  The hand-written schema vs what the runtime actually serves
# --------------------------------------------------------------------------- #

#: Runtime tools deliberately absent from the model's panel, and why. `artifact`
#: fetches and verifies a binary; a manifest-defined tool is expanded into
#: artifact+shell by the SDK (toolset.CustomToolPlan), so the model calls the tool
#: by its own name and never this. Listing it would offer a download primitive as a
#: first-class action.
NOT_OFFERED_TO_THE_MODEL = {"artifact"}


def _runtime_tool_names() -> "set[str] | None":
    """Ask a built ash-runtime what it serves. None if no binary is available."""
    import json
    import shutil
    import subprocess

    binary = shutil.which("ash-runtime")
    for candidate in ("/tmp/ash-rt", "/tmp/serve/ash-runtime",
                      str(Path(__file__).resolve().parents[2] / "runtime" / "ash-runtime")):
        if binary:
            break
        if Path(candidate).is_file():
            binary = candidate
    if not binary:
        return None

    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    try:
        proc = subprocess.run([binary, "--mode", "stdio"], input=request + "\n",
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in proc.stdout.splitlines():
        try:
            tools = json.loads(line).get("result", {}).get("tools")
        except json.JSONDecodeError:
            continue
        if tools:
            return {t["name"] for t in tools}
    return None


def test_the_agent_panel_matches_what_the_runtime_serves():
    """CLAUDE.md requires the runtime, the SDK and this schema to move together, and
    until now nothing checked it: the contract test restated its own hardcoded list,
    so a tool added to the runtime and forgotten here would go unnoticed.

    Skipped rather than failed when no binary is around -- this needs `go build` --
    but it runs in CI, where one is.
    """
    import pytest

    served = _runtime_tool_names()
    if served is None:
        pytest.skip("no ash-runtime binary; run `cd runtime && go build -o ash-runtime .`")

    offered = {t["function"]["name"] for t in TOOLS_SCHEMA}
    missing = served - offered - NOT_OFFERED_TO_THE_MODEL
    invented = offered - served

    assert not missing, (
        f"the runtime serves {sorted(missing)} and the agent panel does not offer "
        f"them; add them to TOOLS_SCHEMA or to NOT_OFFERED_TO_THE_MODEL with a reason")
    assert not invented, (
        f"the agent panel offers {sorted(invented)}, which the runtime does not "
        f"serve; the model would call a tool that fails to route")


def test_every_deliberate_omission_is_still_a_real_runtime_tool():
    """Guards the escape hatch: an omission left behind after the runtime drops a
    tool would hide a genuine mismatch for the next person."""
    import pytest

    served = _runtime_tool_names()
    if served is None:
        pytest.skip("no ash-runtime binary")
    stale = NOT_OFFERED_TO_THE_MODEL - served
    assert not stale, f"{sorted(stale)} is excluded but the runtime no longer serves it"
