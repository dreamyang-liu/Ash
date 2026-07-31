"""SDK-level tests for ash_sandbox.toolset + Sandbox.call_agent_tool.

Run by pytest with the other swebench tests (same sys.path pattern as
test_tool_contract.py); uses a fake Backend, no network; asyncio.run
wrappers (no pytest-asyncio in env). Part of the refactor per user
instruction: "tool 之类的可以用DSL 或者data 配置项来做" (tools as data).
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ash_sandbox import Sandbox, ToolRegistry, parse_manifest  # noqa: E402
from ash_sandbox.backends import Backend  # noqa: E402
from ash_sandbox.result import ToolResult  # noqa: E402

SHA = "c" * 64


class FakeBackend(Backend):
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    async def call(self, tool_name, args):
        self.calls.append((tool_name, dict(args)))
        if tool_name in self.responses:
            return self.responses[tool_name]
        return ToolResult(output="ok", is_error=False)

    async def list_tools(self):
        return []

    async def close(self):
        pass


def make_registry(binary=None):
    reg = ToolRegistry()
    reg.register(parse_manifest({
        "name": "analyzer",
        "description": "d",
        "binary": binary or {"url": "https://example.com/a", "sha256": SHA},
        "parameters": {
            "file": {"type": "string", "required": True, "map": {"positional": 0}},
        },
        "timeout": 20,
    }))
    return reg


def test_call_agent_tool_builtin_routes():
    backend = FakeBackend()
    sb = Sandbox(backend=backend)
    r = asyncio.run(sb.call_agent_tool("shell", {"command": "echo hi"}))
    assert not r.is_error
    assert backend.calls == [("shell", {"command": "echo hi"})]
    # bash alias routes to shell via BUILTIN_ROUTES
    asyncio.run(sb.call_agent_tool("bash", {"command": "ls"}))
    assert backend.calls[-1][0] == "shell"


def test_call_agent_tool_custom_url_two_steps():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="/cache/bin\n", is_error=False),
    })
    sb = Sandbox(backend=backend, tools=make_registry())
    r = asyncio.run(sb.call_agent_tool("analyzer", {"file": "m.py"}))
    assert not r.is_error
    assert [c[0] for c in backend.calls] == ["artifact", "shell"]
    assert backend.calls[1][1]["command"] == "/cache/bin m.py"
    assert backend.calls[1][1]["timeout"] == 20


def test_call_agent_tool_custom_path_single_step():
    backend = FakeBackend()
    sb = Sandbox(backend=backend, tools=make_registry(binary={"path": "/opt/a"}))
    asyncio.run(sb.call_agent_tool("analyzer", {"file": "m.py"}))
    assert [c[0] for c in backend.calls] == ["shell"]
    assert backend.calls[0][1]["command"] == "/opt/a m.py"


def test_call_agent_tool_artifact_failure_short_circuits():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="HTTP 404", is_error=True),
    })
    sb = Sandbox(backend=backend, tools=make_registry())
    r = asyncio.run(sb.call_agent_tool("analyzer", {"file": "m.py"}))
    assert r.is_error
    assert [c[0] for c in backend.calls] == ["artifact"]


def test_per_call_registry_override():
    backend = FakeBackend()
    sb = Sandbox(backend=backend)  # default registry: no custom tools
    with pytest.raises(KeyError):
        asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}))
    # Same sandbox, per-call registry carries the tool panel.
    asyncio.run(sb.call_agent_tool(
        "analyzer", {"file": "x"},
        registry=make_registry(binary={"path": "/opt/a"}),
    ))
    assert backend.calls[-1][0] == "shell"


def test_registries_are_isolated():
    r1, r2 = make_registry(), ToolRegistry()
    assert r1.is_custom_tool("analyzer")
    assert not r2.is_custom_tool("analyzer")  # no cross-instance leakage
