"""How the agent loop treats a custom tool, and manifest loading.

Dispatch itself (artifact -> shell, path memoisation, failure short-circuit)
belongs to the SDK and is tested there (sdk/tests/test_toolset.py). What matters
here is the loop's side of the seam: a manifest-defined tool is passed to the
executor under its own name, so interceptors see one call and the SDK expands it.
Plus load_custom_tools directory semantics (explicit dir must exist; default dir
optional).
User instruction: "manifest是用户提供的…作为参数传进去，然后有一个默认的位置".
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swebench.agent import AshAgent  # noqa: E402
from swebench.agent.custom_tools import (  # noqa: E402
    CUSTOM_TOOL_SPECS,
    load_custom_tools,
    parse_manifest,
    register,
)
from swebench.models import ToolResult  # noqa: E402

SHA = "b" * 64


@pytest.fixture(autouse=True)
def clean_registry():
    CUSTOM_TOOL_SPECS.clear()
    yield
    CUSTOM_TOOL_SPECS.clear()


def make_agent(executor):
    config = types.SimpleNamespace(model="test", step_limit=1, cost_limit=1.0)
    return AshAgent(config, executor=executor)


def make_tool_call(name, args):
    fn = types.SimpleNamespace(name=name, arguments=json.dumps(args))
    return types.SimpleNamespace(id="tc1", function=fn)


class FakeConversation:
    def __init__(self):
        self.tool_results = []

    def add_tool_result(self, *a, **kw):
        self.tool_results.append((a, kw))


def register_analyzer():
    register(parse_manifest({
        "name": "analyzer",
        "description": "test analyzer",
        "binary": {"url": "https://example.com/analyzer", "sha256": SHA},
        "parameters": {
            "file": {"type": "string", "required": True, "map": {"positional": 0}},
        },
        "timeout": 15,
    }))


def test_a_custom_tool_reaches_the_executor_under_its_own_name():
    """The loop no longer expands manifest tools: the executor does, and it
    memoises where the binary landed so a repeat call skips the download. One
    call also means one interceptor decision -- tell a coordination interceptor about such a
    tool through
    `opaque_writers` so its drift scan still runs."""
    register_analyzer()
    calls = []

    def executor(name, args):
        calls.append((name, dict(args)))
        return ToolResult(success=True, output="analyzed ok", error="")

    agent = make_agent(executor)
    agent._run_tool(make_tool_call("analyzer", {"file": "main.py"}),
                    FakeConversation(), "turn1")

    assert calls == [("analyzer", {"file": "main.py"})]


def test_builtin_names_are_still_translated_for_the_interceptors():
    """Unlike custom tools, builtins are routed here on purpose: an interceptor keyed on
    `shell` must not go blind because a run is in bash_only mode."""
    calls = []

    def executor(name, args):
        calls.append(name)
        return ToolResult(success=True, output="ok", error="")

    agent = make_agent(executor)
    agent._run_tool(make_tool_call("bash", {"command": "ls"}),
                    FakeConversation(), "turn1")
    assert calls == ["shell"]


def test_an_unknown_tool_fails_before_reaching_the_executor():
    calls = []
    agent = make_agent(lambda n, a: calls.append(n) or ToolResult(True, "", ""))
    conv = FakeConversation()
    agent._run_tool(make_tool_call("no_such_tool", {}), conv, "turn1")
    assert calls == []
    assert "unknown agent tool" in conv.tool_results[0][0][1]


def test_load_custom_tools_explicit_dir(tmp_path):
    manifest = {
        "name": "mytool",
        "description": "d",
        "binary": {"url": "https://example.com/t", "sha256": SHA},
        "parameters": {"arg": {"type": "string", "map": {"positional": 0}}},
    }
    (tmp_path / "mytool.json").write_text(json.dumps(manifest))
    specs = load_custom_tools(tmp_path)
    assert [s.name for s in specs] == ["mytool"]
    assert "mytool" in CUSTOM_TOOL_SPECS


def test_load_custom_tools_missing_explicit_dir_errors(tmp_path):
    from swebench.agent.custom_tools import ManifestError
    with pytest.raises(ManifestError, match="not found"):
        load_custom_tools(tmp_path / "nope")


def test_load_custom_tools_default_absent_is_noop(monkeypatch, tmp_path):
    import swebench.agent.custom_tools as ct
    monkeypatch.setattr(ct, "DEFAULT_MANIFEST_DIR", tmp_path / "missing")
    assert ct.load_custom_tools(None) == []


def test_a_path_sourced_tool_also_stays_opaque_to_the_loop():
    """Whether a manifest tool needs a download is the SDK's concern; the loop
    treats every one of them the same. (Single vs two-step dispatch is covered
    by sdk/tests/test_toolset.py.)"""
    register(parse_manifest({
        "name": "imagetool",
        "description": "binary baked into the image",
        "binary": {"path": "/opt/tools/analyzer"},
        "parameters": {
            "file": {"type": "string", "required": True, "map": {"positional": 0}},
        },
        "timeout": 20,
    }))
    calls = []

    agent = make_agent(lambda n, a: calls.append((n, dict(a))) or
                       ToolResult(success=True, output="ok", error=""))
    agent._run_tool(make_tool_call("imagetool", {"file": "a.py"}),
                    FakeConversation(), "turn1")

    assert calls == [("imagetool", {"file": "a.py"})]


def test_the_session_dispatches_through_the_sdk_not_a_raw_call():
    """The point of the sink: AshSession hands the SDK the agent-facing name so
    it expands manifest tools and memoises the binary path. Using `call` instead
    would send `mytool` to the runtime, which has no such tool."""
    from swebench.sandbox import AshSession
    from ash_sandbox.result import ToolResult as SdkToolResult

    seen = {}

    class FakeSandbox:
        async def call_agent_tool(self, name, args, registry=None, agent_id=""):
            seen["via"] = "call_agent_tool"
            seen["name"] = name
            seen["registry_has_customs"] = registry is not None and \
                bool(registry.custom_specs)
            return SdkToolResult(output="ok", is_error=False)

        async def call(self, tool_name, agent_id="", **kwargs):
            seen["via"] = "call"
            return SdkToolResult(output="ok", is_error=False)

    register_analyzer()                      # puts a spec in the default registry
    session = AshSession(quiet=True)
    session._sandbox = FakeSandbox()
    result = session.execute("analyzer", {"file": "m.py"})

    assert result.success
    assert seen["via"] == "call_agent_tool"
    assert seen["name"] == "analyzer"
    assert seen["registry_has_customs"], \
        "a Sandbox starts with an empty registry; the loaded manifests must be passed"
