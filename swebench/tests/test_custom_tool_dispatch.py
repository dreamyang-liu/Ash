"""Executor-level integration tests (pytest, alongside test_custom_tools.py).

Exercises AshAgent._run_tool's custom-tool branch (agent/__init__.py) with a
stub executor — no sandbox or network — plus load_custom_tools directory
semantics (explicit dir must exist; default dir optional).
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


class NoopGuardrails:
    def check(self, name, args):
        return None


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


def test_custom_tool_dispatches_artifact_then_shell():
    register_analyzer()
    calls = []

    def executor(name, args):
        calls.append((name, dict(args)))
        if name == "artifact":
            return ToolResult(success=True, output="/tmp/ash-artifacts/bbbb/artifact", error="")
        if name == "shell":
            return ToolResult(success=True, output="analyzed ok", error="")
        raise AssertionError(f"unexpected tool {name}")

    agent = make_agent(executor)
    agent._run_tool(make_tool_call("analyzer", {"file": "main.py"}),
                    FakeConversation(), NoopGuardrails(), "turn1")

    assert [c[0] for c in calls] == ["artifact", "shell"]
    assert calls[0][1] == {"url": "https://example.com/analyzer", "sha256": SHA}
    assert calls[1][1]["command"] == "/tmp/ash-artifacts/bbbb/artifact main.py"
    assert calls[1][1]["timeout"] == 15


def test_custom_tool_artifact_failure_short_circuits():
    register_analyzer()
    calls = []

    def executor(name, args):
        calls.append(name)
        return ToolResult(success=False, output="", error="download failed: HTTP 403")

    agent = make_agent(executor)
    agent._run_tool(make_tool_call("analyzer", {"file": "x"}),
                    FakeConversation(), NoopGuardrails(), "turn1")
    assert calls == ["artifact"]  # shell step never runs


def test_custom_tool_bad_args_fail_before_any_execution():
    register_analyzer()
    calls = []

    def executor(name, args):
        calls.append(name)
        return ToolResult(success=True, output="", error="")

    agent = make_agent(executor)
    agent._run_tool(make_tool_call("analyzer", {}),  # missing required 'file'
                    FakeConversation(), NoopGuardrails(), "turn1")
    assert calls == []  # validation failed at plan time; nothing executed


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


def test_path_sourced_tool_skips_artifact_step():
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

    def executor(name, args):
        calls.append((name, dict(args)))
        return ToolResult(success=True, output="ok", error="")

    agent = make_agent(executor)
    agent._run_tool(make_tool_call("imagetool", {"file": "a.py"}),
                    FakeConversation(), NoopGuardrails(), "turn1")

    # Single step: shell only, no artifact download.
    assert [c[0] for c in calls] == ["shell"]
    assert calls[0][1]["command"] == "/opt/tools/analyzer a.py"
    assert calls[0][1]["timeout"] == 20
