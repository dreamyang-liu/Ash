"""The claude-code harness's wiring, asserted without a CLI or a sandbox.

This path had **no tests at all** while it spawned an MCP subprocess that owned
the sandbox. That is why the ownership defect survived: nothing here could tell
you the harness could not snapshot its own environment, or that an extraction
failure and an empty diff produced the same result.

These cover the wiring the rewrite depends on -- the harness owns the session, the
patch is a direct call, teardown happens after extraction -- with a fake session
and a fake SDK.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from swebench.harnesses.claude_code import ClaudeCodeHarness


# --- fakes -----------------------------------------------------------------
@dataclass
class FakeResult:
    output: str = "ok"
    success: bool = True
    error: Optional[str] = None


@dataclass
class FakeSession:
    """Enough AshSession for the harness: create, execute, patch, destroy."""

    patch: str = "diff --git a/f.py b/f.py\n"
    created: Optional[str] = None
    destroyed: bool = False
    calls: List[tuple] = field(default_factory=list)
    order: List[str] = field(default_factory=list)
    create_ok: bool = True

    def create(self, image: str, resources=None) -> bool:
        self.created = image
        self.order.append("create")
        return self.create_ok

    def executor_for(self, agent_id: str, pipeline=None):
        def execute(name: str, args: dict) -> FakeResult:
            self.calls.append((name, args))
            return FakeResult(output="ran %s" % name)
        return execute

    def get_patch(self) -> str:
        self.order.append("get_patch")
        return self.patch

    def environment(self) -> dict:
        return {"base_image": self.created, "sandbox": "sb-1"}

    def destroy(self) -> None:
        self.order.append("destroy")
        self.destroyed = True


class FakeOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def fake_tool(name, description, schema):
    """Stands in for claude_agent_sdk.tool -- records the registration."""
    def decorate(handler):
        handler._tool = {"name": name, "description": description, "schema": schema}
        return handler
    return decorate


@pytest.fixture
def harness(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("swebench.harnesses.claude_code.AshSession",
                        lambda **kwargs: session)
    h = ClaudeCodeHarness({"model": "opus", "max_turns": 40})
    h._fake_session = session
    return h


def _instance():
    return {"instance_id": "django__django-1", "problem_statement": "it is broken",
            "repo": "django/django", "base_commit": "abc"}


# --- ownership: the point of the rewrite -----------------------------------
def test_the_harness_creates_its_own_sandbox(harness, monkeypatch, tmp_path):
    """Not the MCP server. The old topology had the server create it, which is
    why nothing here could snapshot or re-read the environment."""
    _run(harness, monkeypatch, tmp_path)
    assert harness._fake_session.created, "the harness did not create a sandbox"


def test_the_patch_is_read_before_the_sandbox_is_destroyed(harness, monkeypatch, tmp_path):
    """The old order destroyed the sandbox as the stream closed -- exactly when
    grading needs it -- and compensated by polling a file."""
    _run(harness, monkeypatch, tmp_path)
    order = harness._fake_session.order
    assert order.index("get_patch") < order.index("destroy"), order


def test_the_sandbox_is_destroyed_even_when_the_stream_raises(harness, monkeypatch, tmp_path):
    def exploding_query(prompt, options):
        async def gen():
            raise RuntimeError("stream died")
            yield  # pragma: no cover
        return gen()

    result = _run(harness, monkeypatch, tmp_path, query=exploding_query)
    assert harness._fake_session.destroyed
    assert "error" in result["exit_status"]


def test_a_failed_create_is_reported_and_not_a_crash(harness, monkeypatch, tmp_path):
    harness._fake_session.create_ok = False
    result = _run(harness, monkeypatch, tmp_path)
    assert "sandbox create failed" in result["exit_status"]


# --- the patch -------------------------------------------------------------
def test_a_patch_makes_the_run_completed(harness, monkeypatch, tmp_path):
    result = _run(harness, monkeypatch, tmp_path)
    assert result["exit_status"] == "completed"
    assert result["model_patch"].startswith("diff --git")


def test_an_empty_patch_is_no_patch_not_an_error(harness, monkeypatch, tmp_path):
    """A genuine no-op is distinguishable now. Under the polling scheme an
    extraction failure looked identical to this."""
    harness._fake_session.patch = ""
    result = _run(harness, monkeypatch, tmp_path)
    assert result["exit_status"] == "no_patch"


# --- tool wiring -----------------------------------------------------------
def test_the_four_exec_tools_are_registered_with_the_proxys_schemas(harness):
    """Imported from EXEC_TOOLS_SINGLE so this entry point and the MCP proxy
    cannot drift."""
    from swebench.mcp_server import EXEC_TOOLS_SINGLE

    tools = harness._sandbox_tools(fake_tool, FakeSession())
    registered = {t._tool["name"] for t in tools}
    assert registered == {s["name"] for s in EXEC_TOOLS_SINGLE}
    for t in tools:
        assert t._tool["schema"], "%s registered without a schema" % t._tool["name"]
        # the schema must be the proxy's, not a restatement
        source = next(s for s in EXEC_TOOLS_SINGLE if s["name"] == t._tool["name"])
        assert t._tool["schema"] == source["inputSchema"]


def test_a_tool_call_reaches_the_session_executor(harness):
    session = FakeSession()
    tools = {t._tool["name"]: t for t in harness._sandbox_tools(fake_tool, session)}
    out = asyncio.run(tools["shell"]({"command": "pytest"}))
    assert session.calls == [("shell", {"command": "pytest"})]
    assert out["content"][0]["text"] == "ran shell"
    assert out["is_error"] is False


def test_a_failing_tool_call_is_reported_to_the_model(harness):
    session = FakeSession()

    def failing(agent_id, pipeline=None):
        return lambda name, args: FakeResult(output="", success=False, error="boom")

    session.executor_for = failing
    tools = {t._tool["name"]: t for t in harness._sandbox_tools(fake_tool, session)}
    out = asyncio.run(tools["shell"]({"command": "x"}))
    assert out["is_error"] is True
    assert "boom" in out["content"][0]["text"]


def test_no_guardrail_on_this_path(harness):
    """Its nudges were written to shape *this repo's* agent; nudging Claude Code
    too would measure a different scaffold than the published numbers."""
    import inspect

    source = inspect.getsource(harness._sandbox_tools)
    assert "Guardrail" not in source
    assert "TruncateInterceptor" in source and "OutcomePresenter" in source


# --- SDK options -----------------------------------------------------------
def test_builtin_tools_are_denied(harness, tmp_path):
    """They run on the host, not in the sandbox."""
    options = harness._build_options(FakeOptions, "opus", object(), tmp_path)
    for builtin in ("Bash", "Read", "Edit", "Write"):
        assert builtin in options.disallowed_tools


def test_only_the_ash_tools_are_allowed(harness, tmp_path):
    options = harness._build_options(FakeOptions, "opus", object(), tmp_path)
    assert options.allowed_tools == [
        "mcp__ash-sandbox__shell", "mcp__ash-sandbox__text_editor",
        "mcp__ash-sandbox__grep_files", "mcp__ash-sandbox__process"]


def test_cwd_is_the_output_dir_not_this_repository(harness, tmp_path):
    """Claude Code reads `.claude/` from its cwd; launching from the repo root
    once handed the agent this repo's own `ash` skill mid-task."""
    options = harness._build_options(FakeOptions, "opus", object(), tmp_path)
    assert Path(options.cwd) == tmp_path
    assert "Ash" not in Path(options.cwd).name or options.cwd == str(tmp_path)


def test_bedrock_provider_sets_its_env_flag(tmp_path):
    h = ClaudeCodeHarness({"provider": "bedrock"})
    options = h._build_options(FakeOptions, "opus", object(), tmp_path)
    assert options.env["CLAUDE_CODE_USE_BEDROCK"] == "1"


def test_explicit_env_survives_the_provider_defaults(tmp_path):
    h = ClaudeCodeHarness({"provider": "bedrock",
                           "env": {"CLAUDE_CODE_USE_BEDROCK": "0", "X": "1"}})
    options = h._build_options(FakeOptions, "opus", object(), tmp_path)
    assert options.env["CLAUDE_CODE_USE_BEDROCK"] == "0"   # setdefault, not force
    assert options.env["X"] == "1"


def test_the_in_process_server_is_the_only_mcp_server(harness, tmp_path):
    """No subprocess: the sandbox lives here, so the tools do too."""
    sentinel = object()
    options = harness._build_options(FakeOptions, "opus", sentinel, tmp_path)
    assert options.mcp_servers == {"ash-sandbox": sentinel}


# --- trajectory ------------------------------------------------------------
def test_the_trajectory_records_the_environment(harness, monkeypatch, tmp_path):
    """A SWE-bench image name is usually a mutable tag, so a run that cannot say
    what it ran against is not reproducible."""
    _run(harness, monkeypatch, tmp_path)
    written = json.loads(
        (tmp_path / "trajectories" / "django__django-1.json").read_text())
    assert written["environment"]["base_image"]
    assert written["exit_status"] == "completed"


# --- helper ----------------------------------------------------------------
def _run(harness, monkeypatch, tmp_path, query=None):
    """Drive run_instance with a fake SDK that yields one tool use and a result."""
    import types

    class ToolUseBlock:
        def __init__(self): self.id, self.name, self.input = "t1", "mcp__ash-sandbox__shell", {"command": "ls"}

    class TextBlock:
        def __init__(self): self.text = "working on it"

    class ThinkingBlock:
        thinking = ""

    class ToolResultBlock:
        tool_use_id, content, is_error = "t1", "ok", False

    class AssistantMessage:
        def __init__(self): self.content = [ToolUseBlock(), TextBlock()]

    class UserMessage:
        def __init__(self): self.content = [ToolResultBlock()]

    class ResultMessage:
        is_error, result, total_cost_usd, num_turns, usage = False, "done", 0.5, 3, {}

    def default_query(prompt, options):
        async def gen():
            yield AssistantMessage()
            yield UserMessage()
            yield ResultMessage()
        return gen()

    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = FakeOptions
    fake_sdk.create_sdk_mcp_server = lambda **kwargs: object()
    fake_sdk.query = query or default_query
    fake_sdk.tool = fake_tool

    fake_types = types.ModuleType("claude_agent_sdk.types")
    for cls in (AssistantMessage, UserMessage, ResultMessage, TextBlock,
                ThinkingBlock, ToolUseBlock, ToolResultBlock):
        setattr(fake_types, cls.__name__, cls)

    monkeypatch.setitem(__import__("sys").modules, "claude_agent_sdk", fake_sdk)
    monkeypatch.setitem(__import__("sys").modules, "claude_agent_sdk.types", fake_types)

    return harness.run_instance(_instance(), tmp_path)
