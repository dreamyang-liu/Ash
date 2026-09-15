"""Slot contract + command/wiring construction.

The CLI driver is exercised against a fake agent binary (a python script that
prints a canned JSONL stream) so the whole path -- spawn, stream, normalize,
journal, usage, exit status -- is covered without network or API keys.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from harness.core import events as E
from harness.core.journal import JournalWriter, read_journal
from harness.core.slot import AgentSlot, McpWiring, TaskSpec
from harness.execution.wiring import http_wiring, stdio_wiring
from harness.slots import available, load_slot
from harness.slots.cli_base import JsonlCliSlot
from harness.slots.claude_code import DEFAULT_MAX_BUFFER_SIZE, DENIED_BUILTINS, ClaudeCodeSlot
from harness.slots.codex_sdk import CodexSdkSlot
from harness.slots.opencode import OpenCodeSlot


def test_registry_exposes_every_slot():
    assert available() == ["claude-code", "codex", "opencode", "opencode-cli"]
    for name in available():
        cls = load_slot(name)
        assert issubclass(cls, AgentSlot)
        # `-cli` variants drive the same agent a different way, so they keep the
        # agent's name; the registry key is the driver, cls.name is the agent.
        assert cls.name == name.replace("-cli", "")


def test_default_names_bind_to_the_protocol_drivers():
    """The CLI drivers cannot branch a run and parse an unversioned stdout, so a
    bare slot name must not resolve to one."""
    from harness.slots.codex_sdk import CodexSdkSlot
    from harness.slots.opencode_server import OpenCodeServerSlot

    assert load_slot("codex") is CodexSdkSlot
    assert load_slot("opencode") is OpenCodeServerSlot
    assert load_slot("opencode-cli") is OpenCodeSlot
    for name in ("codex", "opencode", "claude-code"):
        assert load_slot(name).capabilities.fork, "%s should be forkable" % name


def test_unknown_slot_lists_alternatives():
    with pytest.raises(KeyError, match="claude-code"):
        load_slot("nope")


def test_removed_codex_cli_slot_is_rejected():
    with pytest.raises(KeyError, match="unknown slot"):
        load_slot("codex-cli")


# --- command construction --------------------------------------------------
def test_codex_sdk_mcp_stdio_becomes_config_overrides():
    mcp = McpWiring(name="ash", command=["python", "-m", "harness.execution.server", "--x"], env={"K": "v"})
    overrides = CodexSdkSlot()._config_overrides(mcp, {})
    joined = " ".join(overrides)
    assert 'mcp_servers.ash.command="python"' in joined
    assert 'mcp_servers.ash.args=["-m","harness.execution.server","--x"]' in joined
    assert 'mcp_servers.ash.env.K="v"' in joined


def test_codex_sdk_mcp_remote_uses_url_and_session_headers():
    """The header names are a wire contract with harness.execution.server
    (SESSION_HEADERS / SANDBOX_HEADER), not decoration: without X-Session-Owner
    every request is a fresh anonymous session, so a sandbox created by one call
    is invisible to the next."""
    mcp = http_wiring("http://h:8400/mcp", agent_id="a7", sandbox_id="sb-1")
    overrides = CodexSdkSlot()._config_overrides(mcp, {})
    joined = " ".join(overrides)
    assert 'mcp_servers.ash.url="http://h:8400/mcp"' in joined
    assert 'mcp_servers.ash.http_headers.X-Session-Owner="a7"' in joined
    assert 'mcp_servers.ash.http_headers.X-Session-Sandbox="sb-1"' in joined


def test_opencode_command_is_pure_by_default():
    command = OpenCodeSlot().build_command(TaskSpec(prompt="do it", cwd="/w"), None)
    assert command[:6] == ["opencode", "run", "--format", "json", "--dir", "/w"]
    assert "--pure" in command  # ignore developer plugins in eval runs
    assert command[-1] == "do it"


def test_opencode_fork_requires_session():
    task = TaskSpec(prompt="p", cwd="/w", extra={"resume_session_id": "ses_1", "fork": True})
    command = OpenCodeSlot().build_command(task, None)
    assert "--session" in command and "ses_1" in command and "--fork" in command


def test_opencode_writes_isolated_mcp_config(tmp_path):
    slot = OpenCodeSlot()
    mcp = stdio_wiring(args=["--attach", "sb-1"])
    env = slot.build_env(TaskSpec(prompt="p", cwd=str(tmp_path)), mcp)
    config = json.loads(open(env["OPENCODE_CONFIG"]).read())
    entry = config["mcp"]["ash"]
    assert entry["type"] == "local"
    assert entry["command"][1:3] == ["-m", "harness.execution.server"]
    assert entry["enabled"] is True


def test_opencode_fork_capability_is_advertised():
    """Only opencode has native session fork; the ledger relies on this flag."""
    assert OpenCodeSlot.capabilities.fork is True
    assert CodexSdkSlot.capabilities.fork is True
    assert ClaudeCodeSlot.capabilities.fork is True


# --- claude-code verdict seam ---------------------------------------------
@pytest.mark.parametrize("tool", ["Bash", "Read", "Write", "WebFetch"])
def test_claude_code_denies_builtin_tools(tool):
    """Builtins run on the host: their effects escape the sandbox and snapshots."""
    import asyncio

    slot = ClaudeCodeSlot()
    verdict = asyncio.run(slot._can_use_tool(tool, {}))
    assert verdict["behavior"] == "deny"
    assert "sandbox" in verdict["message"]
    assert slot.denied_calls == [tool]


def test_claude_code_allows_mcp_tools_and_counts_boundaries():
    import asyncio

    seen = []
    slot = ClaudeCodeSlot(on_tool_boundary=seen.append)
    verdict = asyncio.run(slot._can_use_tool("mcp__ash__shell", {"command": "ls"}))
    assert verdict["behavior"] == "allow"
    assert verdict["updatedInput"] == {"command": "ls"}
    asyncio.run(slot._can_use_tool("mcp__ash__grep_files", {}))
    assert seen == [1, 2]          # step boundary per approved call
    assert slot.denied_calls == []


def test_claude_code_boundary_callback_failure_does_not_break_the_run():
    import asyncio

    def explode(_index):
        raise RuntimeError("snapshot backend down")

    slot = ClaudeCodeSlot(on_tool_boundary=explode)
    verdict = asyncio.run(slot._can_use_tool("mcp__ash__shell", {}))
    assert verdict["behavior"] == "allow"


def test_claude_code_mcp_config_shapes():
    slot = ClaudeCodeSlot()
    servers, allowed = slot._mcp_config(
        TaskSpec(prompt="p", cwd="/w"), stdio_wiring(args=["--attach", "sb-1"])
    )
    assert servers["ash"]["type"] == "stdio"
    assert servers["ash"]["timeout"] == 600000
    assert allowed == ["mcp__ash"]

    servers, allowed = slot._mcp_config(
        TaskSpec(prompt="p", cwd="/w", extra={"mcp_tools": ["shell", "text_editor"]}),
        http_wiring("http://h/mcp"),
    )
    assert servers["ash"] == {"type": "http", "url": "http://h/mcp", "timeout": 600000}
    assert allowed == ["mcp__ash__shell"]

    servers, _ = slot._mcp_config(
        TaskSpec(prompt="p", cwd="/w", extra={"mcp_server_timeout_ms": 450000}),
        http_wiring("http://h/mcp"),
    )
    assert servers["ash"]["timeout"] == 450000

    sentinel = object()
    servers, _ = slot._mcp_config(
        TaskSpec(prompt="p", cwd="/w", extra={"sdk_mcp_server": sentinel}), None
    )
    assert servers["ash"] is sentinel  # in-process server passed through untouched


def test_denied_builtins_cover_host_escaping_tools():
    for tool in ("Bash", "Read", "Edit", "Write", "Glob", "Grep", "WebFetch"):
        assert tool in DENIED_BUILTINS
    # Out-of-band builtins: a CronCreate from one eval run was delivered into
    # other runs' sessions for hours (2026-09-04). Nothing that schedules,
    # messages other sessions, spawns subagents or touches host git may be offered.
    for tool in ("Task", "CronCreate", "ScheduleWakeup", "Workflow", "SendMessage",
                 "EnterWorktree", "Skill"):
        assert tool in DENIED_BUILTINS
    # ToolSearch stays: the MCP sandbox tools are deferred and found through it.
    assert "ToolSearch" not in DENIED_BUILTINS


@pytest.mark.parametrize("name", ["mcp__ash__text_editor", "mcp__custom__text_editor"])
def test_claude_code_denies_text_editor_in_both_permission_paths(name):
    import asyncio

    slot = ClaudeCodeSlot()
    hook = asyncio.run(slot._pre_tool_use({"tool_name": name}, "call-id"))
    assert hook["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert asyncio.run(slot._can_use_tool(name, {}))["behavior"] == "deny"
    assert "mcp__custom__text_editor" in slot._disallowed_tools({"custom": {}})


# --- CLI driver end-to-end (fake agent) -----------------------------------
FAKE_AGENT = textwrap.dedent(
    """
    import json, sys
    print(json.dumps({"type": "thread.started", "thread_id": "th_fake"}), flush=True)
    print("a banner line that is not json", flush=True)
    print(json.dumps({"type": "item.started",
                      "item": {"id": "i1", "item_type": "command_execution",
                               "command": "echo hi"}}), flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"id": "i1", "item_type": "command_execution",
                               "aggregated_output": "hi\\n", "exit_code": 0}}), flush=True)
    print(json.dumps({"type": "item.completed",
                      "item": {"id": "i2", "item_type": "agent_message",
                               "text": "all done"}}), flush=True)
    print(json.dumps({"type": "turn.completed",
                      "usage": {"input_tokens": 7, "output_tokens": 3}}), flush=True)
    sys.stderr.write("warning: fake stderr\\n")
    sys.exit(0)
    """
)


class FakeCliSlot(JsonlCliSlot):
    name = "fake"
    binary = None

    def __init__(self, script_path):
        super().__init__()
        self._script = script_path

    @property
    def normalizer(self):
        from harness.normalize import codex

        return codex.normalize

    def build_command(self, task, mcp):
        return [sys.executable, str(self._script)]


def test_cli_driver_streams_normalizes_and_accounts(tmp_path):
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    journal_path = tmp_path / "run.jsonl"
    slot = FakeCliSlot(script)

    with JournalWriter(journal_path, run_id="rf") as journal:
        result = slot.run(TaskSpec(prompt="go", cwd=str(tmp_path), timeout_s=60), journal)

    assert result.status == "completed"
    assert result.final_text == "all done"
    assert result.usage["input_tokens"] == 7
    assert result.native_session_id == "th_fake"

    records = read_journal(journal_path)
    kinds = [r["type"] for r in records]
    assert kinds[0] == E.RUN_STARTED and kinds[-1] == E.RUN_FINISHED
    assert E.TOOL_STARTED in kinds and E.TOOL_FINISHED in kinds
    # provenance for reproducibility
    started = records[0]
    assert started["slot"] == "fake" and started["task_prompt"] == "go"
    # non-JSON stdout is journaled, not dropped
    logs = [r for r in records if r["type"] == E.SLOT_LOG]
    assert any("banner" in r["text"] for r in logs if r["stream"] == "stdout")
    assert any("fake stderr" in r["text"] for r in logs if r["stream"] == "stderr")


def test_cli_driver_reports_missing_binary_without_raising(tmp_path):
    class MissingSlot(FakeCliSlot):
        def build_command(self, task, mcp):
            return ["/nonexistent/agent-binary"]

    with JournalWriter(tmp_path / "j.jsonl") as journal:
        result = MissingSlot(tmp_path / "x").run(
            TaskSpec(prompt="p", cwd=str(tmp_path)), journal
        )
    assert result.status == "error"
    assert "nonexistent" in (result.error or "")


def test_cli_driver_surfaces_stderr_on_failure(tmp_path):
    script = tmp_path / "failing.py"
    script.write_text("import sys; sys.stderr.write('boom: bad auth\\n'); sys.exit(3)")
    with JournalWriter(tmp_path / "j.jsonl") as journal:
        result = FakeCliSlot(script).run(TaskSpec(prompt="p", cwd=str(tmp_path)), journal)
    assert result.status == "error"
    assert result.exit_code == 3
    assert "bad auth" in result.error


def test_cli_driver_times_out(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text("import time; time.sleep(30)")
    with JournalWriter(tmp_path / "j.jsonl") as journal:
        result = FakeCliSlot(script).run(
            TaskSpec(prompt="p", cwd=str(tmp_path), timeout_s=1.0), journal
        )
    assert result.status == "timeout"


# --- PreToolUse is the seam that actually fires ----------------------------
def test_claude_code_registers_pretooluse_hook():
    """Under bypassPermissions the SDK skips can_use_tool entirely, so the
    verdict + step boundary must be mounted on PreToolUse or checkpoints never
    fire (the SDK warns about this: CanUseToolShadowedWarning)."""
    hooks = ClaudeCodeSlot()._hooks()
    if hooks is None:
        pytest.skip("claude-agent-sdk not installed")
    assert "PreToolUse" in hooks
    matcher = hooks["PreToolUse"][0]
    assert matcher.hooks, "matcher registered with no callbacks"


@pytest.mark.parametrize("tool", ["Bash", "Write", "WebFetch"])
def test_pretooluse_denies_builtins(tool):
    import asyncio

    slot = ClaudeCodeSlot()
    out = asyncio.run(slot._pre_tool_use({"tool_name": tool, "tool_input": {}}, "tu_1"))
    specific = out["hookSpecificOutput"]
    assert specific["hookEventName"] == "PreToolUse"
    assert specific["permissionDecision"] == "deny"
    assert slot.denied_calls == [tool]
    assert slot.boundary_count == 0  # denied calls are not step boundaries


def test_pretooluse_allows_mcp_and_marks_boundary():
    import asyncio

    seen = []
    slot = ClaudeCodeSlot(on_tool_boundary=seen.append)
    out = asyncio.run(
        slot._pre_tool_use({"tool_name": "mcp__ash__shell", "tool_input": {"command": "ls"}}, "tu_2")
    )
    specific = out["hookSpecificOutput"]
    assert specific["permissionDecision"] == "allow"
    assert specific["updatedInput"] == {"command": "ls"}
    assert seen == [1] and slot.boundary_count == 1


def test_pretooluse_tolerates_missing_fields():
    import asyncio

    slot = ClaudeCodeSlot()
    out = asyncio.run(slot._pre_tool_use({}, None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_can_use_tool_not_registered_under_bypass(monkeypatch):
    """Registering it there is shadowed by the SDK and only emits a warning."""
    import dataclasses

    @dataclasses.dataclass
    class FakeOptions:
        cwd: str = ""
        mcp_servers: dict = dataclasses.field(default_factory=dict)
        allowed_tools: list = dataclasses.field(default_factory=list)
        disallowed_tools: list = dataclasses.field(default_factory=list)
        permission_mode: str = ""
        hooks: object = None
        can_use_tool: object = None

    slot = ClaudeCodeSlot()
    bypass = slot._build_options(FakeOptions, TaskSpec(prompt="p", cwd="/w"), {}, [])
    assert bypass.permission_mode == "bypassPermissions"
    assert bypass.can_use_tool is None

    strict = slot._build_options(
        FakeOptions, TaskSpec(prompt="p", cwd="/w", extra={"permission_mode": "default"}), {}, []
    )
    assert strict.can_use_tool is not None


def test_claude_code_buffer_default_and_override():
    sdk = pytest.importorskip("claude_agent_sdk")
    slot = ClaudeCodeSlot()
    options = slot._build_options(sdk.ClaudeAgentOptions, TaskSpec(prompt="p", cwd="/tmp"), {}, [])
    assert options.max_buffer_size == DEFAULT_MAX_BUFFER_SIZE == 16 * 1024 * 1024
    task = TaskSpec(prompt="p", cwd="/tmp", extra={"max_buffer_size": 4 * 1024 * 1024})
    assert slot._build_options(sdk.ClaudeAgentOptions, task, {}, []).max_buffer_size == 4 * 1024 * 1024


def test_miles_rollout_disables_anthropic_extended_thinking():
    class Options:
        def __init__(self, thinking=None, **kwargs):
            self.thinking = thinking

    task = TaskSpec(
        prompt="p",
        cwd="/tmp",
        extra={"rollout_contract": {"model_endpoint": "http://model"}},
    )
    options = ClaudeCodeSlot()._build_options(Options, task, {}, [])
    assert options.thinking == {"type": "disabled"}


@pytest.mark.parametrize("size", [0, -1, None, True, "large"])
def test_claude_code_rejects_invalid_buffer_sizes(size):
    with pytest.raises(ValueError, match="max_buffer_size"):
        ClaudeCodeSlot()._build_options(
            dict, TaskSpec(prompt="p", cwd="/tmp", extra={"max_buffer_size": size}), {}, [])


def test_claude_code_records_effective_buffer_size(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT", raising=False)
    sdk = pytest.importorskip("claude_agent_sdk")
    options_seen = []

    async def query(prompt, options):
        options_seen.append(options)
        for message in ():
            yield message

    monkeypatch.setattr(sdk, "query", query)
    slot = ClaudeCodeSlot()
    monkeypatch.setattr(slot, "version", lambda: "test")
    path = tmp_path / "buffer.jsonl"
    with JournalWriter(path, run_id="buffer-test") as journal:
        result = slot.run(TaskSpec(prompt="p", cwd="/tmp"), journal)
    started = next(event for event in read_journal(path) if event["type"] == E.RUN_STARTED)
    assert result.status == "completed"
    assert started["config"]["max_buffer_size"] == options_seen[0].max_buffer_size
    assert started["config"]["max_buffer_size"] == DEFAULT_MAX_BUFFER_SIZE
    assert started["config"]["mcp_tool_timeout_ms"] == options_seen[0].env["MCP_TOOL_TIMEOUT"]
    assert started["config"]["mcp_tool_timeout_ms"] == "600000"
    assert started["config"]["mcp_tool_idle_timeout_ms"] == options_seen[0].env[
        "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT"]
    assert started["config"]["mcp_tool_idle_timeout_ms"] == "600000"
    assert started["config"]["mcp_server_timeouts_ms"] == {}


def test_claude_slot_records_complete_model_turns(tmp_path, monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    from harness.tests.test_model_turns import assistant, result
    from harness.normalize.claude_turns import completed_turn_steps

    async def query(prompt, options):
        for message in (assistant("t1", "a", "b"), result("a"), result("b"), assistant("t2")):
            yield message

    monkeypatch.setattr(sdk, "query", query)
    slot = ClaudeCodeSlot()
    monkeypatch.setattr(slot, "version", lambda: "test")
    path = tmp_path / "turns.jsonl"
    with JournalWriter(path) as journal:
        outcome = slot.run(TaskSpec(prompt="p", cwd=str(tmp_path)), journal)
    assert outcome.status == "completed"
    assert list(completed_turn_steps(read_journal(path))) == [2]


def test_claude_sdk_transport_accepts_large_messages_with_slot_buffer():
    import asyncio

    sdk = pytest.importorskip("claude_agent_sdk")
    transport_module = pytest.importorskip("claude_agent_sdk._internal.transport.subprocess_cli")
    payload = {"type": "system", "data": "x" * (2 * 1024 * 1024)}

    class Process:
        async def wait(self):
            return 0

    async def chunks():
        message = json.dumps(payload) + "\n"
        for offset in range(0, len(message), 65536):
            yield message[offset:offset + 65536]

    async def read(options):
        transport = transport_module.SubprocessCLITransport(prompt="p", options=options)
        transport._process = Process()
        transport._stdout_stream = chunks()
        return [message async for message in transport.read_messages()]

    with pytest.raises(sdk.CLIJSONDecodeError, match="maximum buffer size"):
        asyncio.run(read(sdk.ClaudeAgentOptions(max_buffer_size=1024 * 1024)))
    options = ClaudeCodeSlot()._build_options(
        sdk.ClaudeAgentOptions, TaskSpec(prompt="p", cwd="/tmp"), {}, [])
    assert asyncio.run(read(options)) == [payload]


def test_codex_sdk_overrides_are_toml_assignment_strings():
    """CodexConfig.config_overrides is iterated verbatim into `--config <kv>`,
    so each element must already be a `key=value` TOML assignment. This used to
    return a dict, whose iteration yields its KEYS: codex got
    `--config mcp_servers.ash` with no `=` and refused to start -- but only when
    a wiring was present, which is why every bare run passed and the first
    orchestrator-wired one did not."""
    from harness.slots.codex_sdk import CodexSdkSlot

    slot = CodexSdkSlot()
    http = http_wiring("http://h:8400/mcp", agent_id="a7", sandbox_id="sb-1")
    overrides = slot._config_overrides(http, {})
    assert isinstance(overrides, tuple)
    assert all(isinstance(kv, str) and "=" in kv for kv in overrides), overrides
    joined = " ".join(overrides)
    assert 'mcp_servers.ash.url="http://h:8400/mcp"' in joined
    assert 'http_headers.X-Session-Owner="a7"' in joined
    assert 'http_headers.X-Session-Sandbox="sb-1"' in joined


def test_codex_sdk_uses_dedicated_config_helpers():
    import inspect

    from harness.slots import codex_sdk

    source = inspect.getsource(codex_sdk)
    assert "from harness.slots.codex_config import _mcp_config_pairs" in source
    assert "from harness.slots.codex import" not in source


def test_codex_contract_targets_sdk_notifications():
    from contracts.ci_check import CONTRACTS_DIR, Report, check_normalizer_alignment, load_yaml

    contract = load_yaml(CONTRACTS_DIR / "codex.yaml")
    assert contract["slot"] == "codex"
    assert contract["python_api"]["module"] == "openai_codex"
    report = Report()
    check_normalizer_alignment(contract, report)
    assert not report.failures


def test_codex_sdk_defaults_the_host_sandbox_to_read_only():
    """Side effects belong in the
    ash sandbox via MCP. With codex's own policy (None), a live fork demo had
    codex write the task's files to the HOST -- the run looked perfect while
    both "isolated" branches mutated the same real machine."""
    import enum

    from harness.slots.codex_sdk import CodexSdkSlot

    class Sandbox(enum.Enum):
        READ_ONLY = "read-only"
        WORKSPACE_WRITE = "workspace-write"

    slot = CodexSdkSlot()
    assert slot._sandbox(Sandbox, {}) is Sandbox.READ_ONLY
    assert slot._sandbox(Sandbox, {"sandbox": "workspace-write"}) is \
        Sandbox.WORKSPACE_WRITE
    assert slot._sandbox(Sandbox, {"sandbox": "no-such-mode"}) is \
        Sandbox.READ_ONLY, "an unknown mode must fail closed, not open"


def test_codex_sdk_answers_mcp_elicitation_in_its_own_vocabulary():
    """MCP tool approvals arrive as `mcpServer/elicitation/request`, whose reply
    is {action, content} -- NOT the {decision} every other approval method takes.
    Answering {decision: accept} parses as neither accept nor decline, and codex
    cancels the call: the model reported its ash calls "rejected" while our
    journal showed verdict=allow and no tools/call ever reached the server."""
    from harness.slots.codex_sdk import CodexSdkSlot

    class J:
        def emit(self, *a, **k):
            pass

    slot = CodexSdkSlot()
    slot._journal = J()
    reply = slot._on_approval("mcpServer/elicitation/request",
                              {"serverName": "ash", "mode": "form"})
    assert reply == {"action": "accept", "content": {}}

    slot.policy = lambda kind, payload: ("deny", "test")
    reply = slot._on_approval("mcpServer/elicitation/request", {})
    assert reply == {"action": "decline"}


@pytest.mark.parametrize("task_env,inherited,expected", [
    ({}, None, "600000"),
    ({}, "", "600000"),
    ({}, "450000", "450000"),
    ({"MCP_TOOL_TIMEOUT": "180000"}, "450000", "180000"),
])
def test_mcp_tool_wait_defaults_to_600_seconds_and_preserves_overrides(
        monkeypatch, task_env, inherited, expected):
    if inherited is None:
        monkeypatch.delenv("MCP_TOOL_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("MCP_TOOL_TIMEOUT", inherited)

    class Options:
        def __init__(self, env=None, **kwargs):
            self.env = env

    task = TaskSpec(prompt="p", cwd="/w", timeout_s=10800, env=dict(task_env))
    result = ClaudeCodeSlot()._build_options(Options, task, {}, [])
    assert result.env["MCP_TOOL_TIMEOUT"] == expected
    assert task.env == task_env
    assert task.timeout_s == 10800
    assert "CLAUDE_CODE_USE_BEDROCK" not in result.env


@pytest.mark.parametrize("task_env,inherited,expected", [
    ({}, None, "600000"),
    ({}, "", "600000"),
    ({}, "450000", "450000"),
    ({"CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT": "180000"}, "450000", "180000"),
])
def test_mcp_tool_idle_wait_defaults_to_600_seconds_and_preserves_overrides(
        monkeypatch, task_env, inherited, expected):
    if inherited is None:
        monkeypatch.delenv("CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT", inherited)

    class Options:
        def __init__(self, env=None, **kwargs):
            self.env = env

    task = TaskSpec(prompt="p", cwd="/w", timeout_s=10800, env=dict(task_env))
    result = ClaudeCodeSlot()._build_options(Options, task, {}, [])
    assert result.env["CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT"] == expected
    assert task.env == task_env
    assert task.timeout_s == 10800


@pytest.mark.parametrize("value", [0, -1, None, True, "600000"])
def test_claude_code_rejects_invalid_mcp_server_timeout(value):
    with pytest.raises(ValueError, match="mcp_server_timeout_ms"):
        ClaudeCodeSlot()._mcp_config(
            TaskSpec(prompt="p", cwd="/w", extra={"mcp_server_timeout_ms": value}),
            http_wiring("http://h/mcp"),
        )


def test_a_routed_run_disables_the_clis_provider_direct_modes():
    """ANTHROPIC_BASE_URL in the task env means the run routed the traffic (a
    gateway, a vLLM, an RL checkpoint). The CLI's Bedrock/Vertex modes ignore
    the base URL entirely -- measured live: a --gateway run completed against
    Bedrock with the gateway recording zero events and enforcing nothing."""
    class Opts:
        # named parameter, because the slot's kwarg filter reads the signature
        # and drops anything the options class does not declare
        def __init__(self, env=None, **kw):
            self.env = env
            self.__dict__.update(kw)

    slot = ClaudeCodeSlot()

    def build(env):
        task = TaskSpec(prompt="p", cwd="/w", env=env)
        return slot._build_options(Opts, task, {}, []).env

    env = build({"ANTHROPIC_BASE_URL": "http://gw:1", "ANTHROPIC_AUTH_TOKEN": "t"})
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "0"
    assert env["CLAUDE_CODE_USE_VERTEX"] == "0"
    assert "MCP_TOOL_TIMEOUT" in env

    # explicit wins: a run that WANTS bedrock behind its own url keeps it
    env = build({"ANTHROPIC_BASE_URL": "http://gw:1", "CLAUDE_CODE_USE_BEDROCK": "1"})
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"

    # no base url: nothing injected, the CLI keeps its own configuration
    env = build({"X": "1"})
    assert "CLAUDE_CODE_USE_BEDROCK" not in env


# --- opencode on a routed run --------------------------------------------------
def test_opencode_translates_a_routed_run_into_its_own_provider_config(tmp_path):
    """opencode does not read ANTHROPIC_BASE_URL -- it picks a provider from its
    own config, and its default here is Bedrock via AWS env vars. Without this
    translation a --gateway run went straight to the provider and the gateway
    recorded nothing (the same silent-bypass shape the claude slot had)."""
    from harness.slots.opencode_server import OpenCodeServerSlot

    slot = OpenCodeServerSlot()
    task = TaskSpec(prompt="p", cwd=str(tmp_path),
                    env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:9999",
                         "ANTHROPIC_AUTH_TOKEN": "slot-tok"})
    path = slot._render_config(stdio_wiring(args=["--attach", "sb-1"]), task)
    config = json.loads(open(path).read())
    options = config["provider"]["anthropic"]["options"]
    assert options["baseURL"] == "http://127.0.0.1:9999/v1"
    assert options["apiKey"] == "slot-tok"
    assert "ash" in config["mcp"], "the tool wiring must survive alongside it"


def test_opencode_writes_a_config_for_routing_even_with_no_tools(tmp_path):
    """Gating the config file on `mcp` meant a gateway run without sandbox tools
    silently talked to the provider direct."""
    from harness.slots.opencode_server import OpenCodeServerSlot

    slot = OpenCodeServerSlot()
    task = TaskSpec(prompt="p", cwd=str(tmp_path),
                    env={"ANTHROPIC_BASE_URL": "http://gw:1"})
    path = slot._render_config(None, task)
    assert path, "no mcp but routed: the file must still be written"
    config = json.loads(open(path).read())
    assert config["provider"]["anthropic"]["options"]["baseURL"] == "http://gw:1/v1"

    # and an unrouted run with no tools still writes nothing
    assert slot._render_config(None, TaskSpec(prompt="p", cwd=str(tmp_path))) is None


def test_opencode_provider_direct_vars_are_cleared_on_a_routed_run():
    """Leaving AWS_* set makes opencode prefer its Bedrock provider over the
    routed one -- measured: opencode's own auth list shows Bedrock picked up
    from AWS_BEARER_TOKEN_BEDROCK."""
    from harness.slots.opencode_server import _PROVIDER_DIRECT_VARS

    assert "AWS_BEARER_TOKEN_BEDROCK" in _PROVIDER_DIRECT_VARS
