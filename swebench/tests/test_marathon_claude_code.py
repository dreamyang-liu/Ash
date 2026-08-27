"""Wiring for the marathon-claude-code harness.

The recurring failure in this stack is working code nothing calls (see
CLAUDE.md), so these assert consumption: the harness is reachable from the
CLI's registry, the task's resources reach the sandbox, the verifier runs on
the harness's own session, and the tools Claude Code sees are the MCP proxy's
schemas routed through the L2 executor seam.
"""

import asyncio
import inspect
from pathlib import Path

from swebench.harnesses.marathon_claude_code import (MarathonClaudeCodeHarness,
                                                     _SYSTEM_PROMPT)
from swebench.marathon import MarathonTask
from swebench.models import ToolResult


def make_task(tmp_path, *, internet_restricted=False, workdir="/workspace/rj"):
    return MarathonTask(name="abundant/rust-java-lsp", directory=tmp_path,
                        instruction="Build it.", workdir=workdir,
                        internet_restricted=internet_restricted)


class RecordingOptions:
    """Stands in for ClaudeAgentOptions; keeps whatever the harness wired."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_harness_is_registered():
    from swebench.harnesses import HARNESSES
    assert HARNESSES.get("marathon-claude-code") is MarathonClaudeCodeHarness


def test_grading_runs_on_the_harness_session():
    """The whole reason the sandbox lives in the harness rather than in an
    MCP subprocess: the subprocess destroys its sandbox when the stream
    closes, which is exactly when grading needs it."""
    source = inspect.getsource(MarathonClaudeCodeHarness.run_instance)
    assert "grade(session, task)" in source
    # ...and grading is not gated on a clean exit: a deadline'd attempt still
    # left hours of work on disk.
    assert source.index("self._drive") < source.index("grade(session, task)")


def test_task_resources_reach_the_sandbox():
    source = inspect.getsource(MarathonClaudeCodeHarness.run_instance)
    assert '"cpu": task.cpus' in source
    assert '"memory_mb": task.memory_mb' in source
    assert "session.create(image, resources=resources)" in source


def test_tool_schemas_come_from_the_mcp_proxy():
    """Two Claude Code entry points, one description of the tools: the
    schemas are imported from swebench.mcp_server, not restated."""
    from swebench.harnesses import marathon_claude_code as module
    assert "EXEC_TOOLS_SINGLE" in inspect.getsource(module)


def test_restricted_tasks_lose_provider_web_tools(tmp_path):
    """16 of the 20 tasks deny the network; WebSearch/WebFetch are served by
    the provider and would reach it anyway, answering an easier benchmark."""
    harness = MarathonClaudeCodeHarness({})
    restricted = harness._build_options(
        RecordingOptions, make_task(tmp_path, internet_restricted=True), None)
    assert {"WebSearch", "WebFetch"} <= set(restricted.disallowed_tools)

    open_task = harness._build_options(
        RecordingOptions, make_task(tmp_path, internet_restricted=False), None)
    assert "WebSearch" not in open_task.disallowed_tools
    # Built-ins operate on the wrong machine either way.
    assert "Bash" in open_task.disallowed_tools


def test_workdir_reaches_the_prompt(tmp_path):
    """Hardcoding /app dropped an agent a directory away from its own
    deliverables (see marathon.py); the task's WORKDIR is the authority."""
    harness = MarathonClaudeCodeHarness({})
    options = harness._build_options(RecordingOptions, make_task(tmp_path), None)
    assert "/workspace/rj" in options.system_prompt
    assert "/testbed" not in options.system_prompt


def test_turn_budget_is_marathon_shaped(tmp_path):
    """A SWE-bench-shaped 200 turns ends a 5-hour task a tenth of the way in.
    The config's step_limit is the ceiling, matching the marathon harness."""
    harness = MarathonClaudeCodeHarness({"step_limit": 1234})
    options = harness._build_options(RecordingOptions, make_task(tmp_path), None)
    assert options.max_turns == 1234
    default = MarathonClaudeCodeHarness({})._build_options(
        RecordingOptions, make_task(tmp_path), None)
    assert default.max_turns == 2000


def test_bedrock_provider_reaches_the_subprocess_env(tmp_path):
    harness = MarathonClaudeCodeHarness({"provider": "bedrock",
                                         "env": {"AWS_REGION": "us-west-2"}})
    options = harness._build_options(RecordingOptions, make_task(tmp_path), None)
    assert options.env.get("CLAUDE_CODE_USE_BEDROCK") == "1"
    assert options.env.get("AWS_REGION") == "us-west-2"


class FakeExecutorSession:
    """Records what the tools executed, as whom, and what governs them.

    The returned executor takes EXACTLY ``(tool_name, args)`` — the piped
    executor's contract. The first version of this fake accepted a third
    positional argument the real seam does not, which is precisely how a
    TypeError shipped: every tool call in a real run failed while the fake
    kept passing. A fake must be no more permissive than the contract.
    """

    def __init__(self):
        self.calls = []
        self.pipeline = None

    def executor_for(self, agent_id, pipeline=None):
        self.pipeline = pipeline

        def run(tool_name, args):
            self.calls.append((agent_id, tool_name, dict(args)))
            return ToolResult(success=True, output="ok")
        return run


def fake_tool(name, description, schema):
    def wrap(handler):
        return {"name": name, "description": description,
                "schema": schema, "handler": handler}
    return wrap


def test_sandbox_tools_route_through_the_agent_executor(tmp_path):
    """The tools are the L2 seam (executor_for), not a private channel: the
    agent's identity is bound in, and the shell tool lands in the task's own
    working directory rather than wherever the runtime woke up."""
    session = FakeExecutorSession()
    harness = MarathonClaudeCodeHarness({})
    tools = harness._sandbox_tools(fake_tool, session, make_task(tmp_path))

    assert [t["name"] for t in tools] == ["shell", "text_editor",
                                          "grep_files", "process"]
    # The proxy's descriptions name /testbed; this harness's tasks do not.
    assert all("/testbed" not in t["description"] for t in tools)

    shell = tools[0]
    result = asyncio.run(shell["handler"]({"command": "ls"}))
    agent_id, tool_name, args = session.calls[0]
    assert (agent_id, tool_name) == ("agent", "shell")
    assert args["working_dir"] == "/workspace/rj", \
        "the task's WORKDIR is injected when the model names none"
    assert result == {"content": [{"type": "text", "text": "ok"}],
                      "is_error": False}

    # An explicit working_dir wins; the default must not overwrite it.
    asyncio.run(shell["handler"]({"command": "ls", "working_dir": "/tmp"}))
    assert session.calls[1][2]["working_dir"] == "/tmp"


def test_tool_results_are_presented_and_bounded_but_not_nudged(tmp_path):
    """Two of the three default interceptors: the presenter (exit codes and a
    separated stderr, stated like any other agent sees them) and truncation
    (a 10-hour task's build logs otherwise spend the window). The guardrail
    stays off — its nudges shape this repo's agent, and nudging Claude Code
    too would measure a subtly different scaffold. Order matters: the
    presenter is innermost so truncation bounds what it rendered."""
    from swebench.agent.interceptors import (GuardrailInterceptor,
                                             OutcomePresenter,
                                             TruncateInterceptor)

    session = FakeExecutorSession()
    MarathonClaudeCodeHarness({"max_output_bytes": 9000})._sandbox_tools(
        fake_tool, session, make_task(tmp_path))

    mounted = session.pipeline.interceptors
    assert [type(i) for i in mounted] == [TruncateInterceptor, OutcomePresenter]
    assert not any(isinstance(i, GuardrailInterceptor) for i in mounted)
    assert mounted[0].max_len == 9000, "the config's bound reaches the chain"


def test_tools_survive_the_real_piped_executor(tmp_path):
    """The handler against the REAL AshSession.executor_for(pipeline=…), not
    the fake: the seam it returns takes (tool_name, args) and nothing else.
    A model-supplied timeout must travel in args, not as a third positional —
    the run that caught this spent 22 turns diagnosing our TypeError."""
    from unittest import mock
    from swebench.sandbox import AshSession

    session = AshSession()
    with mock.patch.object(AshSession, "_run",
                           return_value=ToolResult(success=True, output="hi")) as run:
        harness = MarathonClaudeCodeHarness({})
        tools = harness._sandbox_tools(fake_tool, session, make_task(tmp_path))
        result = asyncio.run(tools[0]["handler"](
            {"command": "sleep 1", "timeout": 1200}))
    assert result["is_error"] is False
    assert run.call_args[0][1]["timeout"] == 1200, "timeout travels in args"


def test_claude_code_does_not_launch_from_this_repository(tmp_path):
    """Claude Code reads .claude/ from its cwd: launched from the repo root,
    the agent found and invoked the repo's own `ash` skill mid-task. The cwd
    is the run's output directory (created if needed), never the repo."""
    harness = MarathonClaudeCodeHarness({})
    repo_root = Path(__file__).resolve().parents[2]

    explicit = harness._build_options(RecordingOptions, make_task(tmp_path),
                                      None, cwd=tmp_path / "out")
    assert Path(explicit.cwd) == tmp_path / "out"
    assert Path(explicit.cwd).is_dir(), "created so the CLI can start there"

    fallback = harness._build_options(RecordingOptions, make_task(tmp_path), None)
    assert repo_root not in Path(fallback.cwd).parents
    assert Path(fallback.cwd) != repo_root


def test_checkpoints_fire_at_the_tool_boundary(tmp_path):
    """An external agent's only channel into the environment is its tool
    calls, so the tool boundary is its step boundary: after_step runs once
    per call with a monotonic turn, and the tracker sits outermost on the
    chain so it also sees calls the inner interceptors reject."""
    from swebench.agent.checkpoints import MutationTracker

    class RecordingCheckpointer:
        def __init__(self):
            self.tracker = MutationTracker()
            self.turns = []

        def after_step(self, turn):
            self.turns.append(turn)

    session = FakeExecutorSession()
    checkpointer = RecordingCheckpointer()
    tools = MarathonClaudeCodeHarness({})._sandbox_tools(
        fake_tool, session, make_task(tmp_path), checkpointer=checkpointer)

    assert type(session.pipeline.interceptors[0]) is MutationTracker, \
        "the tracker mounts outermost, like install() places it"

    shell = tools[0]
    asyncio.run(shell["handler"]({"command": "echo hi"}))
    asyncio.run(shell["handler"]({"command": "echo again"}))
    assert checkpointer.turns == [1, 2]


def test_checkpointer_persists_a_resumable_map(tmp_path):
    """A killed run's snapshots are unusable without the step→snapshot map
    beside them (a 5-hour run once left 300 snapshots and no record). The
    persisted shape is the trajectory's, so load_step_snapshots and
    --resume-from read an interrupted run unchanged."""
    class SnapshotSession:
        def supports_snapshot(self):
            return True

        def environment(self):
            return {"sandbox_id": "sb-1"}

    harness = MarathonClaudeCodeHarness({})
    checkpointer = harness._install_checkpoints(
        make_task(tmp_path / "rust-java-lsp"), SnapshotSession(), tmp_path,
        {"enabled": True})
    assert checkpointer.persist is not None
    assert checkpointer.disk_only is True
    assert checkpointer.name_prefix.startswith("marathon-cc-rust-java-lsp-")

    checkpointer.persist(checkpointer)
    import json as json_module
    from swebench.replay import load_step_snapshots
    path = tmp_path / "trajectories" / "rust-java-lsp.json"
    saved = json_module.loads(path.read_text())
    assert saved["info"]["exit_status"] == "in_progress"
    assert saved["info"]["environment"] == {"sandbox_id": "sb-1"}
    assert load_step_snapshots(path) == {}, \
        "the replay tooling reads the persisted shape (empty map, no error)"


def test_two_runs_cannot_collide_on_snapshot_names(tmp_path):
    """Aliases are unique per repository; without a per-run id the second
    attempt's every capture fails, softly."""
    class SnapshotSession:
        def supports_snapshot(self):
            return True

        def environment(self):
            return {}

    harness = MarathonClaudeCodeHarness({})
    task = make_task(tmp_path / "x")
    a = harness._install_checkpoints(task, SnapshotSession(), tmp_path, {})
    b = harness._install_checkpoints(task, SnapshotSession(), tmp_path, {})
    assert a.name_prefix != b.name_prefix


def test_failures_carry_the_marathon_report_shape():
    """The batch runner reads reward/partial_score off every report; a report
    missing them is a KeyError three layers up."""
    report = MarathonClaudeCodeHarness({}).run_instance({"instance_id": "x"},
                                                        output_dir=None)
    assert report["grading_error"] == "no task_dir given"
    for key in ("reward", "partial_score", "cost", "turns", "metrics"):
        assert key in report
