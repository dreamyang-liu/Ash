from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from ash_sandbox.pool import Snapshot
from ash_sandbox.result import ToolResult as SdkResult
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from harbor.models.agent.context import AgentContext
from harness.core.journal import JournalWriter, read_journal
from harness.core.result import CommandOutcome, ToolResult
from harness.execution.interceptors import MutationTracker
from harness.execution.pipeline import ToolPipeline
from harness.execution.server import Session, SessionHandler
from harness.slots.claude_code import ClaudeCodeSlot
from harness.orchestrator.run import RunOutcome
from terminalbench.agentenv import AgentENVEnvironment, shell_command
from terminalbench.agentenv_agent import AgentENVClaudeCode, AgentENVOrchestrator
from terminalbench.eval import build_config, parse_args


def environment(tmp_path: Path, **changes) -> AgentENVEnvironment:
    directory = tmp_path / "environment"
    directory.mkdir(exist_ok=True)
    runtime = tmp_path / "runtime"
    runtime.touch()
    config = changes.pop("task_env_config", EnvironmentConfig(docker_image="image:tag", cpus=2, memory_mb=1024))
    return AgentENVEnvironment(environment_dir=directory, environment_name="task", session_id="test",
                               task_env_config=config, trial_paths=TrialPaths(trial_dir=tmp_path),
                               runtime_bin=str(runtime), **changes)


def test_default_config_wires_agentenv_environment_and_checkpoint_actor():
    config = build_config(parse_args(["--model", "test-model"]))
    assert config.environment.type is None
    assert config.environment.import_path == "terminalbench.agentenv:AgentENVEnvironment"
    assert config.agents[0].import_path == "terminalbench.agentenv_agent:AgentENVClaudeCode"
    assert config.environment.kwargs["checkpoint_mode"] == "full"
    assert config.environment.kwargs["sandbox_ttl"] >= 30000


@pytest.mark.parametrize("change", [
    {"task_env_config": EnvironmentConfig(docker_image="image", gpus=1)},
    {"network_policy": NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["example.com"])},
    {"task_env_config": EnvironmentConfig(docker_image="image", storage_mb=1536)},
    {"task_env_config": EnvironmentConfig(docker_image="image", storage_mb=8192)},
    {"phase_network_policies": [NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)]},
])
def test_unsupported_requirements_fail_before_start(tmp_path, change):
    with pytest.raises((ValueError, RuntimeError)):
        environment(tmp_path, **change)


def test_compose_is_not_silently_reduced_to_one_service(tmp_path):
    directory = tmp_path / "environment"
    directory.mkdir()
    (directory / "docker-compose.yaml").write_text("services: {}")
    with pytest.raises(ValueError, match="multi-service"):
        environment(tmp_path)


def test_shell_quotes_arguments_and_honors_user_environment():
    wrapped = shell_command("printf '%s' \"$TOKEN\"", "/work space", {"TOKEN": "x'; touch /wrong; '"}, "123")
    assert "runuser" in wrapped and "getent passwd 123" in wrapped
    assert "cd '/work space'" in wrapped.replace("'\"'\"'", "'")
    assert "TOKEN=" in wrapped


def test_exec_preserves_exit_status_and_does_not_confuse_json_stdout(tmp_path):
    env = environment(tmp_path)
    calls = []

    def execute(name, arguments, timeout):
        calls.append((name, arguments, timeout))
        return ToolResult(success=False, output="", outcome=CommandOutcome(exit_code=7, stdout='{"stdout":"data"}', stderr="failed"))

    env.session = SimpleNamespace(execute=execute)
    env.default_user = "nobody"
    result = asyncio.run(env.exec("test command", cwd="/work", env={"MODE": "test"}, timeout_sec=50))
    assert result.return_code == 7 and result.stdout == '{"stdout":"data"}'
    assert "nobody" in calls[0][1]["command"]
    assert calls[0][1]["timeout"] == 50 and calls[0][2] == 110


def test_transport_failure_is_not_an_exit_code(tmp_path):
    env = environment(tmp_path)
    env.session = SimpleNamespace(execute=lambda *args: ToolResult(False, "", error="connection lost"))
    with pytest.raises(RuntimeError, match="connection lost"):
        asyncio.run(env.exec("test"))


@pytest.mark.parametrize("mode,disk_only", [("full", False), ("disk_only", True)])
def test_real_tool_hook_pipeline_and_bridge_capture_after_execution(tmp_path, mode, disk_only):
    env = environment(tmp_path, checkpoint_mode=mode)
    states = []
    captured = []

    class Sandbox:
        def supports_snapshot(self):
            return True

        async def call(self, name, **arguments):
            states.append(arguments["command"])
            return SdkResult("ok", False)

        def snapshot(self, **kwargs):
            captured.append((list(states), kwargs["disk_only"]))
            return Snapshot(f"snapshot-{len(captured)}")

    env.session = Sandbox()
    orchestrator = AgentENVOrchestrator(env, {}, out_dir=tmp_path)
    tracker = MutationTracker()
    owned = SimpleNamespace(tracker=tracker, server=SimpleNamespace(boundary=None))
    with JournalWriter(tmp_path / "trajectory.jsonl") as journal:
        bridge = orchestrator._wire_checkpoints(None, journal, owned)
        journal.emit("session.ref", native_session_id="native-test")
        slot = ClaudeCodeSlot()
        slot._journal = journal
        slot._checkpoint_server = "ash"
        entry = SimpleNamespace(id="sandbox", sandbox=env.session, visible_to=lambda _: True)
        handler = SessionHandler(Session(id="agent", groups=["owner:agent"], bound_id="sandbox"),
                                 SimpleNamespace(get=lambda _: entry),
                                 pipeline=ToolPipeline([tracker, orchestrator.policy]), boundary=owned.server.boundary)

        async def run():
            for number in range(2):
                call_id = f"call-{number}"
                verdict = await slot._pre_tool_use({"tool_name": "mcp__ash__shell",
                                                   "tool_input": {"command": f"write-{number}"}}, call_id)
                result = await handler.call_tool("shell", verdict["hookSpecificOutput"]["updatedInput"])
                journal.emit("tool.finished", call_id=call_id, status="ok", output=result["text"])
        asyncio.run(run())
        bridge.finalize_calls()
        bridge.close()
    assert [len(state) for state, mode in captured] == [1, 2]
    assert all(mode == disk_only for state, mode in captured)
    records = read_journal(tmp_path / "trajectory.jsonl")
    points = [row for row in records if row["type"] == "checkpoint.captured" and row.get("snapshot_id")]
    assert {row["call_id"] for row in points} == {"call-0", "call-1"}


def test_cpu_memory_disk_are_sent_through_template_and_cold_start(tmp_path, monkeypatch):
    from terminalbench import agentenv

    env = environment(tmp_path, task_env_config=EnvironmentConfig(docker_image="image:tag", cpus=3, memory_mb=2048, storage_mb=65536))
    calls = []

    class FakeSession:
        sandbox_id = "sandbox"

        def __init__(self, **kwargs):
            self.backend = kwargs["backend"]

        def create(self, image, resources):
            calls.append((image, resources))
            return True

        def execute(self, *args):
            return ToolResult(True, "")

    monkeypatch.setattr(agentenv, "SandboxSession", FakeSession)
    monkeypatch.setattr(env, "_prepare_image", lambda force: ("image@sha256:digest", {"WorkingDir": "/work", "User": "nobody"}))

    async def no_upload():
        pass

    monkeypatch.setattr(env, "_upload_environment_dir_after_start", no_upload)
    asyncio.run(env.start())
    assert calls == [("image@sha256:digest", {"cpu": 3, "memory_mb": 2048, "disk_size_mb": 65536})]
    assert env.image_user == "nobody" and env.image_workdir == "/work"


@pytest.mark.parametrize("paired", [True, False])
def test_actor_requires_a_snapshot_for_each_successful_sandbox_call(tmp_path, monkeypatch, paired):
    env = environment(tmp_path)
    env.session = SimpleNamespace(on_swap=[], snapshot=lambda **kwargs: Snapshot("final"))
    original_session = env.session

    def run(self, spec):
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.started", slot="test", task_prompt=spec.prompt)
            journal.emit("tool.started", call_id="call", name="mcp__ash__shell", args={"command": "write"})
            journal.emit("tool.finished", call_id="call", status="ok", output="")
            if paired:
                journal.emit("checkpoint.captured", call_id="call", step=1, snapshot_id="step-one")
            journal.emit("run.finished", status="completed", usage={})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="completed", checkpoints=1)

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    agent = AgentENVClaudeCode(logs_dir=tmp_path / "agent", model_name="test")
    context = AgentContext()
    if paired:
        asyncio.run(agent.run("task", env, context))
        assert context.metadata["final_snapshot_id"] == "final"
    else:
        with pytest.raises(RuntimeError, match="lack snapshots"):
            asyncio.run(agent.run("task", env, context))
    assert env.session is original_session
    assert (tmp_path / "agent/trajectory.json").exists()
