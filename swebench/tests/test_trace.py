from __future__ import annotations

import json
from types import SimpleNamespace

from swebench.agent import AshAgent
from swebench.agent.conversation import Conversation
from swebench.agent.guardrails import Guardrails
from swebench.agent.trace import ToolTraceWriter
from swebench.models import AgentConfig, ToolResult, Trajectory


def _tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _agent_with_trace(path, executor):
    agent = AshAgent(AgentConfig(), executor=executor)
    agent._event_trace = ToolTraceWriter(
        path, run_id="run-1", agent_id="worker-2", sandbox_id="shared",
    )
    return agent


def test_tool_trace_records_turn_sequence_routing_and_background_process(tmp_path):
    path = tmp_path / "trace.events.jsonl"
    calls = []

    def execute(name, args):
        calls.append((name, args))
        return ToolResult(success=True, output='{"pid":"p123"}')

    agent = _agent_with_trace(path, execute)
    conv = Conversation(Trajectory())
    agent._run_tool(
        _tool_call("provider-call", "shell", {"command": "pytest", "background": True}),
        conv,
        Guardrails(),
        turn_id="turn-7",
    )
    agent._event_trace.close()

    started, finished = _events(path)
    assert [started["seq"], finished["seq"]] == [1, 2]
    assert started["turn_id"] == finished["turn_id"] == "turn-7"
    assert started["call_id"] == finished["call_id"]
    assert started["agent"] == {
        "name": "shell",
        "args": {"command": "pytest", "background": True},
    }
    assert started["runtime"] == started["agent"]
    assert started["run_id"] == "run-1"
    assert started["agent_id"] == "worker-2"
    assert started["sandbox_id"] == "shared"
    assert calls == [("shell", {"command": "pytest", "background": True})]
    assert finished["status"] == "ok"
    assert finished["process_id"] == "p123"
    assert finished["result"]["output_bytes"] == len(b'{"pid":"p123"}')
    assert "error_kind" not in finished
    assert "observation" not in finished


def test_tool_trace_classifies_routing_failure(tmp_path):
    path = tmp_path / "trace.events.jsonl"
    agent = _agent_with_trace(
        path, lambda name, args: ToolResult(success=True, output="should not run"),
    )
    conv = Conversation(Trajectory())
    agent._run_tool(
        _tool_call("provider-call", "unknown_tool", {"value": 1}),
        conv,
        Guardrails(),
        turn_id="turn-2",
    )
    agent._event_trace.close()

    _, finished = _events(path)
    assert finished["status"] == "error"
    assert finished["error_kind"] == "routing"
    assert "unknown agent tool" in finished["result"]["error"]
    assert finished["observation"].startswith("Error:")


def test_tool_trace_classifies_runtime_failure_and_records_truncation(tmp_path):
    path = tmp_path / "trace.events.jsonl"
    output = "head\n... [output truncated: 2000 bytes total] ...\ntail"
    agent = _agent_with_trace(
        path,
        lambda name, args: ToolResult(success=False, output=output, error="exit 1"),
    )
    conv = Conversation(Trajectory())
    agent._run_tool(
        _tool_call("provider-call", "shell", {"command": "fail"}),
        conv,
        Guardrails(),
        turn_id="turn-3",
    )
    agent._event_trace.close()

    _, finished = _events(path)
    assert finished["status"] == "error"
    assert finished["error_kind"] == "runtime"
    assert finished["result"]["output_truncated"] is True
    assert finished["duration_ms"] >= 0


def test_process_snapshot_propagates_runtime_truncation_flag(tmp_path):
    path = tmp_path / "trace.events.jsonl"
    output = json.dumps({
        "stdout": "bounded",
        "stderr": "",
        "running": True,
        "exit_code": None,
        "stdout_truncated": True,
        "stderr_truncated": False,
    })
    agent = _agent_with_trace(
        path,
        lambda name, args: ToolResult(success=True, output=output),
    )
    agent._run_tool(
        _tool_call("provider-call", "process", {"pid": "p123", "action": "read"}),
        Conversation(Trajectory()),
        Guardrails(),
        turn_id="turn-4",
    )
    agent._event_trace.close()

    _, finished = _events(path)
    assert finished["result"]["output_truncated"] is True
