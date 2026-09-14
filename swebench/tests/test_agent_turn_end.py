from __future__ import annotations

import json
from types import SimpleNamespace

from swebench.agent import AshAgent
from swebench.agent.conversation import Conversation
from swebench.models import AgentConfig, ToolResult


def _message(content="", tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
    )


def _tool_call(call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="shell", arguments=json.dumps({"command": "printf ready"})
        ),
    )


def test_turn_end_is_after_tool_results_and_is_a_copy(monkeypatch):
    seen = []
    responses = iter([
        _message(tool_calls=[_tool_call()]),
        _message(content="done"),
        _message(content="done"),
    ])
    agent = AshAgent(
        AgentConfig(step_limit=5, cost_limit=100),
        executor=lambda _name, _args: ToolResult(success=True, output="ready"),
        on_turn_end=lambda step, messages: seen.append((step, messages)),
    )

    def fake_query(*_args):
        agent.cost.api_calls += 1
        return next(responses)

    monkeypatch.setattr(agent, "_query", fake_query)
    assert agent.run("test task") == "completed"
    assert len(seen) == 3
    step, messages = seen[0]
    assert step == 1
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "shell"
    assert messages[-1] == {"role": "tool", "tool_call_id": "call-1", "content": "ready"}
    assert len(messages) == 4


def test_empty_model_choices_becomes_explicit_agent_error():
    agent = AshAgent(
        AgentConfig(step_limit=1, cost_limit=100),
        executor=lambda _name, _args: ToolResult(success=True, output="ready"),
    )
    conversation = Conversation(agent.trajectory)
    llm = SimpleNamespace(
        query_with_recovery=lambda _messages: SimpleNamespace(choices=[])
    )

    assert agent._query(llm, conversation, 1) is None
    assert agent.last_model_error == "RuntimeError: model response contained no choices"
    assert agent.trajectory.messages[-1] == {
        "role": "error",
        "content": "model response contained no choices",
    }


def test_length_finish_stops_without_executing_partial_tool_call(monkeypatch):
    executed = []
    agent = AshAgent(
        AgentConfig(step_limit=5, cost_limit=100),
        executor=lambda name, args: executed.append((name, args)) or ToolResult(
            success=True, output="unexpected"
        ),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=_message(content="partial", tool_calls=[_tool_call()]),
            finish_reason="length",
        )]
    )

    monkeypatch.setattr(
        "swebench.agent.llm.LLMClient.query_with_recovery",
        lambda _self, _messages: response,
    )

    assert agent.run("test task") == "length_limit"
    assert executed == []
    assert agent.last_model_finish_reason == "length"
    assert agent.trajectory.messages[-1]["content"] == "partial"
