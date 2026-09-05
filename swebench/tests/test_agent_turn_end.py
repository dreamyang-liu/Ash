from __future__ import annotations

import json
from types import SimpleNamespace

from swebench.agent import AshAgent
from swebench.models import AgentConfig, ToolResult


def _message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


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
