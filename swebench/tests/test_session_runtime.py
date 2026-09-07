from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import swebench.rollout_groups.session_runtime as runtime_module
from swebench.models import ToolResult
from swebench.rollout_groups.protocol import RolloutGroupRequest
from swebench.rollout_groups.runner import RolloutContext
from swebench.rollout_groups.session_runtime import MilesSessionClient, SessionAgentStrategySupport


def _request() -> RolloutGroupRequest:
    return RolloutGroupRequest.from_dict(
        {
            "rollout_job_id": "job",
            "rollout_id": 0,
            "prompt_group_id": "group",
            "sample_slots": [{"sample_slot_id": "slot", "sample_index": 0}],
            "max_samples": 1,
            "minimum_returned_samples": 1,
            "prompt": [{"role": "user", "content": "use a tool"}],
            "prompt_token_ids": [10],
            "model_endpoint": "http://model",
            "session_server_endpoint": "http://miles-session",
            "model": "local-model",
            "expected_weight_version": "7",
            "return_rollout_logprobs": False,
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.4},
            "budgets": {
                "max_model_calls": 3,
                "max_tool_calls": 1,
                "max_wall_time_seconds": 10,
            },
        }
    )


def test_session_agent_support_runs_tool_and_captures_model_history(monkeypatch):
    seen = {}

    class FakeAgent:
        def __init__(self, config, *, executor, agent_id, sandbox_id):
            seen.update(config=config, agent_id=agent_id, sandbox_id=sandbox_id)
            self.executor = executor
            self.cost = SimpleNamespace(api_calls=2)
            self.trajectory = SimpleNamespace(messages=[])
            self.on_turn_end = None

        def run(self, *, task, instance_id, initial_messages):
            assert task == ""
            assert instance_id == "slot"
            assert self.executor("shell", {"command": "true"}).success is True
            self.on_turn_end(
                1,
                initial_messages + [{"role": "assistant", "content": "done"}],
            )
            return "completed"

    monkeypatch.setattr(runtime_module, "AshAgent", FakeAgent)
    sandbox = SimpleNamespace(
        sandbox_id="sandbox",
        call=lambda _name, _args: ToolResult(success=True, output="ok"),
    )
    support = SessionAgentStrategySupport()
    request = _request()

    status, model_calls, tool_calls, messages = support._run_agent(
        request=request,
        context=RolloutContext(
            cancel_event=SimpleNamespace(is_set=lambda: False),
            model_client=None,
            environment_provider=None,
            job_id="job",
        ),
        client=MilesSessionClient("http://miles-session"),
        session_id="session",
        slot=request.sample_slots[0],
        sandbox=sandbox,
        initial_messages=support.initial_messages(request),
        agent_id="job:slot",
        max_model_calls=2,
        max_tool_calls=1,
    )

    assert (status, model_calls, tool_calls) == ("completed", 2, 1)
    assert messages[-1] == {"role": "assistant", "content": "done"}
    assert seen["config"].api_base == "http://miles-session/sessions/session/v1"
    assert seen["config"].step_limit == 2
    assert seen["config"].max_tokens == 32
    assert seen["config"].temperature == 0.4
    assert seen["agent_id"] == "job:slot"
    assert seen["sandbox_id"] == "sandbox"


def test_session_client_accepts_empty_success_body_for_delete():
    class EmptyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b""

    with patch.object(runtime_module.urllib.request, "urlopen", return_value=EmptyResponse()):
        MilesSessionClient("http://miles-session").delete("session")
