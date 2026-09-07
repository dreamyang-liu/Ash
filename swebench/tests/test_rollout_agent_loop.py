from __future__ import annotations

from types import SimpleNamespace

import pytest

import swebench.rollout_groups.strategies.agent_loop as agent_loop_module
from swebench.rollout_groups.protocol import RolloutGroupRequest
from swebench.rollout_groups.runner import RolloutContext
from swebench.rollout_groups.strategies.agent_loop import (
    MilesSessionAgentRolloutStrategy,
    _trajectory_from_session,
    _trajectory_status,
)


def _request(**overrides):
    value = {
        "rollout_job_id": "job",
        "rollout_id": 0,
        "prompt_group_id": "group",
        "sample_slots": [{"sample_slot_id": "slot", "sample_index": 0}],
        "max_samples": 1,
        "minimum_returned_samples": 1,
        "prompt": [{"role": "user", "content": "hello"}],
        "prompt_token_ids": [10, 11],
        "model_endpoint": "http://router:30000",
        "session_server_endpoint": "http://session:31000",
        "model": "openai/local",
        "expected_weight_version": "7",
        "return_rollout_logprobs": True,
        "sampling_params": {},
        "budgets": {"max_model_calls": 2, "max_tool_calls": 2, "max_wall_time_seconds": 10},
    }
    value.update(overrides)
    return RolloutGroupRequest.from_dict(value)


def test_session_records_become_multi_span_trajectory():
    request = _request()
    state = {
        "records": [
            {
                "request": {"input_ids": [10, 11], "messages": request.prompt},
                "response": {
                    "id": "r1",
                    "choices": [{
                        "message": {"role": "assistant", "content": "first"},
                        "finish_reason": "tool_calls",
                        "meta_info": {"weight_version": "7", "output_token_logprobs": [[-0.1, 12]]},
                    }],
                },
            },
            {
                "request": {
                    "input_ids": [10, 11, 12, 13],
                    "messages": request.prompt + [{"role": "assistant", "content": "first"},
                                                     {"role": "tool", "tool_call_id": "c1", "content": "ok"}],
                },
                "response": {
                    "id": "r2",
                    "choices": [{
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                        "meta_info": {"weight_version": "7", "output_token_logprobs": [[-0.2, 14], [-0.3, 15]]},
                    }],
                },
            },
        ],
        "metadata": {"accumulated_token_ids": [10, 11, 12, 13, 14, 15], "tree": {"nodes": []}},
    }
    trajectory = _trajectory_from_session(request, "slot", state, "completed")
    assert trajectory.token_ids == [10, 11, 12, 13, 14, 15]
    assert trajectory.prompt_length == 2
    assert [span.response_id for span in trajectory.generated_spans] == ["r1", "r2"]
    assert trajectory.generated_spans[1].input_token_ids == (10, 11, 12, 13)
    assert trajectory.generated_spans[1].output_token_log_probs == (-0.2, -0.3)


def test_strategy_uses_explicit_session_endpoint():
    request = _request()
    strategy = MilesSessionAgentRolloutStrategy()
    assert request.session_server_endpoint == "http://session:31000"
    assert strategy.agent_config.prompt_cache is True


def test_agent_exit_status_maps_to_trajectory_status():
    assert _trajectory_status("completed") == "completed"
    assert _trajectory_status("step_limit") == "truncated"
    assert _trajectory_status("cost_limit") == "truncated"
    assert _trajectory_status("error") == "failed"


class _SessionClient:
    instances = []

    def __init__(self, endpoint, *, timeout_seconds):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.deleted = False
        self.__class__.instances.append(self)

    def create(self):
        return f"session-{len(self.instances)}"

    def get(self, _session_id):
        return {
            "records": [
                {
                    "request": {"input_ids": [10, 11], "messages": [{"role": "user", "content": "hello"}]},
                    "response": {
                        "id": "response",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "done"},
                                "finish_reason": "stop",
                                "meta_info": {"weight_version": "7", "output_token_logprobs": [[-0.1, 12]]},
                            }
                        ],
                    },
                }
            ],
            "metadata": {"accumulated_token_ids": [10, 11, 12], "tree": {}},
        }

    def delete(self, _session_id):
        self.deleted = True


class _Environment:
    def __init__(self):
        self.spawned = []
        self.destroyed = []

    def spawn(self, _request):
        sandbox = SimpleNamespace(sandbox_id=f"sandbox-{len(self.spawned)}")
        self.spawned.append(sandbox)
        return sandbox

    def destroy(self, sandbox):
        self.destroyed.append(sandbox.sandbox_id)


def test_agent_loop_distributes_group_budgets_across_slots(monkeypatch):
    _SessionClient.instances.clear()
    monkeypatch.setattr(agent_loop_module, "MilesSessionClient", _SessionClient)
    request = _request(
        sample_slots=[
            {"sample_slot_id": "slot-0", "sample_index": 0},
            {"sample_slot_id": "slot-1", "sample_index": 1},
        ],
        max_samples=2,
        minimum_returned_samples=2,
        budgets={"max_model_calls": 3, "max_tool_calls": 2, "max_wall_time_seconds": 10},
    )
    environment = _Environment()
    strategy = MilesSessionAgentRolloutStrategy()
    allocations = []

    def run_agent(**kwargs):
        allocations.append((kwargs["max_model_calls"], kwargs["max_tool_calls"]))
        return "completed", kwargs["max_model_calls"], kwargs["max_tool_calls"], [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "done"},
        ]

    monkeypatch.setattr(strategy, "_run_agent", run_agent)
    result = strategy.run(
        request,
        RolloutContext(
            cancel_event=SimpleNamespace(is_set=lambda: False),
            model_client=None,
            environment_provider=environment,
            job_id="job",
        ),
    )

    assert allocations == [(2, 1), (1, 1)]
    assert result.consumed_budget == {"model_calls": 3, "tool_calls": 2}
    assert environment.destroyed == ["sandbox-0", "sandbox-1"]
    assert all(client.deleted for client in _SessionClient.instances)


def test_agent_loop_destroys_sandbox_when_session_cleanup_fails(monkeypatch):
    class FailingDeleteClient(_SessionClient):
        def delete(self, _session_id):
            raise RuntimeError("session delete failed")

    monkeypatch.setattr(agent_loop_module, "MilesSessionClient", FailingDeleteClient)
    environment = _Environment()
    strategy = MilesSessionAgentRolloutStrategy()
    monkeypatch.setattr(
        strategy,
        "_run_agent",
        lambda **_kwargs: (
            "completed",
            1,
            0,
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "done"}],
        ),
    )

    with pytest.raises(RuntimeError, match="agent rollout cleanup failed"):
        strategy.run(
            _request(),
            RolloutContext(
                cancel_event=SimpleNamespace(is_set=lambda: False),
                model_client=None,
                environment_provider=environment,
                job_id="job",
            ),
        )

    assert environment.destroyed == ["sandbox-0"]
