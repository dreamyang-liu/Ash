from __future__ import annotations

from swebench.rollout_groups.protocol import RolloutGroupRequest
from swebench.rollout_groups.strategies.agent_loop import (
    MilesSessionAgentRolloutStrategy,
    _trajectory_from_session,
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
