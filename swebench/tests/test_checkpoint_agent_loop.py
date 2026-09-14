from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import swebench.rollout_groups.strategies.checkpoint_agent_loop as checkpoint_module
from swebench.rollout_groups.protocol import RolloutGroupRequest
from swebench.rollout_groups.runner import RolloutContext
from swebench.rollout_groups.runner import EnvironmentCheckpoint
from swebench.rollout_groups.strategies.checkpoint_agent_loop import CheckpointAgentLoopRolloutStrategy


def _request() -> RolloutGroupRequest:
    return RolloutGroupRequest.from_dict(
        {
            "rollout_job_id": "job-branch",
            "rollout_id": 0,
            "prompt_group_id": "group-branch",
            "task_id": "task-branch",
            "environment_ref": {
                "kind": "template",
                "id": "swebench-runtime",
                "revision": "sha256:test",
                "resource_profile": "standard",
            },
            "sample_slots": [
                {"sample_slot_id": "slot-parent", "sample_index": 0},
                {"sample_slot_id": "slot-child", "sample_index": 1},
            ],
            "max_samples": 2,
            "minimum_returned_samples": 2,
            "prompt": [{"role": "user", "content": "use the tool"}],
            "prompt_token_ids": [10],
            "model_endpoint": "http://model",
            "session_server_endpoint": "http://session",
            "expected_weight_version": "3",
            "return_rollout_logprobs": False,
            "sampling_params": {},
            "budgets": {
                "max_model_calls": 6,
                "max_tool_calls": 2,
                "max_wall_time_seconds": 30,
            },
        }
    )


def _record(messages, input_ids, output_id, response_id):
    return {
        "request": {"messages": messages, "input_ids": input_ids},
        "response": {
            "id": response_id,
            "choices": [
                {
                    "message": {"role": "assistant", "content": response_id},
                    "finish_reason": "stop",
                    "meta_info": {
                        "weight_version": "3",
                        "output_token_logprobs": [[-0.1, output_id]],
                    },
                }
            ],
        },
    }


class _SessionClient:
    instances = []

    def __init__(self, endpoint, timeout_seconds):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.get_calls = 0
        self.collected = False
        self.deleted = False
        self.__class__.instances.append(self)

    def create(self):
        return "session-1"

    def get(self, session_id):
        assert session_id == "session-1"
        self.get_calls += 1
        prompt = [{"role": "user", "content": "use the tool"}]
        checkpoint = prompt + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ]
        shared = _record(prompt, [10], 11, "shared")
        if self.get_calls == 1:
            records = [shared, _record(checkpoint, [10, 11, 12], 13, "parent-final")]
            tokens = [10, 11, 12, 13]
        else:
            records = [shared, _record(checkpoint, [10, 11, 12], 14, "child-final")]
            tokens = [10, 11, 12, 14]
        return {
            "records": records,
            "metadata": {
                "accumulated_token_ids": tokens,
                "tree": {"leaves": ["parent", "child"]},
            },
        }

    def collect_samples(self, session_id):
        assert session_id == "session-1"
        self.collected = True
        return b"samples"

    def delete(self, session_id):
        assert session_id == "session-1"
        self.deleted = True


class _Environment:
    def __init__(self):
        self.parent = SimpleNamespace(sandbox_id="parent")
        self.child = SimpleNamespace(sandbox_id="child")
        self.destroyed = []
        self.released = []

    def spawn(self, _request):
        return self.parent

    def create_checkpoint(self, sandbox, *, owner_job_id, name):
        assert sandbox is self.parent
        assert name.startswith("job-branch-branch-step-1-")
        return EnvironmentCheckpoint(
            checkpoint_id="snapshot-1",
            owner_job_id=owner_job_id,
            source_sandbox_id="parent",
            backend="agentenv-microvm",
            state_scope="full-runtime",
            multiple_restore=True,
            explicit_release=True,
        )

    def restore_checkpoint(self, checkpoint, *, agent_id=""):
        assert checkpoint.checkpoint_id == "snapshot-1"
        assert agent_id.endswith(":slot-child")
        return self.child

    def release_checkpoint(self, checkpoint):
        self.released.append((checkpoint.source_sandbox_id, checkpoint.checkpoint_id))
        return True

    def destroy(self, sandbox):
        self.destroyed.append(sandbox.sandbox_id)


def test_checkpoint_agent_loop_restores_child_and_releases_resources(monkeypatch):
    _SessionClient.instances.clear()
    monkeypatch.setattr(checkpoint_module, "MilesSessionClient", _SessionClient)
    environment = _Environment()
    strategy = CheckpointAgentLoopRolloutStrategy()
    checkpoint_messages = [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]

    def run_agent(**kwargs):
        if kwargs["slot"].sample_index == 0:
            kwargs["on_turn_end"](1, checkpoint_messages)
            return "completed", 3, 1, checkpoint_messages + [
                {"role": "assistant", "content": "parent-final"}
            ], 0.25
        assert kwargs["initial_messages"] == checkpoint_messages
        return "completed", 2, 0, checkpoint_messages + [
            {"role": "assistant", "content": "child-final"}
        ], 0.10

    monkeypatch.setattr(strategy, "_run_agent", run_agent)
    result = strategy.run(
        _request(),
        RolloutContext(
            cancel_event=SimpleNamespace(is_set=lambda: False),
            model_client=None,
            environment_provider=environment,
            job_id="job-branch",
        ),
    )

    assert result.actual_samples == 2
    assert result.search_branches == 1
    assert result.consumed_budget == {
        "model_calls": 5,
        "tool_calls": 1,
        "session_tree_leaves": 2,
    }
    root, child = result.trajectories
    assert child.parent_branch_id == root.branch_id
    assert child.branch_point_token_count == 3
    assert child.status == "completed"
    assert environment.destroyed == ["child", "parent"]
    assert environment.released == [("parent", "snapshot-1")]
    assert _SessionClient.instances[0].collected is True
    assert _SessionClient.instances[0].deleted is True


def test_checkpoint_agent_loop_preserves_spawn_error_and_deletes_session(monkeypatch):
    _SessionClient.instances.clear()
    monkeypatch.setattr(checkpoint_module, "MilesSessionClient", _SessionClient)

    class FailingEnvironment:
        def spawn(self, _request):
            raise RuntimeError("environment unavailable")

    strategy = CheckpointAgentLoopRolloutStrategy()
    with pytest.raises(RuntimeError, match="environment unavailable"):
        strategy.run(
            _request(),
            RolloutContext(
                cancel_event=SimpleNamespace(is_set=lambda: False),
                model_client=None,
                environment_provider=FailingEnvironment(),
                job_id="job-branch",
            ),
        )

    assert _SessionClient.instances[0].deleted is True


def test_checkpoint_agent_loop_rejects_insufficient_model_call_budget(monkeypatch):
    _SessionClient.instances.clear()
    monkeypatch.setattr(checkpoint_module, "MilesSessionClient", _SessionClient)
    request = _request()
    request = replace(
        request,
        budgets=replace(request.budgets, max_model_calls=1),
    )

    with pytest.raises(ValueError, match="one model call per allocated sample slot"):
        CheckpointAgentLoopRolloutStrategy().run(
            request,
            RolloutContext(
                cancel_event=SimpleNamespace(is_set=lambda: False),
                model_client=None,
                environment_provider=_Environment(),
                job_id="job-branch",
            ),
        )

    assert _SessionClient.instances == []


def test_checkpoint_agent_loop_reports_release_failure_after_other_cleanup(monkeypatch):
    _SessionClient.instances.clear()
    monkeypatch.setattr(checkpoint_module, "MilesSessionClient", _SessionClient)
    environment = _Environment()

    def fail_release(_checkpoint):
        raise RuntimeError("snapshot release failed")

    environment.release_checkpoint = fail_release
    strategy = CheckpointAgentLoopRolloutStrategy()
    checkpoint_messages = [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]

    def run_agent(**kwargs):
        if kwargs["slot"].sample_index == 0:
            kwargs["on_turn_end"](1, checkpoint_messages)
        return "completed", 1, 0, checkpoint_messages + [
            {"role": "assistant", "content": "done"}
        ], 0.05

    monkeypatch.setattr(strategy, "_run_agent", run_agent)
    with pytest.raises(RuntimeError, match="checkpoint rollout cleanup failed"):
        strategy.run(
            _request(),
            RolloutContext(
                cancel_event=SimpleNamespace(is_set=lambda: False),
                model_client=None,
                environment_provider=environment,
                job_id="job-branch",
            ),
        )

    assert environment.destroyed == ["child", "parent"]
    assert _SessionClient.instances[0].deleted is True
