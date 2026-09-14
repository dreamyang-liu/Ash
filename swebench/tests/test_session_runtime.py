from __future__ import annotations

from dataclasses import replace
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import swebench.rollout_groups.session_runtime as runtime_module
from swebench.models import ToolResult
from swebench.rollout_groups.protocol import RolloutGroupRequest
from swebench.rollout_groups.runner import RolloutCancelled, RolloutContext
from swebench.rollout_groups.session_runtime import MilesSessionClient, SessionAgentStrategySupport


def _request() -> RolloutGroupRequest:
    return RolloutGroupRequest.from_dict(
        {
            "rollout_job_id": "job",
            "rollout_id": 0,
            "prompt_group_id": "group",
            "task_id": "task",
            "environment_ref": {
                "kind": "template",
                "id": "swebench-runtime",
                "revision": "sha256:test",
                "resource_profile": "standard",
            },
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
    progress = []

    class FakeAgent:
        def __init__(self, config, *, executor, agent_id, sandbox_id):
            seen.update(config=config, agent_id=agent_id, sandbox_id=sandbox_id)
            self.executor = executor
            self.cost = SimpleNamespace(api_calls=2)
            self.trajectory = SimpleNamespace(messages=[])
            self.on_turn_end = None
            self.before_query_hooks = []

        def run(self, *, task, instance_id, initial_messages):
            assert task == ""
            assert instance_id == "task"
            for hook in self.before_query_hooks:
                hook(self, SimpleNamespace(messages=initial_messages))
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

    status, model_calls, tool_calls, messages, tool_seconds = support._run_agent(
        request=request,
        context=RolloutContext(
            cancel_event=SimpleNamespace(is_set=lambda: False),
            model_client=None,
            environment_provider=None,
            job_id="job",
            progress_callback=lambda **values: progress.append(values),
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
    assert tool_seconds >= 0
    assert messages[-1] == {"role": "assistant", "content": "done"}
    assert seen["config"].api_base == "http://miles-session/sessions/session/v1"
    assert seen["config"].step_limit == 2
    assert seen["config"].max_tokens == 32
    assert seen["config"].temperature == 0.4
    assert seen["agent_id"] == "job:slot"
    assert seen["sandbox_id"] == "sandbox"
    assert any(item.get("model_calls") == 3 for item in progress)


def test_session_agent_support_deletes_session_during_model_call_on_cancel(monkeypatch):
    cancel_event = threading.Event()
    session_deleted = threading.Event()

    class BlockingAgent:
        def __init__(self, *_args, **_kwargs):
            self.cost = SimpleNamespace(api_calls=1)
            self.trajectory = SimpleNamespace(messages=[])
            self.on_turn_end = None

        def run(self, **_kwargs):
            cancel_event.set()
            assert session_deleted.wait(2.0), "cancellation did not reach Miles session"
            return "error"

    class RecordingSessionClient:
        endpoint = "http://miles-session"

        def delete(self, session_id):
            assert session_id == "session"
            session_deleted.set()

    monkeypatch.setattr(runtime_module, "AshAgent", BlockingAgent)
    support = SessionAgentStrategySupport()
    request = _request()
    sandbox = SimpleNamespace(
        sandbox_id="sandbox",
        call=lambda _name, _args: ToolResult(success=True, output="ok"),
    )

    with pytest.raises(RolloutCancelled, match="rollout job was cancelled"):
        support._run_agent(
            request=request,
            context=RolloutContext(
                cancel_event=cancel_event,
                model_client=None,
                environment_provider=None,
                job_id="job",
            ),
            client=RecordingSessionClient(),
            session_id="session",
            slot=request.sample_slots[0],
            sandbox=sandbox,
            initial_messages=support.initial_messages(request),
            agent_id="job:slot",
            max_model_calls=2,
            max_tool_calls=1,
        )

    assert session_deleted.is_set()


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


def test_session_client_requires_reported_capabilities():
    client = MilesSessionClient("http://miles-session")
    with patch.object(
        client,
        "_request",
        return_value={
            "status": "ok",
            "capabilities": [
                "session-tree-v2",
                "context-aware-completion-cap",
            ],
        },
    ) as request:
        health = client.require_capabilities(
            "session-tree-v2", "context-aware-completion-cap"
        )

    assert health["status"] == "ok"
    request.assert_called_once_with("GET", "/health", None)


@pytest.mark.parametrize(
    "health, message",
    [
        ({"status": "ok"}, "does not report capabilities"),
        (
            {"status": "ok", "capabilities": ["session-tree-v2"]},
            "context-aware-completion-cap",
        ),
    ],
)
def test_session_client_rejects_incompatible_endpoint(health, message):
    client = MilesSessionClient("http://miles-session")
    with patch.object(client, "_request", return_value=health):
        with pytest.raises(RuntimeError, match=message):
            client.require_capabilities(
                "session-tree-v2", "context-aware-completion-cap"
            )


def test_agent_config_maps_only_supported_sampling_fields():
    support = SessionAgentStrategySupport()
    request = _request()
    request = replace(
        request,
        sampling_params={
            "max_new_tokens": 64,
            "temperature": 0.7,
            "seed": 20260913,
            "top_p": 0.9,
            "top_k": 20,
            "stop": ["</tool>"],
            "stop_token_ids": [2],
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    config = support._agent_config(
        request,
        MilesSessionClient("http://miles-session"),
        "session",
        max_model_calls=3,
    )

    assert config.max_tokens == 64
    assert config.temperature == 0.7
    assert config.extra_body == {
        "seed": 20260913,
        "top_p": 0.9,
        "top_k": 20,
        "stop": ["</tool>"],
        "stop_token_ids": [2],
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert not hasattr(config, "top_p")


def test_agent_config_caps_request_timeout_at_remaining_wall_time():
    support = SessionAgentStrategySupport(
        agent_config=runtime_module.AgentConfig(
            model="openai/local",
            request_timeout=1800.0,
            request_max_retries=0,
            retry_attempts=3,
        )
    )

    config = support._agent_config(
        _request(),
        MilesSessionClient("http://miles-session"),
        "session",
        max_model_calls=None,
        remaining_wall_time_seconds=900.0,
    )

    assert config.request_timeout == 900.0
    assert config.request_max_retries == 0
    assert config.retry_attempts == 3


def test_agent_config_rejects_non_object_extra_body():
    support = SessionAgentStrategySupport()
    request = replace(_request(), sampling_params={"extra_body": "invalid"})

    with pytest.raises(ValueError, match="extra_body must be an object"):
        support._agent_config(
            request,
            MilesSessionClient("http://miles-session"),
            "session",
            max_model_calls=3,
        )


def test_agent_config_uses_no_step_cap_for_unbounded_model_calls():
    config = SessionAgentStrategySupport()._agent_config(
        _request(),
        MilesSessionClient("http://miles-session"),
        "session",
        max_model_calls=None,
    )

    assert config.step_limit == sys.maxsize
    assert config.cost_limit == sys.float_info.max


def test_session_agent_support_propagates_model_failure(monkeypatch):
    class FailedAgent:
        def __init__(self, *_args, **_kwargs):
            self.cost = SimpleNamespace(api_calls=0)
            self.trajectory = SimpleNamespace(messages=[])
            self.last_model_error = "RuntimeError: model response contained no choices"
            self.on_turn_end = None

        def run(self, **_kwargs):
            return "error"

    monkeypatch.setattr(runtime_module, "AshAgent", FailedAgent)
    support = SessionAgentStrategySupport()
    request = _request()
    sandbox = SimpleNamespace(
        sandbox_id="sandbox",
        call=lambda _name, _args: ToolResult(success=True, output="ok"),
    )

    with pytest.raises(RuntimeError, match="model response contained no choices"):
        support._run_agent(
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
