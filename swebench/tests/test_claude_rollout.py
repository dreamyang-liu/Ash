from __future__ import annotations

import importlib.util
import threading
from types import SimpleNamespace

import pytest

from swebench.rollout_groups.claude_runtime import ClaudeAgentRuntime, ClaudeRunResult
from swebench.rollout_groups.protocol import RolloutGroupRequest
from swebench.rollout_groups.runner import EnvironmentCheckpoint, RolloutContext
from swebench.rollout_groups.strategies.claude_checkpoint_agent_loop import (
    ClaudeCheckpointAgentLoopRolloutStrategy,
)


def _request() -> RolloutGroupRequest:
    return RolloutGroupRequest.from_dict(
        {
            "rollout_job_id": "claude-job",
            "rollout_id": 0,
            "prompt_group_id": "claude-group",
            "task_id": "task",
            "environment_ref": {
                "kind": "template",
                "id": "runtime",
                "revision": "r1",
                "resource_profile": "standard",
            },
            "sample_slots": [
                {"sample_slot_id": "parent", "sample_index": 0},
                {"sample_slot_id": "child", "sample_index": 1},
            ],
            "max_samples": 2,
            "minimum_returned_samples": 2,
            "prompt": "use shell",
            "prompt_token_ids": [10],
            "model_endpoint": "http://model",
            "session_server_endpoint": "http://session",
            "model": "local",
            "expected_weight_version": "1",
            "return_rollout_logprobs": False,
            "sampling_params": {},
            "budgets": {
                "max_model_calls": 6,
                "max_tool_calls": 2,
                "max_wall_time_seconds": 30,
            },
        }
    )


def _record(response_id, input_ids, output_id, message, messages):
    return {
        "request": {"messages": messages, "input_ids": input_ids},
        "response": {
            "id": response_id,
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    "meta_info": {
                        "weight_version": "1",
                        "output_token_logprobs": [[-0.1, output_id]],
                    },
                }
            ],
        },
    }


class _SessionClient:
    instance = None

    def __init__(self, endpoint, timeout_seconds):
        self.endpoint = endpoint
        self.deleted = False
        self.collected = False
        self.active = "parent"
        self.__class__.instance = self

    def require_capabilities(self, *required):
        assert "anthropic-messages" in required

    def create(self):
        return "miles-session"

    def get(self, _session_id):
        prompt = [{"role": "user", "content": "use shell"}]
        tool_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tool-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        }
        prefix = _record("tool-response", [10], 11, tool_call, prompt)
        if self.active == "parent":
            final_id, output_id = "parent-response", 13
        else:
            final_id, output_id = "child-response", 14
        tool_result = {"role": "tool", "tool_call_id": "tool-1", "content": "ok"}
        final = _record(
            final_id,
            [10, 11, 12],
            output_id,
            {"role": "assistant", "content": final_id},
            prompt + [tool_call, tool_result],
        )
        return {
            "records": [prefix, final],
            "metadata": {
                "accumulated_token_ids": [10, 11, 12, output_id],
                "tree": {
                    "nodes": [
                        {
                            "id": 0,
                            "response_id": "tool-response",
                            "completion_span": [1, 2],
                        },
                        {
                            "id": 1,
                            "response_id": final_id,
                            "completion_span": [3, 4],
                        },
                    ],
                    "leaves": [{"node_id": 1}, {"node_id": 2}],
                },
            },
        }

    def collect_samples(self, _session_id):
        self.collected = True
        return b"samples"

    def delete(self, _session_id):
        self.deleted = True


class _Environment:
    def __init__(self):
        self.parent = SimpleNamespace(sandbox_id="parent")
        self.child = SimpleNamespace(sandbox_id="child")
        self.destroyed = []
        self.released = []

    def spawn(self, _request):
        return self.parent

    def restore_checkpoint(self, checkpoint, *, agent_id):
        assert checkpoint.checkpoint_id == "checkpoint-1"
        assert agent_id.endswith(":child")
        return self.child

    def release_checkpoint(self, checkpoint):
        self.released.append(checkpoint.checkpoint_id)

    def destroy(self, sandbox):
        self.destroyed.append(sandbox.sandbox_id)


class _ClaudeRuntime:
    def run_parent(self, **kwargs):
        checkpoint = EnvironmentCheckpoint(
            checkpoint_id="checkpoint-1",
            owner_job_id="claude-job",
            source_sandbox_id="parent",
            backend="agentenv-microvm",
            state_scope="full-runtime",
            multiple_restore=True,
            explicit_release=True,
        )
        kwargs["on_checkpoint"](checkpoint)
        return ClaudeRunResult(
            status="completed",
            session_id="claude-parent",
            messages=[{"message_id": "parent-response"}],
            model_calls=2,
            tool_calls=1,
            tool_seconds=0.2,
            checkpoint=checkpoint,
            checkpoint_tool_use_id="tool-1",
            checkpoint_message_uuid="tool-result-uuid",
            model_response_ids=["tool-response", "parent-response"],
        )

    def run_child(self, **kwargs):
        assert kwargs["parent_claude_session_id"] == "claude-parent"
        assert kwargs["checkpoint_message_uuid"] == "tool-result-uuid"
        _SessionClient.instance.active = "child"
        return ClaudeRunResult(
            status="completed",
            session_id="claude-child",
            messages=[{"message_id": "child-response"}],
            model_calls=1,
            tool_calls=0,
            tool_seconds=0.0,
            synthetic_message_uuids=["synthetic-user"],
            model_response_ids=["child-response"],
        )


def test_claude_checkpoint_strategy_joins_sdk_and_environment_positions(monkeypatch):
    import swebench.rollout_groups.strategies.claude_checkpoint_agent_loop as module

    monkeypatch.setattr(module, "MilesSessionClient", _SessionClient)
    environment = _Environment()
    result = ClaudeCheckpointAgentLoopRolloutStrategy(
        runtime=_ClaudeRuntime()
    ).run(
        _request(),
        RolloutContext(
            cancel_event=threading.Event(),
            model_client=None,
            environment_provider=environment,
            job_id="claude-job",
        ),
    )

    assert result.actual_samples == 2
    assert result.search_branches == 1
    assert result.consumed_budget == {
        "model_calls": 3,
        "tool_calls": 1,
        "session_tree_leaves": 2,
    }
    parent, child = result.trajectories
    assert parent.prompt_token_alignment == "harness_rendered"
    assert child.prompt_token_alignment == "harness_rendered"
    assert child.parent_branch_id == parent.branch_id
    # The joint checkpoint is after the tool result, so the branch boundary is
    # the child continuation's model input, not the earlier tool-call output.
    assert child.branch_point_token_count == 3
    assert child.metadata["synthetic_message_uuids"] == ["synthetic-user"]
    assert environment.destroyed == ["child", "parent"]
    assert environment.released == ["checkpoint-1"]
    assert _SessionClient.instance.collected is True
    assert _SessionClient.instance.deleted is True


def test_claude_tool_call_moves_sync_ash_adapter_off_event_loop():
    import asyncio

    from swebench.rollout_groups.claude_runtime import _call_tool

    caller_thread = threading.get_ident()
    observed = {}

    def call(name, args):
        observed.update(name=name, args=args, thread=threading.get_ident())
        return "ok"

    panel = SimpleNamespace(route=lambda name, args: (name, args))
    result = asyncio.run(_call_tool(call, "shell", panel, {"command": "pwd"}))

    assert result == "ok"
    assert observed == {
        "name": "shell",
        "args": {"command": "pwd"},
        "thread": observed["thread"],
    }
    assert observed["thread"] != caller_thread


@pytest.mark.skipif(
    importlib.util.find_spec("claude_agent_sdk") is None,
    reason="Claude Agent SDK is optional",
)
def test_claude_runtime_maps_sdk_response_ids_and_tool_results(monkeypatch, tmp_path):
    import swebench.rollout_groups.claude_runtime as module
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    captured_options = None

    async def fake_query(*, prompt, options):
        nonlocal captured_options
        captured_options = options
        assert prompt == "use shell"
        yield AssistantMessage(
            content=[TextBlock(text="calling shell")],
            model="local",
            message_id="response-1",
        )
        yield AssistantMessage(
            content=[ToolUseBlock(id="tool-1", name="mcp__ash-sandbox__shell", input={})],
            model="local",
            message_id="response-1",
        )
        yield UserMessage(
            content=[ToolResultBlock(tool_use_id="tool-1", content="ok")],
            uuid="tool-result-uuid",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="claude-parent",
            result="done",
            terminal_reason="completed",
        )

    fake_sdk = SimpleNamespace(
        AssistantMessage=AssistantMessage,
        UserMessage=UserMessage,
        ResultMessage=ResultMessage,
        ClaudeAgentOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        create_sdk_mcp_server=lambda *_args, **_kwargs: object(),
        tool=lambda *_args, **_kwargs: lambda handler: handler,
        query=fake_query,
    )
    monkeypatch.setattr(module, "_load_claude_sdk", lambda: fake_sdk)
    monkeypatch.setattr(
        module,
        "_build_claude_panel",
        lambda _name: SimpleNamespace(
            schema=[
                {
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {},
                    }
                }
                for name in ClaudeAgentRuntime.tool_names
            ],
            route=lambda name, args: (name, args),
        ),
    )
    sandbox = SimpleNamespace(
        call=lambda _name, _args: SimpleNamespace(
            success=True, output="ok", error=None
        )
    )
    result = ClaudeAgentRuntime(model="local").run_parent(
        request=_request(),
        context=RolloutContext(
            cancel_event=threading.Event(),
            model_client=None,
            environment_provider=SimpleNamespace(
                create_checkpoint=lambda *_args, **_kwargs: EnvironmentCheckpoint(
                    checkpoint_id="checkpoint-1",
                    owner_job_id="claude-job",
                    source_sandbox_id="parent",
                    backend="agentenv-microvm",
                    state_scope="full-runtime",
                    multiple_restore=True,
                    explicit_release=True,
                )
            ),
            job_id="claude-job",
        ),
        session_client=SimpleNamespace(endpoint="http://session", delete=lambda _id: None),
        miles_session_id="miles-session",
        sandbox=sandbox,
        agent_id="parent",
        max_model_calls=2,
        max_tool_calls=1,
        capture_checkpoint=True,
        config_dir=tmp_path,
    )

    assert result.model_calls == 1
    assert result.model_response_ids == ["response-1"]
    assert result.checkpoint_tool_use_id == "tool-1"
    assert result.checkpoint_message_uuid == "tool-result-uuid"


@pytest.mark.skipif(
    importlib.util.find_spec("claude_agent_sdk") is None,
    reason="Claude Agent SDK is optional",
)
def test_claude_runtime_checkpoints_after_the_complete_tool_batch(monkeypatch, tmp_path):
    import swebench.rollout_groups.claude_runtime as module
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    checkpoint_observations = []

    async def fake_query(*, prompt, options):
        yield AssistantMessage(
            content=[
                ToolUseBlock(id="tool-1", name="mcp__ash-sandbox__shell", input={}),
                ToolUseBlock(id="tool-2", name="mcp__ash-sandbox__grep_files", input={}),
            ],
            model="local",
            message_id="response-1",
        )
        yield UserMessage(
            content=[ToolResultBlock(tool_use_id="tool-1", content="one")],
            uuid="tool-result-1",
        )
        assert checkpoint_observations == []
        yield UserMessage(
            content=[ToolResultBlock(tool_use_id="tool-2", content="two")],
            uuid="tool-result-2",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="claude-parent",
            result="done",
            terminal_reason="completed",
        )

    fake_sdk = SimpleNamespace(
        AssistantMessage=AssistantMessage,
        UserMessage=UserMessage,
        ResultMessage=ResultMessage,
        ClaudeAgentOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        create_sdk_mcp_server=lambda *_args, **_kwargs: object(),
        tool=lambda *_args, **_kwargs: lambda handler: handler,
        query=fake_query,
    )
    monkeypatch.setattr(module, "_load_claude_sdk", lambda: fake_sdk)
    monkeypatch.setattr(
        module,
        "_build_claude_panel",
        lambda _name: SimpleNamespace(
            schema=[
                {
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {},
                    }
                }
                for name in ClaudeAgentRuntime.tool_names
            ],
            route=lambda name, args: (name, args),
        ),
    )
    checkpoint = EnvironmentCheckpoint(
        checkpoint_id="checkpoint-batch",
        owner_job_id="claude-job",
        source_sandbox_id="parent",
        backend="agentenv-microvm",
        state_scope="full-runtime",
        multiple_restore=True,
        explicit_release=True,
    )

    def create_checkpoint(*_args, **_kwargs):
        checkpoint_observations.append("created")
        return checkpoint

    result = ClaudeAgentRuntime(model="local").run_parent(
        request=_request(),
        context=RolloutContext(
            cancel_event=threading.Event(),
            model_client=None,
            environment_provider=SimpleNamespace(
                create_checkpoint=create_checkpoint
            ),
            job_id="claude-job",
        ),
        session_client=SimpleNamespace(endpoint="http://session", delete=lambda _id: None),
        miles_session_id="miles-session",
        sandbox=SimpleNamespace(
            call=lambda _name, _args: SimpleNamespace(
                success=True, output="ok", error=None
            )
        ),
        agent_id="parent",
        max_model_calls=2,
        max_tool_calls=2,
        capture_checkpoint=True,
        config_dir=tmp_path,
    )

    assert checkpoint_observations == ["created"]
    assert result.checkpoint_tool_use_id == "tool-2"
    assert result.checkpoint_message_uuid == "tool-result-2"
