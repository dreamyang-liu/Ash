"""Checkpoint-aware Ash agent loop backed by one Miles v2 SessionTree.

This strategy is deliberately separate from ``agent_loop``.  The ordinary
agent-loop strategy creates one sandbox and one Miles session per sample.  This
variant keeps one session for the whole prompt group, records one exact
model-facing checkpoint after the first tool turn, and resumes child agents
from that environment/message boundary.  Miles therefore sees a real sibling
branch instead of an unrelated second session.
"""

from __future__ import annotations

import copy
import sys
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..protocol import GeneratedSpan, RolloutGroupRequest, RolloutGroupResult, Trajectory
from ..runner import RolloutCancelled, RolloutContext
from ...agent import AshAgent
from ...models import AgentConfig, ToolResult
from .agent_loop import (
    MilesSessionClient,
    _initial_messages,
    _run_async,
    _tool_call_count,
    _trajectory_status,
)


class CheckpointAgentLoopRolloutStrategy:
    """Run a parent and restored child agents through one shared Miles session.

    This validation strategy deliberately selects the first tool-complete
    checkpoint. Production policies can replace the strategy without changing
    the HTTP protocol or the environment checkpoint contract.
    """

    def __init__(
        self,
        *,
        agent_config: AgentConfig | None = None,
        session_server_endpoint: str | None = None,
        allow_text_prompt: bool = True,
    ) -> None:
        self.agent_config = agent_config or AgentConfig(model="openai/local")
        self.session_server_endpoint = session_server_endpoint.rstrip("/") if session_server_endpoint else None
        self.allow_text_prompt = allow_text_prompt

    def run(self, request: RolloutGroupRequest, context: RolloutContext) -> RolloutGroupResult:
        if context.environment_provider is None:
            raise RuntimeError("checkpoint-agent-loop requires an environment provider")
        endpoint = request.session_server_endpoint or self.session_server_endpoint or request.model_endpoint
        if "/sessions" in endpoint:
            raise ValueError("session_server_endpoint must be the Miles server base URL, not a session URL")

        slots = list(request.sample_slots[: request.max_samples])
        if len(slots) < 2:
            raise ValueError("checkpoint-agent-loop requires at least two allocated sample slots")
        if request.budgets.max_model_calls < len(slots):
            raise ValueError(
                "checkpoint-agent-loop requires at least one model call per allocated sample slot"
            )

        model_calls_per_agent, parent_model_remainder = divmod(
            request.budgets.max_model_calls, len(slots)
        )
        tool_calls_per_agent, parent_tool_remainder = divmod(
            request.budgets.max_tool_calls, len(slots)
        )

        client = MilesSessionClient(endpoint, timeout_seconds=context.remaining_wall_time_seconds or 120.0)
        session_id = client.create()
        parent = None
        children: list[Any] = []
        parent_checkpoint: dict[str, Any] = {}
        try:
            parent = context.environment_provider.spawn(request)
            parent_slot = slots[0]

            def capture_first_checkpoint(step_id: int, messages: list[dict[str, Any]]) -> None:
                if (
                    parent_checkpoint
                    or step_id <= 0
                    or not any(message.get("role") == "tool" for message in messages)
                ):
                    return
                # AshAgent calls this hook only after all tool results for the
                # turn have entered the model-visible history.  Snapshot and
                # message list therefore describe one recoverable joint point.
                # AgentENV treats snapshot names as immutable aliases.  A
                # rollout job may be retried with the same logical id, so the
                # physical alias must still be unique per checkpoint attempt.
                checkpoint = context.environment_provider.create_checkpoint(
                    parent,
                    owner_job_id=request.rollout_job_id,
                    name=(
                        f"{request.rollout_job_id}-branch-step-{step_id}-"
                        f"{uuid.uuid4().hex[:12]}"
                    ),
                )
                parent_checkpoint.update(
                    checkpoint=checkpoint,
                    step_id=step_id,
                    messages=copy.deepcopy(messages),
                )

            # Divide the group budget without exceeding it. Remainders go to
            # the parent because it must first reach a tool-complete boundary.
            parent_status, parent_calls, parent_tools, parent_messages = self._run_agent(
                request=request,
                context=context,
                client=client,
                session_id=session_id,
                slot=parent_slot,
                sandbox=parent,
                initial_messages=_initial_messages(request.prompt, allow_text=self.allow_text_prompt),
                agent_id=f"{request.rollout_job_id}:{parent_slot.sample_slot_id}",
                max_model_calls=model_calls_per_agent + parent_model_remainder,
                max_tool_calls=tool_calls_per_agent + parent_tool_remainder,
                on_turn_end=capture_first_checkpoint,
            )
            if not parent_checkpoint:
                raise RuntimeError("parent agent did not produce a tool-complete checkpoint")

            checkpoint = parent_checkpoint["checkpoint"]

            parent_state = client.get(session_id)
            trajectories = [
                _trajectory_from_session(
                    request,
                    parent_slot.sample_slot_id,
                    parent_state,
                    parent_status,
                    branch_id=f"{request.rollout_job_id}:root:{parent_slot.sample_index}",
                    messages=parent_messages,
                    metadata={
                        "strategy": "checkpoint-agent-loop-v1",
                        "session_id": session_id,
                        "sandbox_id": getattr(parent, "sandbox_id", "unknown"),
                        "environment_checkpoint_id": checkpoint.checkpoint_id,
                    },
                )
            ]
            consumed_model_calls = parent_calls
            consumed_tool_calls = parent_tools

            for slot in slots[1:]:
                context.check_cancelled()
                child = context.environment_provider.restore_checkpoint(
                    checkpoint,
                    agent_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                )
                children.append(child)
                child_status, child_calls, child_tools, child_messages = self._run_agent(
                    request=request,
                    context=context,
                    client=client,
                    session_id=session_id,
                    slot=slot,
                    sandbox=child,
                    initial_messages=copy.deepcopy(parent_checkpoint["messages"]),
                    agent_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                    max_model_calls=model_calls_per_agent,
                    max_tool_calls=tool_calls_per_agent,
                )
                consumed_model_calls += child_calls
                consumed_tool_calls += child_tools
                child_state = client.get(session_id)
                trajectories.append(
                    _trajectory_from_session(
                        request,
                        slot.sample_slot_id,
                        child_state,
                        child_status,
                        branch_id=f"{request.rollout_job_id}:child:{slot.sample_index}",
                        parent_branch_id=f"{request.rollout_job_id}:root:{parent_slot.sample_index}",
                        branch_point_token_count=_branch_input_length(
                            child_state, parent_checkpoint["messages"]
                        ),
                        messages=child_messages,
                        metadata={
                            "strategy": "checkpoint-agent-loop-v1",
                            "session_id": session_id,
                            "sandbox_id": getattr(child, "sandbox_id", "unknown"),
                            "environment_checkpoint_id": checkpoint.checkpoint_id,
                            "parent_checkpoint_messages": len(parent_checkpoint["messages"]),
                        },
                    )
                )

            # This is intentionally a real request, not a local leaf count:
            # Miles assembles and validates every SessionTree leaf here.
            client.collect_samples(session_id)
            final_state = client.get(session_id)
            tree = (final_state.get("metadata") or {}).get("tree") or {}
            leaf_count = len(tree.get("leaves") or [])
            if leaf_count < len(trajectories):
                raise RuntimeError(
                    f"Miles SessionTree returned {leaf_count} leaves for {len(trajectories)} trajectories"
                )

            return RolloutGroupResult(
                rollout_job_id=request.rollout_job_id,
                prompt_group_id=request.prompt_group_id,
                status="completed",
                max_samples=request.max_samples,
                trajectories=trajectories,
                search_branches=len(trajectories) - 1,
                consumed_budget={
                    # A child trajectory contains the generated checkpoint
                    # prefix for training, but reusing that prefix does not
                    # execute those model/tool calls again.
                    "model_calls": consumed_model_calls,
                    "tool_calls": consumed_tool_calls,
                    "session_tree_leaves": leaf_count,
                },
            )
        finally:
            active_error = sys.exc_info()[1]
            cleanup_errors: list[Exception] = []
            for child in children:
                try:
                    context.environment_provider.destroy(child)
                except Exception as exc:
                    cleanup_errors.append(exc)
            if parent_checkpoint:
                try:
                    context.environment_provider.release_checkpoint(
                        parent_checkpoint["checkpoint"]
                    )
                except Exception as exc:
                    cleanup_errors.append(exc)
            if parent is not None:
                try:
                    context.environment_provider.destroy(parent)
                except Exception as exc:
                    cleanup_errors.append(exc)
            try:
                client.delete(session_id)
            except Exception as exc:
                cleanup_errors.append(exc)
            if active_error is None and cleanup_errors:
                raise RuntimeError(
                    f"checkpoint rollout cleanup failed: {cleanup_errors[0]}"
                ) from cleanup_errors[0]

    def _run_agent(
        self,
        *,
        request: RolloutGroupRequest,
        context: RolloutContext,
        client: MilesSessionClient,
        session_id: str,
        slot,
        sandbox,
        initial_messages: list[dict[str, Any]],
        agent_id: str,
        max_model_calls: int,
        max_tool_calls: int,
        on_turn_end=None,
    ) -> tuple[str, int, int, list[dict[str, Any]]]:
        config = self._config(request, client, session_id, max_model_calls)
        executor, executed_tool_calls = self._executor(
            sandbox,
            agent_id=agent_id,
            context=context,
            max_tool_calls=max_tool_calls,
        )
        agent = AshAgent(
            config,
            executor=executor,
            agent_id=agent_id,
            sandbox_id=str(getattr(sandbox, "sandbox_id", "unknown")),
        )
        agent.stream = False
        # Ash's durable trajectory uses ``tool_result`` internally, while the
        # model-facing/OpenAI history must use ``tool``.  Keep the latter as
        # the checkpoint and returned trajectory representation.
        latest_messages: list[dict[str, Any]] = []

        def capture_messages(step: int, messages: list[dict[str, Any]]) -> None:
            latest_messages[:] = copy.deepcopy(messages)
            if on_turn_end is not None:
                on_turn_end(step, messages)

        agent.on_turn_end = capture_messages
        status = agent.run(task="", instance_id=slot.sample_slot_id, initial_messages=initial_messages)
        if not latest_messages:
            errors = [
                str(message.get("content") or "")
                for message in agent.trajectory.messages
                if message.get("role") == "error"
            ]
            detail = errors[-1] if errors else "no recorded model error"
            raise RuntimeError(
                f"AshAgent stopped before its first model-visible response "
                f"(status={status}): {detail}"
            )
        requested_tool_calls = _tool_call_count(latest_messages) - _tool_call_count(initial_messages)
        if requested_tool_calls < 0:
            raise RuntimeError("agent history lost tool results from its checkpoint prefix")
        return status, agent.cost.api_calls, executed_tool_calls(), latest_messages

    def _config(self, request, client, session_id: str, max_model_calls: int) -> AgentConfig:
        sampling = dict(request.sampling_params)
        model = request.model or sampling.pop("model", None) or self.agent_config.model
        if "/" not in str(model):
            model = f"openai/{model}"
        max_tokens = sampling.get("max_tokens", sampling.get("max_new_tokens", self.agent_config.max_tokens))
        temperature = sampling.get("temperature", self.agent_config.temperature)
        extra_body = sampling.get("extra_body")
        if extra_body is None and "chat_template_kwargs" in sampling:
            extra_body = {"chat_template_kwargs": sampling["chat_template_kwargs"]}
        return replace(
            self.agent_config,
            model=str(model),
            api_base=f"{client.endpoint}/sessions/{session_id}/v1",
            api_key=self.agent_config.api_key or "EMPTY",
            max_tokens=int(max_tokens),
            temperature=None if temperature is None else float(temperature),
            prompt_cache=False,
            extra_body=extra_body,
            step_limit=min(self.agent_config.step_limit, max_model_calls),
        )

    @staticmethod
    def _executor(
        sandbox,
        *,
        agent_id: str,
        context: RolloutContext,
        max_tool_calls: int,
    ) -> tuple[Callable[[str, dict[str, Any]], ToolResult], Callable[[], int]]:
        call = getattr(sandbox, "call_agent_tool", None) or getattr(sandbox, "call", None)
        if call is None:
            session_executor = getattr(sandbox, "executor_for", None)
            if session_executor is not None:
                call = session_executor(agent_id)
        if call is None:
            raise TypeError("environment sandbox must expose call_agent_tool or call")
        calls = 0

        def execute(name: str, args: dict[str, Any]) -> ToolResult:
            nonlocal calls
            context.check_cancelled()
            if calls >= max_tool_calls:
                raise RolloutCancelled("rollout tool-call budget exhausted")
            calls += 1
            result = _run_async(call(name, args))
            if isinstance(result, ToolResult):
                return result
            return ToolResult.from_sdk(result)

        return execute, lambda: calls



def _branch_input_length(state: dict[str, Any], checkpoint_messages: list[dict[str, Any]]) -> int:
    """Return the first child request's actual tokenized input length."""
    for record in state.get("records") or []:
        request = record.get("request") or {}
        if request.get("messages") == checkpoint_messages:
            input_ids = request.get("input_ids") or []
            if input_ids:
                return len(input_ids)
    raise ValueError("Miles session did not expose the child request at the checkpoint message boundary")


def _trajectory_from_session(
    request: RolloutGroupRequest,
    sample_slot_id: str,
    state: dict[str, Any],
    status: str,
    *,
    branch_id: str,
    messages: list[dict[str, Any]],
    parent_branch_id: str | None = None,
    branch_point_token_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    records = state.get("records")
    session_metadata = state.get("metadata") or {}
    token_ids = session_metadata.get("accumulated_token_ids")
    if not isinstance(records, list) or not records or not isinstance(token_ids, list) or not token_ids:
        raise ValueError("Miles session did not return records and accumulated_token_ids")

    spans: list[GeneratedSpan] = []
    for record in records:
        req = record.get("request") or {}
        response_envelope = record.get("response") or {}
        response = response_envelope.get("choices", [{}])[0]
        info = response.get("meta_info") or {}
        pairs = info.get("output_token_logprobs") or []
        output_ids = [int(pair[1]) for pair in pairs]
        input_ids = [int(token) for token in req.get("input_ids", [])]
        if not output_ids or not input_ids:
            raise ValueError("Miles session record lacks token-level generation data")
        logs = [float(pair[0]) for pair in pairs] if request.return_rollout_logprobs else None
        spans.append(
            GeneratedSpan(
                response_id=str(response_envelope.get("id") or f"{request.rollout_job_id}:{len(spans)}"),
                start=len(input_ids),
                end=len(input_ids) + len(output_ids),
                input_token_ids=tuple(input_ids),
                output_token_ids=tuple(output_ids),
                weight_version=str(info.get("weight_version") or request.expected_weight_version or "unknown"),
                finish_reason=str(response.get("finish_reason") or "stop"),
                output_token_log_probs=None if logs is None else tuple(logs),
            )
        )

    response_text = ""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            response_text = str(message.get("content") or "")
            break
    prompt_length = len(records[0].get("request", {}).get("input_ids", []))
    return Trajectory(
        sample_slot_id=sample_slot_id,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        branch_point_token_count=branch_point_token_count,
        messages=copy.deepcopy(messages),
        token_ids=[int(token) for token in token_ids],
        prompt_length=prompt_length,
        generated_spans=spans,
        response_text=response_text,
        status=_trajectory_status(status),
        metadata={**(metadata or {}), "session_tree": session_metadata.get("tree", {})},
    )
