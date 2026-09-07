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
from typing import Any

from ..protocol import RolloutGroupRequest, RolloutGroupResult
from ..runner import RolloutContext
from ..session_runtime import (
    MilesSessionClient,
    SessionAgentStrategySupport,
    branch_input_length,
    trajectory_from_session,
)
from ...models import AgentConfig


class CheckpointAgentLoopRolloutStrategy(SessionAgentStrategySupport):
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
        super().__init__(
            agent_config=agent_config,
            session_server_endpoint=session_server_endpoint,
            allow_text_prompt=allow_text_prompt,
        )

    def run(self, request: RolloutGroupRequest, context: RolloutContext) -> RolloutGroupResult:
        if context.environment_provider is None:
            raise RuntimeError("checkpoint-agent-loop requires an environment provider")
        endpoint = self.session_endpoint(request)

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
                initial_messages=self.initial_messages(request),
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
                trajectory_from_session(
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
                        "session_tree": (parent_state.get("metadata") or {}).get("tree", {}),
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
                    trajectory_from_session(
                        request,
                        slot.sample_slot_id,
                        child_state,
                        child_status,
                        branch_id=f"{request.rollout_job_id}:child:{slot.sample_index}",
                        parent_branch_id=f"{request.rollout_job_id}:root:{parent_slot.sample_index}",
                        branch_point_token_count=branch_input_length(
                            child_state, parent_checkpoint["messages"]
                        ),
                        messages=child_messages,
                        metadata={
                            "strategy": "checkpoint-agent-loop-v1",
                            "session_id": session_id,
                            "sandbox_id": getattr(child, "sandbox_id", "unknown"),
                            "environment_checkpoint_id": checkpoint.checkpoint_id,
                            "parent_checkpoint_messages": len(parent_checkpoint["messages"]),
                            "session_tree": (child_state.get("metadata") or {}).get("tree", {}),
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
