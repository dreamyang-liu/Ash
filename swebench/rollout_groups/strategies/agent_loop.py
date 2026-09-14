"""Ash agent-loop strategy backed by a Miles v2 session server.

This adapter is intentionally small: Ash owns the agent/tool loop and the
environment, while Miles owns tokenization, session-tree records and the
training sample.  A future branch policy can replace this strategy without
changing the rollout-groups protocol.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from ..protocol import RolloutGroupRequest, RolloutGroupResult, Trajectory
from ..runner import RolloutContext
from ..session_runtime import (
    MilesSessionClient,
    SessionAgentStrategySupport,
    trajectory_from_session as _trajectory_from_session_common,
    trajectory_status as _trajectory_status,
    session_model_seconds,
)
from ...models import AgentConfig


class MilesSessionAgentRolloutStrategy(SessionAgentStrategySupport):
    """Run one AshAgent per allocated slot through Miles v2 sessions."""

    def __init__(
        self,
        *,
        agent_config: AgentConfig | None = None,
        session_server_endpoint: str | None = None,
        allow_text_prompt: bool = True,
    ) -> None:
        # Miles exposes an OpenAI-compatible local endpoint.  Do not inherit
        # AshAgent's Anthropic-oriented standalone default unless the caller
        # explicitly supplies an AgentConfig.
        super().__init__(
            agent_config=agent_config,
            session_server_endpoint=session_server_endpoint,
            allow_text_prompt=allow_text_prompt,
        )

    def run(self, request: RolloutGroupRequest, context: RolloutContext) -> RolloutGroupResult:
        if context.environment_provider is None:
            raise RuntimeError("agent-loop rollout requires an environment provider")
        endpoint = self.session_endpoint(request)
        first_session_client = MilesSessionClient(
            endpoint, timeout_seconds=context.remaining_wall_time_seconds or 120.0
        )
        require_capabilities = getattr(first_session_client, "require_capabilities", None)
        if callable(require_capabilities):
            require_capabilities("session-tree-v2", "context-aware-completion-cap")
        trajectories: list[Trajectory] = []
        model_calls = 0
        tool_calls = 0
        slots = list(request.sample_slots[: request.max_samples])
        if (
            request.budgets.max_model_calls is not None
            and request.budgets.max_model_calls < len(slots)
        ):
            raise ValueError("agent-loop requires at least one model call per allocated sample slot")
        if request.budgets.max_model_calls is None:
            model_calls_per_agent = model_remainder = None
        else:
            model_calls_per_agent, model_remainder = divmod(
                request.budgets.max_model_calls, len(slots)
            )
        if request.budgets.max_tool_calls is None:
            tool_calls_per_agent = tool_remainder = None
        else:
            tool_calls_per_agent, tool_remainder = divmod(
                request.budgets.max_tool_calls, len(slots)
            )
        for position, slot in enumerate(slots):
            context.check_cancelled()
            context.update_progress(
                "creating_environment",
                model_calls=model_calls,
                tool_calls=tool_calls,
                completed_samples=len(trajectories),
                active_sample_slot_id=slot.sample_slot_id,
            )
            setup_started = time.monotonic()
            sandbox = context.environment_provider.spawn(request)
            sandbox_setup_seconds = time.monotonic() - setup_started
            session_client = (
                first_session_client
                if position == 0
                else MilesSessionClient(
                    endpoint,
                    timeout_seconds=context.remaining_wall_time_seconds or 120.0,
                )
            )
            session_id = None
            try:
                prepared = context.prepare_task(request, sandbox)
                session_id = session_client.create()
                status, calls, tools, messages, tool_seconds = self._run_agent(
                    request=request,
                    context=context,
                    client=session_client,
                    session_id=session_id,
                    slot=slot,
                    sandbox=sandbox,
                    initial_messages=self.initial_messages(request),
                    agent_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                    max_model_calls=(
                        None
                        if model_calls_per_agent is None
                        else model_calls_per_agent + (position < model_remainder)
                    ),
                    max_tool_calls=(
                        None
                        if tool_calls_per_agent is None
                        else tool_calls_per_agent + (position < tool_remainder)
                    ),
                    model_calls_offset=model_calls,
                    tool_calls_offset=tool_calls,
                )
                model_calls += calls
                tool_calls += tools
                state = session_client.get(session_id)
                trajectory = _trajectory_from_session_common(
                    request,
                    slot.sample_slot_id,
                    state,
                    status,
                    branch_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                    messages=messages,
                    metadata={
                        "strategy": "miles-session-agent-loop",
                        "session": (state.get("metadata") or {}).get("tree", {}),
                        "sandbox_setup_seconds": sandbox_setup_seconds,
                        "model_time_seconds": session_model_seconds(state),
                        "tool_time_seconds": tool_seconds,
                    },
                )
                trajectories.append(
                    context.evaluate_trajectory(
                        request, sandbox, trajectory, prepared
                    )
                )
                context.update_progress(
                    "sample_completed",
                    model_calls=model_calls,
                    tool_calls=tool_calls,
                    completed_samples=len(trajectories),
                    active_sample_slot_id=slot.sample_slot_id,
                )
            finally:
                context.update_progress(
                    "cleaning_up",
                    model_calls=model_calls,
                    tool_calls=tool_calls,
                    completed_samples=len(trajectories),
                    active_sample_slot_id=slot.sample_slot_id,
                )
                active_error = sys.exc_info()[1]
                cleanup_errors: list[Exception] = []
                if session_id is not None:
                    try:
                        session_client.delete(session_id)
                    except Exception as exc:
                        cleanup_errors.append(exc)
                try:
                    context.environment_provider.destroy(sandbox)
                except Exception as exc:
                    cleanup_errors.append(exc)
                if active_error is None and cleanup_errors:
                    raise RuntimeError(
                        f"agent rollout cleanup failed: {cleanup_errors[0]}"
                    ) from cleanup_errors[0]

        return RolloutGroupResult(
            rollout_job_id=request.rollout_job_id,
            prompt_group_id=request.prompt_group_id,
            status="completed",
            max_samples=request.max_samples,
            trajectories=trajectories,
            consumed_budget={"model_calls": model_calls, "tool_calls": tool_calls},
        )


def _trajectory_from_session(
    request: RolloutGroupRequest, sample_slot_id: str, state: dict[str, Any], status: str
) -> Trajectory:
    return _trajectory_from_session_common(
        request,
        sample_slot_id,
        state,
        status,
        branch_id=f"{request.rollout_job_id}:{sample_slot_id}",
        metadata={
            "strategy": "miles-session-agent-loop",
            "session": (state.get("metadata") or {}).get("tree", {}),
        },
    )
