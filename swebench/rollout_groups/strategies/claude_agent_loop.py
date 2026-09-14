"""Independent-sample Claude Agent SDK rollout through Ash MCP tools."""

from __future__ import annotations

import sys
import tempfile
import time

from ..claude_runtime import ClaudeAgentRuntime
from ..protocol import RolloutGroupRequest, RolloutGroupResult
from ..runner import RolloutContext
from ..session_runtime import MilesSessionClient, session_model_seconds, trajectory_from_session


class ClaudeAgentLoopRolloutStrategy:
    """Run one isolated Claude session and sandbox per allocated sample."""

    def __init__(
        self,
        *,
        model: str = "local",
        session_server_endpoint: str | None = None,
        runtime: ClaudeAgentRuntime | None = None,
    ) -> None:
        self.model = model
        self.session_server_endpoint = (
            session_server_endpoint.rstrip("/") if session_server_endpoint else None
        )
        self.runtime = runtime or ClaudeAgentRuntime(model=model)

    def run(
        self, request: RolloutGroupRequest, context: RolloutContext
    ) -> RolloutGroupResult:
        if context.environment_provider is None:
            raise RuntimeError("claude-agent-loop requires an environment provider")
        endpoint = (
            request.session_server_endpoint
            or self.session_server_endpoint
            or request.model_endpoint
        )
        if "/sessions" in endpoint:
            raise ValueError("session_server_endpoint must be a Miles server base URL")
        slots = list(request.sample_slots[: request.max_samples])
        trajectories = []
        consumed_model_calls = 0
        consumed_tool_calls = 0
        with tempfile.TemporaryDirectory(prefix="ash-claude-rollout-") as config_dir:
            for position, slot in enumerate(slots):
                client = MilesSessionClient(
                    endpoint,
                    timeout_seconds=context.remaining_wall_time_seconds or 120.0,
                )
                client.require_capabilities(
                    "session-tree-v2",
                    "context-aware-completion-cap",
                    "anthropic-messages",
                )
                miles_session_id = client.create()
                sandbox = None
                try:
                    context.update_progress(
                        "creating_environment",
                        model_calls=consumed_model_calls,
                        tool_calls=consumed_tool_calls,
                        completed_samples=len(trajectories),
                        active_sample_slot_id=slot.sample_slot_id,
                    )
                    started = time.monotonic()
                    sandbox = context.environment_provider.spawn(request)
                    setup_seconds = time.monotonic() - started
                    prepared = context.prepare_task(request, sandbox)
                    run = self.runtime.run_parent(
                        request=request,
                        context=context,
                        session_client=client,
                        miles_session_id=miles_session_id,
                        sandbox=sandbox,
                        agent_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                        max_model_calls=_slot_budget(
                            request.budgets.max_model_calls, len(slots), position
                        ),
                        max_tool_calls=_slot_budget(
                            request.budgets.max_tool_calls, len(slots), position
                        ),
                        capture_checkpoint=False,
                        config_dir=config_dir,
                    )
                    consumed_model_calls += run.model_calls
                    consumed_tool_calls += run.tool_calls
                    state = client.get(miles_session_id)
                    trajectory = trajectory_from_session(
                        request,
                        slot.sample_slot_id,
                        state,
                        run.status,
                        branch_id=f"{request.rollout_job_id}:{slot.sample_index}",
                        prompt_token_alignment="harness_rendered",
                        metadata={
                            "strategy": "claude-agent-loop",
                            "claude_session_id": run.session_id,
                            "sandbox_setup_seconds": setup_seconds,
                            "model_time_seconds": session_model_seconds(state),
                            "tool_time_seconds": run.tool_seconds,
                            "session_tree": (state.get("metadata") or {}).get(
                                "tree", {}
                            ),
                        },
                    )
                    trajectories.append(
                        context.evaluate_trajectory(
                            request, sandbox, trajectory, prepared
                        )
                    )
                finally:
                    active_error = sys.exc_info()[1]
                    cleanup_errors = []
                    if sandbox is not None:
                        try:
                            context.environment_provider.destroy(sandbox)
                        except Exception as exc:
                            cleanup_errors.append(exc)
                    try:
                        client.delete(miles_session_id)
                    except Exception as exc:
                        cleanup_errors.append(exc)
                    if active_error is None and cleanup_errors:
                        raise RuntimeError(
                            f"Claude rollout cleanup failed: {cleanup_errors[0]}"
                        ) from cleanup_errors[0]

        return RolloutGroupResult(
            rollout_job_id=request.rollout_job_id,
            prompt_group_id=request.prompt_group_id,
            status="completed",
            max_samples=request.max_samples,
            trajectories=trajectories,
            consumed_budget={
                "model_calls": consumed_model_calls,
                "tool_calls": consumed_tool_calls,
            },
        )


def _slot_budget(total: int | None, slots: int, position: int) -> int | None:
    if total is None:
        return None
    base, remainder = divmod(total, slots)
    return base + (position < remainder)
