"""Claude Agent SDK parent/child rollout over Ash and Miles SessionTree."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..claude_runtime import ClaudeAgentRuntime, ClaudeRunResult
from ..protocol import RolloutGroupRequest, RolloutGroupResult
from ..runner import EnvironmentCheckpoint, RolloutContext
from ..session_runtime import (
    MilesSessionClient,
    response_input_length,
    session_model_seconds,
    trajectory_from_session,
)


class ClaudeCheckpointAgentLoopRolloutStrategy:
    """Reference policy: branch after the parent's first completed tool."""

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
            raise RuntimeError("claude-checkpoint-agent-loop requires an environment provider")
        slots = list(request.sample_slots[: request.max_samples])
        if len(slots) < 2:
            raise ValueError(
                "claude-checkpoint-agent-loop requires at least two sample slots"
            )
        endpoint = (
            request.session_server_endpoint
            or self.session_server_endpoint
            or request.model_endpoint
        )
        if "/sessions" in endpoint:
            raise ValueError("session_server_endpoint must be a Miles server base URL")
        client = MilesSessionClient(
            endpoint, timeout_seconds=context.remaining_wall_time_seconds or 120.0
        )
        client.require_capabilities(
            "session-tree-v2",
            "context-aware-completion-cap",
            "anthropic-messages",
        )
        miles_session_id = client.create()
        parent = None
        children: list[Any] = []
        checkpoint: EnvironmentCheckpoint | None = None
        try:
            parent, prepared, setup_seconds = self._spawn_parent(request, context)
            parent_slot = slots[0]

            def own_checkpoint(created: EnvironmentCheckpoint) -> None:
                nonlocal checkpoint
                checkpoint = created

            with tempfile.TemporaryDirectory(prefix="ash-claude-rollout-") as config_dir:
                parent_run = self.runtime.run_parent(
                    request=request,
                    context=context,
                    session_client=client,
                    miles_session_id=miles_session_id,
                    sandbox=parent,
                    agent_id=f"{request.rollout_job_id}:{parent_slot.sample_slot_id}",
                    max_model_calls=_parent_model_budget(request, len(slots)),
                    max_tool_calls=_parent_tool_budget(request, len(slots)),
                    capture_checkpoint=True,
                    config_dir=config_dir,
                    on_checkpoint=own_checkpoint,
                )
                if (
                    checkpoint is None
                    or parent_run.checkpoint_message_uuid is None
                    or parent_run.checkpoint_tool_use_id is None
                ):
                    raise RuntimeError(
                        "parent Claude session did not produce an aligned tool checkpoint"
                    )

                pending_runs = [
                    (
                        parent,
                        parent_slot,
                        parent_run,
                        f"{request.rollout_job_id}:root:{parent_slot.sample_index}",
                        None,
                        None,
                        setup_seconds,
                        prepared,
                    )
                ]
                consumed_model_calls = parent_run.model_calls
                consumed_tool_calls = parent_run.tool_calls

                for slot in slots[1:]:
                    context.check_cancelled()
                    context.update_progress(
                        "restoring_checkpoint",
                        model_calls=consumed_model_calls,
                        tool_calls=consumed_tool_calls,
                        completed_samples=len(pending_runs),
                        active_sample_slot_id=slot.sample_slot_id,
                    )
                    started = time.monotonic()
                    child = context.environment_provider.restore_checkpoint(
                        checkpoint,
                        agent_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                    )
                    child_setup_seconds = time.monotonic() - started
                    children.append(child)
                    child_run = self.runtime.run_child(
                        request=request,
                        context=context,
                        session_client=client,
                        miles_session_id=miles_session_id,
                        sandbox=child,
                        agent_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                        max_model_calls=_child_model_budget(request, len(slots)),
                        max_tool_calls=_child_tool_budget(request, len(slots)),
                        parent_claude_session_id=parent_run.session_id,
                        checkpoint_message_uuid=parent_run.checkpoint_message_uuid,
                        config_dir=config_dir,
                    )
                    consumed_model_calls += child_run.model_calls
                    consumed_tool_calls += child_run.tool_calls
                    child_state = client.get(miles_session_id)
                    pending_runs.append(
                        (
                            child,
                            slot,
                            child_run,
                            f"{request.rollout_job_id}:child:{slot.sample_index}",
                            (
                                f"{request.rollout_job_id}:root:"
                                f"{parent_slot.sample_index}"
                            ),
                            response_input_length(
                                child_state,
                                _first_response_id(child_run),
                            ),
                            child_setup_seconds,
                            prepared,
                        )
                    )

            client.collect_samples(miles_session_id)
            final_state = client.get(miles_session_id)
            leaf_count = len(
                ((final_state.get("metadata") or {}).get("tree") or {}).get(
                    "leaves", []
                )
            )
            if leaf_count < len(pending_runs):
                raise RuntimeError(
                    f"Miles SessionTree returned {leaf_count} leaves for "
                    f"{len(pending_runs)} trajectories"
                )
            leaves = _leaf_states(final_state)
            trajectories = []
            for (
                sandbox,
                slot,
                run,
                branch_id,
                parent_branch_id,
                branch_point_token_count,
                sandbox_setup_seconds,
                state,
            ) in pending_runs:
                response_ids = _model_response_ids(run)
                leaf_state = _matching_leaf(leaves, response_ids)
                trajectory = self._trajectory(
                    request,
                    slot,
                    leaf_state,
                    run,
                    branch_id=branch_id,
                    parent_branch_id=parent_branch_id,
                    branch_point_token_count=branch_point_token_count,
                    setup_seconds=sandbox_setup_seconds,
                    checkpoint=checkpoint,
                )
                trajectories.append(
                    context.evaluate_trajectory(request, sandbox, trajectory, state)
                )
            return RolloutGroupResult(
                rollout_job_id=request.rollout_job_id,
                prompt_group_id=request.prompt_group_id,
                status="completed",
                max_samples=request.max_samples,
                trajectories=trajectories,
                search_branches=len(trajectories) - 1,
                consumed_budget={
                    "model_calls": consumed_model_calls,
                    "tool_calls": consumed_tool_calls,
                    "session_tree_leaves": leaf_count,
                },
            )
        finally:
            context.update_progress("cleaning_up")
            active_error = sys.exc_info()[1]
            cleanup_errors: list[Exception] = []
            for child in children:
                _cleanup(cleanup_errors, context.environment_provider.destroy, child)
            if checkpoint is not None:
                _cleanup(
                    cleanup_errors,
                    context.environment_provider.release_checkpoint,
                    checkpoint,
                )
            if parent is not None:
                _cleanup(cleanup_errors, context.environment_provider.destroy, parent)
            _cleanup(cleanup_errors, client.delete, miles_session_id)
            if active_error is None and cleanup_errors:
                raise RuntimeError(
                    f"Claude checkpoint rollout cleanup failed: {cleanup_errors[0]}"
                ) from cleanup_errors[0]

    @staticmethod
    def _spawn_parent(request, context):
        context.update_progress(
            "creating_environment",
            active_sample_slot_id=request.sample_slots[0].sample_slot_id,
        )
        started = time.monotonic()
        sandbox = context.environment_provider.spawn(request)
        setup_seconds = time.monotonic() - started
        prepared = context.prepare_task(request, sandbox)
        return sandbox, prepared, setup_seconds

    @staticmethod
    def _trajectory(
        request,
        slot,
        state,
        run: ClaudeRunResult,
        *,
        branch_id,
        setup_seconds,
        checkpoint,
        parent_branch_id=None,
        branch_point_token_count=None,
    ):
        return trajectory_from_session(
            request,
            slot.sample_slot_id,
            state,
            run.status,
            branch_id=branch_id,
            prompt_token_alignment="harness_rendered",
            parent_branch_id=parent_branch_id,
            branch_point_token_count=branch_point_token_count,
            metadata={
                "strategy": "claude-checkpoint-agent-loop-v1",
                "claude_session_id": run.session_id,
                "synthetic_message_uuids": run.synthetic_message_uuids,
                "environment_checkpoint_id": checkpoint.checkpoint_id,
                "sandbox_setup_seconds": setup_seconds,
                "model_time_seconds": session_model_seconds(state),
                "tool_time_seconds": run.tool_seconds,
                "session_tree": (state.get("metadata") or {}).get("tree", {}),
            },
        )


def _model_response_ids(run: ClaudeRunResult) -> set[str]:
    return set(run.model_response_ids)


def _first_response_id(run: ClaudeRunResult) -> str:
    if not run.model_response_ids:
        raise ValueError("Claude child run has no model response IDs")
    return run.model_response_ids[0]


def _leaf_states(state: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = state.get("metadata") or {}
    tree = metadata.get("tree") or {}
    records = metadata.get("tree_records") or {}
    if not records:
        return [state]
    nodes = {node["id"]: node for node in tree.get("nodes") or []}
    result = []
    for leaf in tree.get("leaves") or []:
        path = leaf.get("path_node_ids") or []
        if not path or any(str(node_id) not in records for node_id in path):
            continue
        path_records = [records[str(node_id)]["record"] for node_id in path]
        result.append(
            {
                "records": path_records,
                "metadata": {
                    "accumulated_token_ids": records[str(path[-1])]["token_ids"],
                    "tree": tree,
                    "leaf_response_ids": [
                        nodes[node_id]["response_id"] for node_id in path
                    ],
                },
            }
        )
    return result or [state]


def _matching_leaf(
    leaves: list[dict[str, Any]], response_ids: set[str]
) -> dict[str, Any]:
    if len(leaves) == 1:
        return leaves[0]
    matches = [
        leaf
        for leaf in leaves
        if response_ids
        and response_ids.issubset(
            set((leaf.get("metadata") or {}).get("leaf_response_ids") or [])
        )
    ]
    if len(matches) != 1:
        leaf_response_ids = [
            (leaf.get("metadata") or {}).get("leaf_response_ids") or []
            for leaf in leaves
        ]
        raise ValueError(
            "could not associate one Claude session with exactly one Miles leaf: "
            f"Claude response_ids={sorted(response_ids)!r}, "
            f"SessionTree leaf response_ids={leaf_response_ids!r}"
        )
    return matches[0]


def _split_budget(total: int | None, samples: int) -> tuple[int | None, int]:
    if total is None:
        return None, 0
    return divmod(total, samples)


def _parent_model_budget(request, samples):
    base, remainder = _split_budget(request.budgets.max_model_calls, samples)
    return None if base is None else base + remainder


def _child_model_budget(request, samples):
    return _split_budget(request.budgets.max_model_calls, samples)[0]


def _parent_tool_budget(request, samples):
    base, remainder = _split_budget(request.budgets.max_tool_calls, samples)
    return None if base is None else base + remainder


def _child_tool_budget(request, samples):
    return _split_budget(request.budgets.max_tool_calls, samples)[0]


def _cleanup(errors, function, argument) -> None:
    try:
        function(argument)
    except Exception as exc:
        errors.append(exc)
