"""Trajectory/environment checkpoint glue for branching rollouts.

The sandbox backend and the LLM conversation are intentionally separate state
machines. A valid branch point must capture both at the same turn boundary:

    exact model-facing messages + durable environment snapshot

``SessionCheckpointer`` is callable and plugs directly into
``AshAgent(on_turn_end=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

from ash_sandbox import (
    CheckpointRecord,
    CacheMode,
    CheckpointStore,
    TrajectoryCache,
)
from ash_sandbox.relaxed_prefix import DEFAULT_WORKSPACE_ROOTS

from .checkpoint_policy import OverheadBudgetCheckpointPolicy
from .sandbox import AshSession


@dataclass
class SessionCheckpointer:
    session: AshSession
    store: CheckpointStore
    task_id: str
    trajectory_id: str = ""
    env_fingerprint: str = ""
    every_n_turns: int = 1
    adaptive_policy: OverheadBudgetCheckpointPolicy | None = None
    step_offset: int = 0
    name_prefix: str = "checkpoint"
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics_provider: Callable[[], dict[str, Any]] | None = None
    trajectory_cache: TrajectoryCache | None = None
    trajectory_cache_mode: CacheMode = "none"
    workspace_digest_provider: Callable[[], str] | None = None
    relaxed_allow_safe_shell: bool = False
    relaxed_workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS
    records: list[CheckpointRecord] = field(default_factory=list, init=False)
    cache_registrations: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.every_n_turns < 1:
            raise ValueError("every_n_turns must be >= 1")
        if self.step_offset < 0:
            raise ValueError("step_offset must be >= 0")
        if self.trajectory_cache_mode not in TrajectoryCache.MODES:
            raise ValueError(f"unknown trajectory_cache_mode: {self.trajectory_cache_mode!r}")
        if self.trajectory_cache_mode == "none":
            if self.trajectory_cache is not None or self.workspace_digest_provider is not None:
                raise ValueError("cache objects/providers require trajectory_cache_mode exact or relaxed")
        elif self.trajectory_cache_mode == "exact":
            if self.trajectory_cache is None:
                raise ValueError("trajectory_cache is required for exact mode")
            if self.workspace_digest_provider is not None:
                raise ValueError("exact cache mode must not pay workspace fingerprint overhead")
        else:
            if self.trajectory_cache is None:
                raise ValueError("relaxed cache mode requires trajectory_cache")
        if self.trajectory_cache_mode != "none" and not self.env_fingerprint:
            raise ValueError("env_fingerprint is required when trajectory cache is enabled")

    def __call__(self, step_id: int, messages: list[dict]) -> None:
        """Capture one safe turn boundary under fixed or adaptive cadence."""
        if step_id <= 0:
            return
        if self.adaptive_policy is not None:
            if not self.adaptive_policy.should_checkpoint(step_id):
                return
        elif step_id % self.every_n_turns:
            return
        logical_step = self.step_offset + step_id
        # Freeze rollout/model counters before snapshot I/O so prefix runtime and
        # state-capture overhead remain separable in efficiency accounting.
        runtime_metrics = self.metrics_provider() if self.metrics_provider else {}

        workspace_digest: str | None = None
        workspace_fingerprint_ms = 0.0
        cache_lookup_ms = 0.0
        snapshot_reuse_match = None
        if self.trajectory_cache is not None and self.trajectory_cache_mode != "none":
            # Always try the zero-filesystem-scan exact/projection tiers first. An
            # optional convergence fingerprint is paid only after those miss.
            lookup_started = time.perf_counter()
            snapshot_reuse_match = self.trajectory_cache.lookup_materialized_state(
                mode=self.trajectory_cache_mode,
                task_id=self.task_id,
                env_fingerprint=self.env_fingerprint,
                trajectory_prefix=messages,
                messages=messages,
                allow_safe_shell=self.relaxed_allow_safe_shell,
                workspace_roots=self.relaxed_workspace_roots,
            )
            cache_lookup_ms += (time.perf_counter() - lookup_started) * 1000.0

            if snapshot_reuse_match is None and self.workspace_digest_provider is not None:
                fingerprint_started = time.perf_counter()
                workspace_digest = self.workspace_digest_provider()
                workspace_fingerprint_ms = (time.perf_counter() - fingerprint_started) * 1000.0
                lookup_started = time.perf_counter()
                snapshot_reuse_match = self.trajectory_cache.lookup_materialized_state(
                    mode=self.trajectory_cache_mode,
                    task_id=self.task_id,
                    env_fingerprint=self.env_fingerprint,
                    trajectory_prefix=messages,
                    workspace_digest=workspace_digest,
                    messages=messages,
                    allow_safe_shell=self.relaxed_allow_safe_shell,
                    workspace_roots=self.relaxed_workspace_roots,
                )
                cache_lookup_ms += (time.perf_counter() - lookup_started) * 1000.0

        snapshot_reused = snapshot_reuse_match is not None
        if snapshot_reused:
            snapshot_id = snapshot_reuse_match.reference
            snapshot_ms = 0.0
        else:
            snapshot_started = time.perf_counter()
            snapshot_id = self.session.snapshot(
                name=f"{self.name_prefix}-{self._safe_task_id()}-step-{logical_step}"
            )
            snapshot_ms = (time.perf_counter() - snapshot_started) * 1000.0

        adaptive_policy_state = None
        if self.adaptive_policy is not None:
            if snapshot_reused:
                self.adaptive_policy.observe_reuse(step_id=step_id)
            else:
                rollout_elapsed_ms = runtime_metrics.get("rollout_elapsed_ms")
                if rollout_elapsed_ms is None:
                    rollout_elapsed_ms = runtime_metrics.get("branch_rollout_elapsed_ms")
                self.adaptive_policy.observe_checkpoint(
                    step_id=step_id,
                    rollout_elapsed_ms=(float(rollout_elapsed_ms) if rollout_elapsed_ms is not None else None),
                    snapshot_ms=snapshot_ms,
                )
            adaptive_policy_state = self.adaptive_policy.to_dict()

        cache_registration = None
        exact_target = None
        projection_target = None
        relaxed_target = None
        cache_registration_ms = 0.0
        if self.trajectory_cache is not None and self.trajectory_cache_mode != "none":
            registration_started = time.perf_counter()
            cache_metadata = {
                **self.metadata,
                "trajectory_id": self.trajectory_id,
                "sandbox_id": self.session.sandbox_id,
            }
            if self.trajectory_cache_mode == "exact":
                exact_target = self.trajectory_cache.register_exact(
                    task_id=self.task_id,
                    env_fingerprint=self.env_fingerprint,
                    trajectory_prefix=messages,
                    reference=snapshot_id,
                    step_id=logical_step,
                    metadata=cache_metadata,
                )
                cache_registration = exact_target
            else:
                cache_registration = self.trajectory_cache.register(
                    task_id=self.task_id,
                    env_fingerprint=self.env_fingerprint,
                    trajectory_prefix=messages,
                    workspace_digest=workspace_digest,
                    messages=messages,
                    reference=snapshot_id,
                    step_id=logical_step,
                    metadata=cache_metadata,
                    allow_safe_shell=self.relaxed_allow_safe_shell,
                    workspace_roots=self.relaxed_workspace_roots,
                )
                exact_target = cache_registration.exact
                projection_target = cache_registration.projection
                relaxed_target = cache_registration.relaxed
            cache_registration_ms = (time.perf_counter() - registration_started) * 1000.0
            self.cache_registrations.append(cache_registration)

        prefix_tool_calls = sum(1 for message in messages if message.get("role") == "tool")
        record = self.store.put(
            task_id=self.task_id,
            trajectory_prefix=messages,
            snapshot_id=snapshot_id,
            step_id=logical_step,
            trajectory_id=self.trajectory_id,
            backend=type(self.session._pool).__name__ if self.session._pool else "",
            env_fingerprint=self.env_fingerprint,
            metadata={
                **self.metadata,
                **runtime_metrics,
                "sandbox_id": self.session.sandbox_id,
                "snapshot_ms": round(snapshot_ms, 3),
                "snapshot_reused": snapshot_reused,
                "snapshot_reuse_kind": snapshot_reuse_match.kind if snapshot_reuse_match else None,
                "snapshot_reuse_relaxed_tier": snapshot_reuse_match.relaxed_tier if snapshot_reuse_match else None,
                "snapshot_reuse_source_step": snapshot_reuse_match.step_id if snapshot_reuse_match else None,
                "trajectory_cache_lookup_ms": round(cache_lookup_ms, 3),
                "workspace_digest": workspace_digest or None,
                "workspace_fingerprint_ms": round(workspace_fingerprint_ms, 3),
                "trajectory_cache_registration_ms": round(cache_registration_ms, 3),
                "trajectory_cache_mode": self.trajectory_cache_mode,
                "adaptive_checkpoint_policy": adaptive_policy_state,
                "cache_exact_chain_hash": (
                    exact_target.cursor.chain_hash if exact_target else None
                ),
                "cache_relaxed_projection_state_hash": (
                    projection_target.state_hash if projection_target else None
                ),
                "cache_relaxed_convergence_key": (
                    relaxed_target.convergence_key if relaxed_target else None
                ),
                "prefix_messages": len(messages),
                "prefix_tool_calls": prefix_tool_calls,
            },
        )
        self.records.append(record)

    @property
    def last_record(self) -> CheckpointRecord | None:
        return self.records[-1] if self.records else None

    def _safe_task_id(self) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "-" for c in self.task_id)
