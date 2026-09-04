"""Cache-aware environment restore for branching continuations.

The central safety rule is deliberately explicit: a cache hit chooses only the
*environment snapshot*. The model-facing continuation always keeps the target
branch's own messages. A Relaxed hit therefore never substitutes the source
trajectory prefix that originally materialized the reusable environment state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from ash_sandbox import CacheMode, TrajectoryCache, TrajectoryCacheMatch
from ash_sandbox.relaxed_prefix import DEFAULT_WORKSPACE_ROOTS

from .sandbox import AshSession


@dataclass(frozen=True)
class CacheResumeDecision:
    mode: CacheMode
    target_messages: list[dict[str, Any]]
    match: TrajectoryCacheMatch | None

    @property
    def hit(self) -> bool:
        return self.match is not None

    @property
    def snapshot_id(self) -> str | None:
        return self.match.reference if self.match is not None else None

    @property
    def match_kind(self) -> str | None:
        return self.match.kind if self.match is not None else None

    @property
    def relaxed_tier(self) -> str | None:
        return self.match.relaxed_tier if self.match is not None else None

    @property
    def initial_messages(self) -> list[dict[str, Any]]:
        """Detached target-branch model history for ``AshAgent.run``."""
        return deepcopy(self.target_messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "hit": self.hit,
            "snapshot_id": self.snapshot_id,
            "match_kind": self.match_kind,
            "relaxed_tier": self.relaxed_tier,
            "exact_history_match": self.match.exact_history_match if self.match else None,
            "environment_reusable": self.match.environment_reusable if self.match else False,
            "model_prefix_reusable": self.match.model_prefix_reusable if self.match else False,
            "kv_reuse": self.match.kv_reuse if self.match else False,
            "target_message_count": len(self.target_messages),
        }


def resolve_cache_resume(
    cache: TrajectoryCache,
    *,
    mode: CacheMode,
    task_id: str,
    env_fingerprint: str,
    target_messages: Iterable[dict[str, Any]],
    workspace_digest: str | None = None,
    allow_safe_shell: bool = False,
    workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
) -> CacheResumeDecision:
    """Resolve a reusable environment for a target branch, without changing history."""
    messages = deepcopy(list(target_messages))
    match = cache.lookup(
        mode=mode,
        task_id=task_id,
        env_fingerprint=env_fingerprint,
        trajectory_prefix=messages,
        workspace_digest=workspace_digest,
        messages=messages,
        allow_safe_shell=allow_safe_shell,
        workspace_roots=workspace_roots,
    )
    return CacheResumeDecision(mode=mode, target_messages=messages, match=match)


def restore_cache_resume(
    session: AshSession,
    decision: CacheResumeDecision,
    *,
    agent_id: str,
) -> bool:
    """Restore only the matched environment snapshot into ``session``.

    The caller should pass ``decision.initial_messages`` to ``AshAgent.run`` after
    this returns. Keeping these operations separate makes source-history leakage
    difficult to introduce accidentally.
    """
    if not decision.hit or not decision.snapshot_id:
        return False
    return session.restore(decision.snapshot_id, agent_id=agent_id)
