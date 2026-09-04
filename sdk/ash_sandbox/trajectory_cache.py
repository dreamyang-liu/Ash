from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .checkpoints import trajectory_prefix_length
from .prefix_index import ExactPrefixIndex, PrefixTarget
from .relaxed_change_index import (
    RelaxedChangeIndex,
    RelaxedChangeTarget,
    RelaxedProjectionTarget,
)
from .relaxed_prefix import DEFAULT_WORKSPACE_ROOTS

CacheMode = Literal["none", "exact", "relaxed"]
MatchKind = Literal["exact", "relaxed"]


@dataclass(frozen=True)
class TrajectoryCacheRegistration:
    """Exact plus relaxed keys bound to one materialized environment state."""

    exact: PrefixTarget
    projection: RelaxedProjectionTarget
    relaxed: RelaxedChangeTarget | None = None


@dataclass(frozen=True)
class TrajectoryCacheMatch:
    """One reusable environment state returned by a cache-mode lookup.

    ``exact`` means the complete model-facing trajectory prefix is identical.
    ``relaxed`` means only environment state has been proven equivalent. Relaxed
    hits expose whether they came from the zero-filesystem-scan projection tier or
    the slower final-workspace convergence tier.
    """

    kind: MatchKind
    target: PrefixTarget | RelaxedProjectionTarget | RelaxedChangeTarget

    @property
    def reference(self) -> str:
        return self.target.reference

    @property
    def step_id(self) -> int:
        return self.target.step_id

    @property
    def exact_history_match(self) -> bool:
        return self.kind == "exact"

    @property
    def environment_reusable(self) -> bool:
        return True

    @property
    def model_prefix_reusable(self) -> bool:
        # Exact history is necessary but not sufficient for model-prefix reuse:
        # callers still need a matching ModelPrefixHandle/backend entry.
        return False

    @property
    def kv_reuse(self) -> bool:
        return False

    @property
    def relaxed_tier(self) -> str | None:
        if self.kind != "relaxed":
            return None
        if isinstance(self.target, RelaxedProjectionTarget):
            return "projection"
        return "convergence"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "relaxed_tier": self.relaxed_tier,
            "reference": self.reference,
            "step_id": self.step_id,
            "exact_history_match": self.exact_history_match,
            "environment_reusable": True,
            "model_prefix_reusable": False,
            "kv_reuse": False,
        }


class TrajectoryCache:
    """Unified environment-cache interface for No Cache / Exact / Relaxed ablations.

    Relaxed lookup is intentionally ordered by proof cost:

    1. full-depth Exact hit;
    2. projection hit after dropping only proven safe reads (no filesystem scan);
    3. optional final-workspace convergence hit when a workspace digest is supplied;
    4. shallower Exact fallback.

    This makes read-only relaxed matching cheap while keeping mutation-equivalent
    convergence available as a separately measurable, more expensive tier.
    """

    MODES: tuple[CacheMode, ...] = ("none", "exact", "relaxed")

    def __init__(self, exact_index: ExactPrefixIndex, relaxed_index: RelaxedChangeIndex) -> None:
        self.exact_index = exact_index
        self.relaxed_index = relaxed_index

    @staticmethod
    def _validate_mode(mode: str) -> CacheMode:
        if mode not in TrajectoryCache.MODES:
            raise ValueError(f"unknown cache mode: {mode!r}")
        return mode  # type: ignore[return-value]

    def register_exact(
        self,
        *,
        task_id: str,
        env_fingerprint: str,
        trajectory_prefix: Any,
        reference: str,
        step_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> PrefixTarget:
        """Register only the exact-prefix target without workspace fingerprinting."""
        return self.exact_index.put(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            trajectory_prefix=trajectory_prefix,
            reference=reference,
            step_id=step_id,
            metadata=metadata,
        )

    def register_projection(
        self,
        *,
        task_id: str,
        env_fingerprint: str,
        messages: Iterable[dict[str, Any]],
        reference: str,
        exact_prefix_hash: str = "",
        step_id: int = 0,
        metadata: dict[str, Any] | None = None,
        allow_safe_shell: bool = False,
        workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    ) -> RelaxedProjectionTarget:
        return self.relaxed_index.register_projection(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            messages=list(messages),
            reference=reference,
            exact_prefix_hash=exact_prefix_hash,
            step_id=step_id,
            metadata=metadata,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )

    def register_relaxed(
        self,
        *,
        task_id: str,
        env_fingerprint: str,
        workspace_digest: str,
        messages: Iterable[dict[str, Any]],
        reference: str,
        exact_prefix_hash: str = "",
        step_id: int = 0,
        metadata: dict[str, Any] | None = None,
        allow_safe_shell: bool = False,
        workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    ) -> RelaxedChangeTarget:
        """Register the optional final-workspace convergence target."""
        return self.relaxed_index.register(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            workspace_digest=workspace_digest,
            messages=list(messages),
            reference=reference,
            exact_prefix_hash=exact_prefix_hash,
            step_id=step_id,
            metadata=metadata,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )

    def register(
        self,
        *,
        task_id: str,
        env_fingerprint: str,
        trajectory_prefix: Any,
        messages: Iterable[dict[str, Any]],
        reference: str,
        workspace_digest: str | None = None,
        step_id: int = 0,
        metadata: dict[str, Any] | None = None,
        allow_safe_shell: bool = False,
        workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    ) -> TrajectoryCacheRegistration:
        materialized_messages = list(messages)
        exact = self.register_exact(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            trajectory_prefix=trajectory_prefix,
            reference=reference,
            step_id=step_id,
            metadata=metadata,
        )
        projection = self.register_projection(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            messages=materialized_messages,
            reference=reference,
            exact_prefix_hash=exact.cursor.chain_hash,
            step_id=step_id,
            metadata=metadata,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )
        relaxed = None
        if workspace_digest is not None:
            relaxed = self.register_relaxed(
                task_id=task_id,
                env_fingerprint=env_fingerprint,
                workspace_digest=workspace_digest,
                messages=materialized_messages,
                reference=reference,
                exact_prefix_hash=exact.cursor.chain_hash,
                step_id=step_id,
                metadata=metadata,
                allow_safe_shell=allow_safe_shell,
                workspace_roots=workspace_roots,
            )
        return TrajectoryCacheRegistration(exact=exact, projection=projection, relaxed=relaxed)

    def lookup(
        self,
        *,
        mode: CacheMode,
        task_id: str,
        env_fingerprint: str,
        trajectory_prefix: Any,
        workspace_digest: str | None = None,
        messages: Iterable[dict[str, Any]] = (),
        allow_safe_shell: bool = False,
        workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    ) -> TrajectoryCacheMatch | None:
        mode = self._validate_mode(mode)
        if mode == "none":
            return None

        exact = self.exact_index.longest_match(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            trajectory_prefix=trajectory_prefix,
        )
        if mode == "exact":
            return TrajectoryCacheMatch(kind="exact", target=exact) if exact is not None else None

        query_depth = trajectory_prefix_length(trajectory_prefix)
        if exact is not None and exact.cursor.depth == query_depth:
            return TrajectoryCacheMatch(kind="exact", target=exact)

        materialized_messages = list(messages)
        projection = self.relaxed_index.lookup_projection(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            messages=materialized_messages,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )
        if projection is not None:
            return TrajectoryCacheMatch(kind="relaxed", target=projection)

        if workspace_digest is not None:
            convergence = self.relaxed_index.lookup(
                task_id=task_id,
                env_fingerprint=env_fingerprint,
                workspace_digest=workspace_digest,
                messages=materialized_messages,
                allow_safe_shell=allow_safe_shell,
                workspace_roots=workspace_roots,
            )
            if convergence is not None:
                return TrajectoryCacheMatch(kind="relaxed", target=convergence)

        return TrajectoryCacheMatch(kind="exact", target=exact) if exact is not None else None

    def lookup_materialized_state(
        self,
        *,
        mode: CacheMode,
        task_id: str,
        env_fingerprint: str,
        trajectory_prefix: Any,
        workspace_digest: str | None = None,
        messages: Iterable[dict[str, Any]] = (),
        allow_safe_shell: bool = False,
        workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    ) -> TrajectoryCacheMatch | None:
        """Return only a hit that already represents the current environment.

        Unlike :meth:`lookup`, this method never returns a shallower exact prefix
        that would require replaying an unmatched suffix. It is safe for
        pre-snapshot coalescing: a hit can be reused immediately without executing
        more tool actions.
        """
        mode = self._validate_mode(mode)
        if mode == "none":
            return None

        exact = self.exact_index.longest_match(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            trajectory_prefix=trajectory_prefix,
        )
        query_depth = trajectory_prefix_length(trajectory_prefix)
        if exact is not None and exact.cursor.depth == query_depth:
            return TrajectoryCacheMatch(kind="exact", target=exact)
        if mode == "exact":
            return None

        materialized_messages = list(messages)
        projection = self.relaxed_index.lookup_projection(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            messages=materialized_messages,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )
        if projection is not None:
            return TrajectoryCacheMatch(kind="relaxed", target=projection)

        if workspace_digest is not None:
            convergence = self.relaxed_index.lookup(
                task_id=task_id,
                env_fingerprint=env_fingerprint,
                workspace_digest=workspace_digest,
                messages=materialized_messages,
                allow_safe_shell=allow_safe_shell,
                workspace_roots=workspace_roots,
            )
            if convergence is not None:
                return TrajectoryCacheMatch(kind="relaxed", target=convergence)
        return None
