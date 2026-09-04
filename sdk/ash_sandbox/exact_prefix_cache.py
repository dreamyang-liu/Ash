from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .checkpoints import CheckpointRecord, CheckpointStore, trajectory_prefix_length
from .model_prefix_cache import ModelPrefixHandle, ModelPrefixStore


@dataclass(frozen=True)
class ExactPrefixMatch:
    """A reusable exact trajectory prefix across environment and model state."""

    checkpoint: CheckpointRecord
    model_prefix: ModelPrefixHandle | None = None

    @property
    def matched_length(self) -> int:
        if self.checkpoint.prefix_length >= 0:
            return self.checkpoint.prefix_length
        return trajectory_prefix_length(self.checkpoint.trajectory_prefix)

    @property
    def snapshot_id(self) -> str:
        return self.checkpoint.snapshot_id

    @property
    def has_model_prefix(self) -> bool:
        return self.model_prefix is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.checkpoint.task_id,
            "prefix_hash": self.checkpoint.prefix_hash,
            "matched_length": self.matched_length,
            "snapshot_id": self.checkpoint.snapshot_id,
            "step_id": self.checkpoint.step_id,
            "env_fingerprint": self.checkpoint.env_fingerprint,
            "model_prefix": self.model_prefix.to_dict() if self.model_prefix else None,
        }


class ExactPrefixCache:
    """Coordinate exact-prefix lookup across environment and model-prefix stores.

    The environment checkpoint store owns longest-prefix matching. Once the
    deepest reusable checkpoint is known, its canonical full-prefix SHA-256 is
    also the key used by ``ModelPrefixStore``. This avoids maintaining a second
    trie/index and guarantees that environment and model/history state refer to
    the same logical trajectory prefix.
    """

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        model_prefix_store: ModelPrefixStore,
    ) -> None:
        self.checkpoint_store = checkpoint_store
        self.model_prefix_store = model_prefix_store

    def lookup(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        env_fingerprint: str = "",
        model: str = "",
        model_backend: str = "",
    ) -> ExactPrefixMatch | None:
        """Return the deepest exact reusable prefix for the current trajectory.

        ``model`` and ``model_backend`` are optional as a pair. If supplied, the
        returned match includes model/history state only when an entry exists for
        the exact same prefix hash. Environment reuse remains valid when model
        state is unavailable; callers must not count that as KV/prefill reuse.
        """
        if bool(model) != bool(model_backend):
            raise ValueError("model and model_backend must be provided together")

        checkpoint = self.checkpoint_store.longest_prefix_match(
            task_id=task_id,
            trajectory_prefix=trajectory_prefix,
            env_fingerprint=env_fingerprint,
        )
        if checkpoint is None:
            return None

        model_prefix = None
        if model and model_backend:
            model_prefix = self.model_prefix_store.get_by_hash(
                task_id=task_id,
                prefix_hash=checkpoint.prefix_hash,
                model=model,
                backend=model_backend,
            )

        return ExactPrefixMatch(
            checkpoint=checkpoint,
            model_prefix=model_prefix,
        )

    def bind_model_prefix(
        self,
        *,
        checkpoint: CheckpointRecord,
        model_prefix: ModelPrefixHandle,
    ) -> ModelPrefixHandle:
        """Attach model/history state to an already materialized checkpoint."""
        if checkpoint.task_id != model_prefix.task_id:
            raise ValueError("checkpoint and model prefix task_id differ")
        if checkpoint.prefix_hash != model_prefix.prefix_hash:
            raise ValueError("checkpoint and model prefix refer to different prefixes")
        return self.model_prefix_store.put(model_prefix)
