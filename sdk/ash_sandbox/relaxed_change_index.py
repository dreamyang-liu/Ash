from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .relaxed_prefix import (
    DEFAULT_WORKSPACE_ROOTS,
    external_barrier_hash,
    project_environment_prefix,
    workspace_convergence_key,
)


@dataclass(frozen=True)
class RelaxedProjectionTarget:
    """Fast relaxed target for histories differing only by proven safe reads.

    ``state_hash`` preserves the exact ordered mutation/barrier event history while
    dropping only actions proven not to mutate environment state. Unlike workspace
    convergence, this tier requires no filesystem scan.
    """

    task_id: str
    env_fingerprint: str
    state_hash: str
    reference: str
    exact_prefix_hash: str = ""
    step_id: int = 0
    created_at: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def model_prefix_reusable(self) -> bool:
        return False

    @property
    def kv_reuse(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "env_fingerprint": self.env_fingerprint,
            "state_hash": self.state_hash,
            "reference": self.reference,
            "exact_prefix_hash": self.exact_prefix_hash,
            "step_id": self.step_id,
            "created_at": self.created_at,
            "metadata": self.metadata or {},
            "relaxed_tier": "projection",
            "model_prefix_reusable": False,
            "kv_reuse": False,
        }


@dataclass(frozen=True)
class RelaxedChangeTarget:
    """Environment-only reusable state reached by workspace convergence.

    This slower tier allows different structured file-edit histories to converge
    when the final workspace digest and every external barrier are identical. It
    deliberately carries no reusable model/KV state.
    """

    task_id: str
    env_fingerprint: str
    convergence_key: str
    workspace_digest: str
    external_barrier_hash: str
    reference: str
    exact_prefix_hash: str = ""
    step_id: int = 0
    created_at: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def model_prefix_reusable(self) -> bool:
        return False

    @property
    def kv_reuse(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "env_fingerprint": self.env_fingerprint,
            "convergence_key": self.convergence_key,
            "workspace_digest": self.workspace_digest,
            "external_barrier_hash": self.external_barrier_hash,
            "reference": self.reference,
            "exact_prefix_hash": self.exact_prefix_hash,
            "step_id": self.step_id,
            "created_at": self.created_at,
            "metadata": self.metadata or {},
            "relaxed_tier": "convergence",
            "model_prefix_reusable": False,
            "kv_reuse": False,
        }


class RelaxedChangeIndex:
    """Persistent two-tier environment-equivalence index.

    Tier 1 (projection) is the fast path. It drops only proven safe reads and
    preserves the exact mutation/barrier event sequence, so read reordering or
    irrelevant reads can reuse state without scanning the workspace.

    Tier 2 (convergence) is the expensive path. It requires an exact final
    workspace digest plus the external-barrier hash, allowing different trusted
    structured edit histories to converge safely.

    Both tiers are environment-only and never imply model-history, prompt-prefix,
    prefill, or KV-cache reuse.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS relaxed_projection_targets (
                task_id TEXT NOT NULL,
                env_fingerprint TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                reference TEXT NOT NULL,
                exact_prefix_hash TEXT NOT NULL DEFAULT '',
                step_id INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (task_id, env_fingerprint, state_hash)
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_relaxed_projection_lookup "
            "ON relaxed_projection_targets(task_id, env_fingerprint, created_at DESC)"
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS relaxed_change_targets (
                task_id TEXT NOT NULL,
                env_fingerprint TEXT NOT NULL,
                convergence_key TEXT NOT NULL,
                workspace_digest TEXT NOT NULL,
                external_barrier_hash TEXT NOT NULL,
                reference TEXT NOT NULL,
                exact_prefix_hash TEXT NOT NULL DEFAULT '',
                step_id INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (task_id, env_fingerprint, convergence_key)
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_relaxed_change_lookup "
            "ON relaxed_change_targets(task_id, env_fingerprint, created_at DESC)"
        )
        self._db.commit()

    @staticmethod
    def _validate_sha256(value: str, *, field: str, allow_empty: bool = False) -> None:
        if allow_empty and not value:
            return
        if len(value) != 64:
            raise ValueError(f"{field} must be a SHA-256 hex digest")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be hexadecimal") from exc

    @staticmethod
    def _validate_common(
        *,
        task_id: str,
        env_fingerprint: str,
        reference: str,
        exact_prefix_hash: str,
        step_id: int,
    ) -> None:
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if not env_fingerprint:
            raise ValueError("env_fingerprint must be non-empty")
        if not reference:
            raise ValueError("reference must be non-empty")
        if step_id < 0:
            raise ValueError("step_id must be >= 0")
        RelaxedChangeIndex._validate_sha256(
            exact_prefix_hash, field="exact_prefix_hash", allow_empty=True
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
        """Register the fast safe-read-elision state key."""
        self._validate_common(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            reference=reference,
            exact_prefix_hash=exact_prefix_hash,
            step_id=step_id,
        )
        projection = project_environment_prefix(
            messages,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )
        state_hash = projection.state_hash
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            **(metadata or {}),
            "projection_tool_steps": projection.tool_steps,
            "projection_state_steps": projection.state_steps,
            "projection_ignored_read_steps": projection.ignored_read_steps,
        }
        self._db.execute(
            """
            INSERT INTO relaxed_projection_targets (
                task_id, env_fingerprint, state_hash, reference, exact_prefix_hash,
                step_id, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, env_fingerprint, state_hash) DO UPDATE SET
                reference=excluded.reference,
                exact_prefix_hash=excluded.exact_prefix_hash,
                step_id=excluded.step_id,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json
            """,
            (
                task_id,
                env_fingerprint,
                state_hash,
                reference,
                exact_prefix_hash,
                step_id,
                created_at,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._db.commit()
        return RelaxedProjectionTarget(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            state_hash=state_hash,
            reference=reference,
            exact_prefix_hash=exact_prefix_hash,
            step_id=step_id,
            created_at=created_at,
            metadata=metadata,
        )

    def lookup_projection(
        self,
        *,
        task_id: str,
        env_fingerprint: str,
        messages: Iterable[dict[str, Any]],
        allow_safe_shell: bool = False,
        workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    ) -> RelaxedProjectionTarget | None:
        if not task_id or not env_fingerprint:
            return None
        state_hash = project_environment_prefix(
            messages,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        ).state_hash
        row = self._db.execute(
            """
            SELECT * FROM relaxed_projection_targets
            WHERE task_id=? AND env_fingerprint=? AND state_hash=?
            """,
            (task_id, env_fingerprint, state_hash),
        ).fetchone()
        return self._projection_target(row) if row is not None else None

    def register(
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
        """Register the slower final-workspace convergence key."""
        self._validate_common(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            reference=reference,
            exact_prefix_hash=exact_prefix_hash,
            step_id=step_id,
        )
        self._validate_sha256(workspace_digest, field="workspace_digest")

        materialized = [m for m in messages]
        barrier_hash = external_barrier_hash(
            materialized,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )
        convergence_key = workspace_convergence_key(
            env_fingerprint=env_fingerprint,
            workspace_digest=workspace_digest,
            messages=materialized,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = metadata or {}

        self._db.execute(
            """
            INSERT INTO relaxed_change_targets (
                task_id, env_fingerprint, convergence_key, workspace_digest,
                external_barrier_hash, reference, exact_prefix_hash, step_id,
                created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, env_fingerprint, convergence_key) DO UPDATE SET
                reference=excluded.reference,
                exact_prefix_hash=excluded.exact_prefix_hash,
                step_id=excluded.step_id,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json
            """,
            (
                task_id,
                env_fingerprint,
                convergence_key,
                workspace_digest,
                barrier_hash,
                reference,
                exact_prefix_hash,
                step_id,
                created_at,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._db.commit()
        return RelaxedChangeTarget(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            convergence_key=convergence_key,
            workspace_digest=workspace_digest,
            external_barrier_hash=barrier_hash,
            reference=reference,
            exact_prefix_hash=exact_prefix_hash,
            step_id=step_id,
            created_at=created_at,
            metadata=metadata,
        )

    def lookup(
        self,
        *,
        task_id: str,
        env_fingerprint: str,
        workspace_digest: str,
        messages: Iterable[dict[str, Any]],
        allow_safe_shell: bool = False,
        workspace_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS,
    ) -> RelaxedChangeTarget | None:
        if not task_id or not env_fingerprint:
            return None
        self._validate_sha256(workspace_digest, field="workspace_digest")
        materialized = [m for m in messages]
        convergence_key = workspace_convergence_key(
            env_fingerprint=env_fingerprint,
            workspace_digest=workspace_digest,
            messages=materialized,
            allow_safe_shell=allow_safe_shell,
            workspace_roots=workspace_roots,
        )
        row = self._db.execute(
            """
            SELECT * FROM relaxed_change_targets
            WHERE task_id=? AND env_fingerprint=? AND convergence_key=?
            """,
            (task_id, env_fingerprint, convergence_key),
        ).fetchone()
        return self._target(row) if row is not None else None

    def stats(self) -> dict[str, int]:
        target_count = int(self._db.execute("SELECT COUNT(*) FROM relaxed_change_targets").fetchone()[0])
        projection_target_count = int(
            self._db.execute("SELECT COUNT(*) FROM relaxed_projection_targets").fetchone()[0]
        )
        metadata_bytes = int(
            self._db.execute(
                "SELECT COALESCE(SUM(LENGTH(metadata_json)), 0) FROM relaxed_change_targets"
            ).fetchone()[0]
        )
        projection_metadata_bytes = int(
            self._db.execute(
                "SELECT COALESCE(SUM(LENGTH(metadata_json)), 0) FROM relaxed_projection_targets"
            ).fetchone()[0]
        )
        return {
            "target_count": target_count,
            "projection_target_count": projection_target_count,
            "metadata_json_bytes": metadata_bytes,
            "projection_metadata_json_bytes": projection_metadata_bytes,
        }

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "RelaxedChangeIndex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _projection_target(row: sqlite3.Row) -> RelaxedProjectionTarget:
        return RelaxedProjectionTarget(
            task_id=row["task_id"],
            env_fingerprint=row["env_fingerprint"],
            state_hash=row["state_hash"],
            reference=row["reference"],
            exact_prefix_hash=row["exact_prefix_hash"],
            step_id=int(row["step_id"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    @staticmethod
    def _target(row: sqlite3.Row) -> RelaxedChangeTarget:
        return RelaxedChangeTarget(
            task_id=row["task_id"],
            env_fingerprint=row["env_fingerprint"],
            convergence_key=row["convergence_key"],
            workspace_digest=row["workspace_digest"],
            external_barrier_hash=row["external_barrier_hash"],
            reference=row["reference"],
            exact_prefix_hash=row["exact_prefix_hash"],
            step_id=int(row["step_id"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
