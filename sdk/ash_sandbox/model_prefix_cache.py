from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .checkpoints import trajectory_prefix_hash


@dataclass(frozen=True)
class PrefixCacheCapabilities:
    """Capabilities must distinguish history reuse from true model-state reuse."""

    history_reuse: bool
    kv_reuse: bool
    persistent: bool = False
    cross_process: bool = False
    backend_scope: str = "none"

    @property
    def avoids_prefill_compute(self) -> bool:
        return self.kv_reuse

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelPrefixHandle:
    task_id: str
    prefix_hash: str
    model: str
    backend: str
    reference: str
    capabilities: PrefixCacheCapabilities
    prefix_units: int = 0
    created_at: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def estimated_prefill_units_avoided(self) -> int:
        return self.prefix_units if self.capabilities.kv_reuse else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prefix_hash": self.prefix_hash,
            "model": self.model,
            "backend": self.backend,
            "reference": self.reference,
            "capabilities": self.capabilities.to_dict(),
            "prefix_units": self.prefix_units,
            "estimated_prefill_units_avoided": self.estimated_prefill_units_avoided,
            "created_at": self.created_at,
            "metadata": self.metadata or {},
        }


class ModelPrefixBackend(Protocol):
    @property
    def capabilities(self) -> PrefixCacheCapabilities: ...

    def capture(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        model: str,
        reference: str,
        prefix_units: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ModelPrefixHandle: ...

    def release(self, handle: ModelPrefixHandle) -> None: ...


class HistoryOnlyPrefixBackend:
    """Exact history/session reuse with no provider-exposed KV-state reuse."""

    def __init__(
        self,
        backend: str = "history-only",
        *,
        persistent: bool = True,
        cross_process: bool = True,
        backend_scope: str = "host",
    ):
        self.backend = backend
        self._capabilities = PrefixCacheCapabilities(
            history_reuse=True,
            kv_reuse=False,
            persistent=persistent,
            cross_process=cross_process,
            backend_scope=backend_scope,
        )

    @property
    def capabilities(self) -> PrefixCacheCapabilities:
        return self._capabilities

    def capture(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        model: str,
        reference: str,
        prefix_units: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ModelPrefixHandle:
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        if not reference:
            raise ValueError("reference must be non-empty")
        if prefix_units < 0:
            raise ValueError("prefix_units must be >= 0")
        return ModelPrefixHandle(
            task_id=task_id,
            prefix_hash=trajectory_prefix_hash(trajectory_prefix),
            model=model,
            backend=self.backend,
            reference=reference,
            capabilities=self.capabilities,
            prefix_units=prefix_units,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

    def release(self, handle: ModelPrefixHandle) -> None:
        # This generic adapter does not own the referenced history artifact.
        return None


class ModelPrefixStore:
    """Persistent prefix-state index; heavy cache state remains backend-owned."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS model_prefixes (
                task_id TEXT NOT NULL,
                prefix_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                backend TEXT NOT NULL,
                reference TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                prefix_units INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (task_id, prefix_hash, model, backend)
            )
            """
        )
        self._db.commit()

    def put(self, handle: ModelPrefixHandle) -> ModelPrefixHandle:
        if handle.prefix_units < 0:
            raise ValueError("prefix_units must be >= 0")
        created_at = handle.created_at or datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """
            INSERT INTO model_prefixes (
                task_id, prefix_hash, model, backend, reference, capabilities_json,
                prefix_units, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, prefix_hash, model, backend) DO UPDATE SET
                reference=excluded.reference,
                capabilities_json=excluded.capabilities_json,
                prefix_units=excluded.prefix_units,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json
            """,
            (
                handle.task_id,
                handle.prefix_hash,
                handle.model,
                handle.backend,
                handle.reference,
                json.dumps(handle.capabilities.to_dict(), sort_keys=True),
                handle.prefix_units,
                created_at,
                json.dumps(handle.metadata or {}, sort_keys=True),
            ),
        )
        self._db.commit()
        if handle.created_at:
            return handle
        return ModelPrefixHandle(
            task_id=handle.task_id,
            prefix_hash=handle.prefix_hash,
            model=handle.model,
            backend=handle.backend,
            reference=handle.reference,
            capabilities=handle.capabilities,
            prefix_units=handle.prefix_units,
            created_at=created_at,
            metadata=handle.metadata,
        )

    def get(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        model: str,
        backend: str,
    ) -> ModelPrefixHandle | None:
        return self.get_by_hash(
            task_id=task_id,
            prefix_hash=trajectory_prefix_hash(trajectory_prefix),
            model=model,
            backend=backend,
        )

    def get_by_hash(
        self,
        *,
        task_id: str,
        prefix_hash: str,
        model: str,
        backend: str,
    ) -> ModelPrefixHandle | None:
        row = self._db.execute(
            """
            SELECT * FROM model_prefixes
            WHERE task_id=? AND prefix_hash=? AND model=? AND backend=?
            """,
            (task_id, prefix_hash, model, backend),
        ).fetchone()
        return self._record(row) if row else None

    def list_for_task(self, task_id: str) -> list[ModelPrefixHandle]:
        rows = self._db.execute(
            "SELECT * FROM model_prefixes WHERE task_id=? ORDER BY created_at ASC",
            (task_id,),
        ).fetchall()
        return [self._record(row) for row in rows]

    def delete(self, handle: ModelPrefixHandle) -> bool:
        cursor = self._db.execute(
            """
            DELETE FROM model_prefixes
            WHERE task_id=? AND prefix_hash=? AND model=? AND backend=?
            """,
            (handle.task_id, handle.prefix_hash, handle.model, handle.backend),
        )
        self._db.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ModelPrefixStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> ModelPrefixHandle:
        capabilities = PrefixCacheCapabilities(**json.loads(row["capabilities_json"]))
        return ModelPrefixHandle(
            task_id=row["task_id"],
            prefix_hash=row["prefix_hash"],
            model=row["model"],
            backend=row["backend"],
            reference=row["reference"],
            capabilities=capabilities,
            prefix_units=int(row["prefix_units"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


def prefix_reuse_accounting(handle: ModelPrefixHandle) -> dict[str, Any]:
    """Normalize semantic reuse and compute reuse without conflating them."""
    return {
        "backend": handle.backend,
        "history_reuse": handle.capabilities.history_reuse,
        "kv_reuse": handle.capabilities.kv_reuse,
        "persistent": handle.capabilities.persistent,
        "cross_process": handle.capabilities.cross_process,
        "backend_scope": handle.capabilities.backend_scope,
        "prefix_units": handle.prefix_units,
        "estimated_prefill_units_avoided": handle.estimated_prefill_units_avoided,
        "reference": handle.reference,
        "prefix_hash": handle.prefix_hash,
    }
