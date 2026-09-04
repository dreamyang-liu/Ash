from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _jsonable(value: Any) -> Any:
    """Convert common model/message objects into deterministic JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        return _jsonable(dumped)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    raise TypeError(f"trajectory prefix contains unsupported value: {type(value).__name__}")


def canonical_prefix(prefix: Any) -> str:
    """Canonical JSON representation used as the cache identity input."""
    return json.dumps(
        _jsonable(prefix),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def trajectory_prefix_hash(prefix: Any) -> str:
    """Stable SHA-256 of a model/tool trajectory prefix."""
    return hashlib.sha256(canonical_prefix(prefix).encode("utf-8")).hexdigest()


_PREFIX_CHAIN_SEED = b"ash-prefix-chain-v1"


def trajectory_prefix_chain_seed_hash() -> str:
    """Return the deterministic root hash for an empty trajectory prefix."""
    return hashlib.sha256(_PREFIX_CHAIN_SEED).hexdigest()


def extend_trajectory_prefix_chain(parent_chain_hash: str, step: Any) -> str:
    """Append one canonical trajectory item to an incremental prefix hash.

    ``parent_chain_hash`` may be empty to denote the root. This lets a prefix
    DAG append logical steps in O(1) without re-serializing the complete history.
    """
    if parent_chain_hash:
        try:
            state = bytes.fromhex(parent_chain_hash)
        except ValueError as exc:
            raise ValueError("parent_chain_hash must be a hexadecimal SHA-256 digest") from exc
        if len(state) != 32:
            raise ValueError("parent_chain_hash must be a SHA-256 digest")
    else:
        state = hashlib.sha256(_PREFIX_CHAIN_SEED).digest()
    encoded = json.dumps(
        _jsonable(step),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(state + b"\x00" + encoded).hexdigest()


def trajectory_prefix_length(prefix: Any) -> int:
    """Return the number of canonical trajectory items in ``prefix``."""
    value = _jsonable(prefix)
    return len(value) if isinstance(value, list) else 1


def trajectory_prefix_chain_hashes(prefix: Any) -> list[str]:
    """Compute incremental exact-prefix hashes in one linear pass."""
    value = _jsonable(prefix)
    steps = value if isinstance(value, list) else [value]
    parent = ""
    hashes: list[str] = []
    for step in steps:
        parent = extend_trajectory_prefix_chain(parent, step)
        hashes.append(parent)
    return hashes


def trajectory_prefix_chain_hash(prefix: Any) -> str:
    """Return the incremental hash for the complete exact prefix."""
    hashes = trajectory_prefix_chain_hashes(prefix)
    if hashes:
        return hashes[-1]
    return trajectory_prefix_chain_seed_hash()


@dataclass(frozen=True)
class CheckpointRecord:
    task_id: str
    prefix_hash: str
    trajectory_prefix: Any
    snapshot_id: str
    step_id: int
    trajectory_id: str = ""
    backend: str = ""
    env_fingerprint: str = ""
    prefix_length: int = -1
    chain_hash: str = ""
    created_at: str = ""
    metadata: dict[str, Any] | None = None


class CheckpointStore:
    """Persistent ``(task, trajectory_prefix, environment) -> snapshot`` index.

    AgentENV owns the heavy VM snapshot artifacts. This store only owns the
    logical lookup needed by rollout orchestration, so it stays small enough to
    use SQLite locally and can later be replaced by a distributed metadata
    service without changing the snapshot backend.
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                task_id TEXT NOT NULL,
                prefix_hash TEXT NOT NULL,
                prefix_json TEXT NOT NULL DEFAULT '[]',
                env_fingerprint TEXT NOT NULL DEFAULT '',
                snapshot_id TEXT NOT NULL,
                step_id INTEGER NOT NULL,
                trajectory_id TEXT NOT NULL DEFAULT '',
                backend TEXT NOT NULL DEFAULT '',
                prefix_length INTEGER NOT NULL DEFAULT -1,
                chain_hash TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (task_id, prefix_hash, env_fingerprint)
            )
            """
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(checkpoints)")}
        if "prefix_json" not in columns:
            self._db.execute(
                "ALTER TABLE checkpoints ADD COLUMN prefix_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "prefix_length" not in columns:
            self._db.execute(
                "ALTER TABLE checkpoints ADD COLUMN prefix_length INTEGER NOT NULL DEFAULT -1"
            )
        if "chain_hash" not in columns:
            self._db.execute(
                "ALTER TABLE checkpoints ADD COLUMN chain_hash TEXT NOT NULL DEFAULT ''"
            )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_task_step "
            "ON checkpoints(task_id, step_id)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_task_env_chain "
            "ON checkpoints(task_id, env_fingerprint, prefix_length, chain_hash)"
        )
        self._db.commit()

    def put(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        snapshot_id: str,
        step_id: int,
        trajectory_id: str = "",
        backend: str = "",
        env_fingerprint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if not snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        if step_id < 0:
            raise ValueError("step_id must be >= 0")

        prefix_json = canonical_prefix(trajectory_prefix)
        prefix_hash = hashlib.sha256(prefix_json.encode("utf-8")).hexdigest()
        prefix_value = json.loads(prefix_json)
        prefix_length = trajectory_prefix_length(prefix_value)
        chain_hash = trajectory_prefix_chain_hash(prefix_value)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = metadata or {}
        metadata_json = json.dumps(_jsonable(metadata), ensure_ascii=False, sort_keys=True)
        self._db.execute(
            """
            INSERT INTO checkpoints (
                task_id, prefix_hash, prefix_json, env_fingerprint, snapshot_id, step_id,
                trajectory_id, backend, prefix_length, chain_hash, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, prefix_hash, env_fingerprint) DO UPDATE SET
                prefix_json=excluded.prefix_json,
                snapshot_id=excluded.snapshot_id,
                step_id=excluded.step_id,
                trajectory_id=excluded.trajectory_id,
                backend=excluded.backend,
                prefix_length=excluded.prefix_length,
                chain_hash=excluded.chain_hash,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json
            """,
            (
                task_id,
                prefix_hash,
                prefix_json,
                env_fingerprint,
                snapshot_id,
                step_id,
                trajectory_id,
                backend,
                prefix_length,
                chain_hash,
                created_at,
                metadata_json,
            ),
        )
        self._db.commit()
        return CheckpointRecord(
            task_id=task_id,
            prefix_hash=prefix_hash,
            trajectory_prefix=prefix_value,
            snapshot_id=snapshot_id,
            step_id=step_id,
            trajectory_id=trajectory_id,
            backend=backend,
            env_fingerprint=env_fingerprint,
            prefix_length=prefix_length,
            chain_hash=chain_hash,
            created_at=created_at,
            metadata=metadata,
        )

    def get(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        env_fingerprint: str = "",
    ) -> CheckpointRecord | None:
        return self.get_by_hash(
            task_id=task_id,
            prefix_hash=trajectory_prefix_hash(trajectory_prefix),
            env_fingerprint=env_fingerprint,
        )

    def get_by_hash(
        self,
        *,
        task_id: str,
        prefix_hash: str,
        env_fingerprint: str = "",
    ) -> CheckpointRecord | None:
        row = self._db.execute(
            """
            SELECT task_id, prefix_hash, prefix_json, snapshot_id, step_id, trajectory_id,
                   backend, env_fingerprint, prefix_length, chain_hash, created_at, metadata_json
            FROM checkpoints
            WHERE task_id=? AND prefix_hash=? AND env_fingerprint=?
            """,
            (task_id, prefix_hash, env_fingerprint),
        ).fetchone()
        return self._record(row) if row else None

    def longest_prefix_match(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        env_fingerprint: str = "",
    ) -> CheckpointRecord | None:
        """Return the deepest stored exact prefix reusable by a new trajectory.

        New records use an incremental hash chain, so lookup is O(T + K) for a
        T-item query and K stored checkpoints. Legacy rows fall back to direct
        canonical-prefix comparison.
        """
        query_value = json.loads(canonical_prefix(trajectory_prefix))
        query_steps = query_value if isinstance(query_value, list) else [query_value]
        query_hashes = trajectory_prefix_chain_hashes(query_value)
        hashes_by_length = {i + 1: value for i, value in enumerate(query_hashes)}
        if not query_steps:
            hashes_by_length[0] = trajectory_prefix_chain_hash([])

        rows = self._db.execute(
            """
            SELECT prefix_hash, prefix_length, chain_hash
            FROM checkpoints
            WHERE task_id=? AND env_fingerprint=?
              AND prefix_length>=0 AND prefix_length<=? AND chain_hash!=''
            ORDER BY prefix_length DESC
            """,
            (task_id, env_fingerprint, len(query_steps)),
        ).fetchall()
        for row in rows:
            length = int(row["prefix_length"])
            if hashes_by_length.get(length) == row["chain_hash"]:
                return self.get_by_hash(
                    task_id=task_id,
                    prefix_hash=row["prefix_hash"],
                    env_fingerprint=env_fingerprint,
                )

        legacy_rows = self._db.execute(
            """
            SELECT task_id, prefix_hash, prefix_json, snapshot_id, step_id, trajectory_id,
                   backend, env_fingerprint, prefix_length, chain_hash, created_at, metadata_json
            FROM checkpoints
            WHERE task_id=? AND env_fingerprint=? AND (prefix_length<0 OR chain_hash='')
            ORDER BY step_id DESC, created_at DESC
            """,
            (task_id, env_fingerprint),
        ).fetchall()
        best: tuple[int, sqlite3.Row] | None = None
        for row in legacy_rows:
            stored = json.loads(row["prefix_json"] or "[]")
            if isinstance(stored, list):
                length = len(stored)
                matches = length <= len(query_steps) and stored == query_steps[:length]
            else:
                length = 1
                matches = bool(query_steps) and stored == query_steps[0]
            if matches and (best is None or length > best[0]):
                best = (length, row)
        return self._record(best[1]) if best else None

    def list_for_task(self, task_id: str) -> list[CheckpointRecord]:
        rows = self._db.execute(
            """
            SELECT task_id, prefix_hash, prefix_json, snapshot_id, step_id, trajectory_id,
                   backend, env_fingerprint, prefix_length, chain_hash, created_at, metadata_json
            FROM checkpoints
            WHERE task_id=?
            ORDER BY step_id ASC, created_at ASC
            """,
            (task_id,),
        ).fetchall()
        return [self._record(row) for row in rows]

    def delete_record(self, record: CheckpointRecord) -> bool:
        """Delete one logical checkpoint record from the metadata index."""
        cursor = self._db.execute(
            """
            DELETE FROM checkpoints
            WHERE task_id=? AND prefix_hash=? AND env_fingerprint=?
            """,
            (record.task_id, record.prefix_hash, record.env_fingerprint),
        )
        self._db.commit()
        return cursor.rowcount > 0

    def gc_candidates(
        self,
        *,
        task_id: str,
        retain_latest: int = 1,
        protected_snapshot_ids: Iterable[str] = (),
        protected_prefix_hashes: Iterable[str] = (),
    ) -> list[CheckpointRecord]:
        """Plan GC while preserving the newest and active-frontier checkpoints."""
        if retain_latest < 0:
            raise ValueError("retain_latest must be >= 0")
        records = self.list_for_task(task_id)
        keep_hashes = {str(value) for value in protected_prefix_hashes}
        keep_snapshots = {str(value) for value in protected_snapshot_ids}
        if retain_latest:
            for record in records[-retain_latest:]:
                keep_hashes.add(record.prefix_hash)
                keep_snapshots.add(record.snapshot_id)
        return [
            record
            for record in records
            if record.prefix_hash not in keep_hashes
            and record.snapshot_id not in keep_snapshots
        ]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            task_id=row["task_id"],
            prefix_hash=row["prefix_hash"],
            trajectory_prefix=json.loads(row["prefix_json"] or "[]"),
            snapshot_id=row["snapshot_id"],
            step_id=int(row["step_id"]),
            trajectory_id=row["trajectory_id"],
            backend=row["backend"],
            env_fingerprint=row["env_fingerprint"],
            prefix_length=int(row["prefix_length"]),
            chain_hash=row["chain_hash"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


class BranchManager:
    """Small orchestration layer over a snapshot-capable :class:`Pool`.

    It deliberately does not know about a benchmark or an LLM. A harness hands
    it the task id and the exact model/tool prefix that produced the environment
    state. The manager persists the VM state, indexes it, and can later restore
    one or many independent sandboxes from that checkpoint.
    """

    def __init__(self, pool, store: CheckpointStore):
        self.pool = pool
        self.store = store

    async def checkpoint(
        self,
        sandbox,
        *,
        task_id: str,
        trajectory_prefix: Any,
        step_id: int,
        trajectory_id: str = "",
        env_fingerprint: str = "",
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointRecord:
        if not self.pool.supports_snapshot():
            raise NotImplementedError(
                f"{type(self.pool).__name__} does not support durable snapshots"
            )
        snapshot_id = await self.pool.snapshot(sandbox, name=name)
        return self.store.put(
            task_id=task_id,
            trajectory_prefix=trajectory_prefix,
            snapshot_id=snapshot_id,
            step_id=step_id,
            trajectory_id=trajectory_id,
            backend=type(self.pool).__name__,
            env_fingerprint=env_fingerprint,
            metadata=metadata,
        )

    async def restore(
        self,
        record: CheckpointRecord,
        *,
        count: int = 1,
        agent_ids: Iterable[str] | None = None,
    ) -> list:
        if count < 1:
            raise ValueError("count must be >= 1")
        if not self.pool.supports_restore():
            raise NotImplementedError(
                f"{type(self.pool).__name__} does not support snapshot restore"
            )
        ids = list(agent_ids or [])
        sandboxes = []
        for i in range(count):
            agent_id = ids[i] if i < len(ids) else ""
            sandboxes.append(await self.pool.restore(record.snapshot_id, agent_id=agent_id))
        return sandboxes

    async def gc_checkpoints(
        self,
        *,
        task_id: str,
        retain_latest: int = 1,
        protected_snapshot_ids: Iterable[str] = (),
        protected_prefix_hashes: Iterable[str] = (),
        dry_run: bool = False,
    ) -> list[CheckpointRecord]:
        """Release inactive checkpoints while preserving active-frontier provenance.

        Physical snapshot deletion happens before the SQLite row is removed. If
        backend deletion fails, metadata remains intact instead of orphaning the
        durable snapshot silently.
        """
        candidates = self.store.gc_candidates(
            task_id=task_id,
            retain_latest=retain_latest,
            protected_snapshot_ids=protected_snapshot_ids,
            protected_prefix_hashes=protected_prefix_hashes,
        )
        if dry_run or not candidates:
            return candidates
        supports_delete = getattr(self.pool, "supports_snapshot_delete", None)
        if not callable(supports_delete) or not supports_delete():
            raise NotImplementedError(
                f"{type(self.pool).__name__} does not support snapshot deletion"
            )
        for record in candidates:
            await self.pool.delete_snapshot(record.snapshot_id)
            self.store.delete_record(record)
        return candidates

    async def fork_live(self, sandbox, *, count: int = 1, agent_ids: list[str] | None = None) -> list:
        if not self.pool.supports_fork():
            raise NotImplementedError(
                f"{type(self.pool).__name__} does not support live fork"
            )
        # MicroVMPool accepts agent_ids; the base interface intentionally keeps
        # fork minimal. Fall back to the common signature for other pools.
        try:
            return await self.pool.fork(sandbox, count=count, agent_ids=agent_ids)
        except TypeError:
            return await self.pool.fork(sandbox, count=count)
