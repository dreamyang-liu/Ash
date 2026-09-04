from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import canonical_prefix, trajectory_prefix_chain_hash, trajectory_prefix_chain_hashes


@dataclass(frozen=True)
class PrefixCursor:
    """One node in the exact trajectory-prefix DAG."""

    chain_hash: str
    depth: int


@dataclass(frozen=True)
class PrefixTarget:
    """A reusable artifact (normally an environment snapshot) attached to a prefix."""

    task_id: str
    env_fingerprint: str
    cursor: PrefixCursor
    reference: str
    step_id: int = 0
    created_at: str = ""
    metadata: dict[str, Any] | None = None


class ExactPrefixIndex:
    """Compact exact-prefix DAG plus longest-prefix lookup.

    Prefix nodes are content-addressed by an incremental hash chain. Shared
    trajectory prefixes therefore occupy one set of nodes even when many
    checkpoints/branches reference them. For dense rollouts callers can keep a
    :class:`PrefixCursor` and append one new step in O(1) per step.
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
            CREATE TABLE IF NOT EXISTS prefix_nodes (
                chain_hash TEXT PRIMARY KEY,
                parent_hash TEXT NOT NULL,
                depth INTEGER NOT NULL,
                item_json TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS prefix_targets (
                task_id TEXT NOT NULL,
                env_fingerprint TEXT NOT NULL DEFAULT '',
                chain_hash TEXT NOT NULL,
                depth INTEGER NOT NULL,
                reference TEXT NOT NULL,
                step_id INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (task_id, env_fingerprint, chain_hash)
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_prefix_targets_lookup "
            "ON prefix_targets(task_id, env_fingerprint, depth DESC)"
        )
        self._db.commit()

    @staticmethod
    def root() -> PrefixCursor:
        return PrefixCursor(chain_hash=trajectory_prefix_chain_hash([]), depth=0)

    @staticmethod
    def _extend_hash(parent_hash: str, item: Any) -> tuple[str, str]:
        if len(parent_hash) != 64:
            raise ValueError("parent chain hash must be a SHA-256 hex digest")
        try:
            parent = bytes.fromhex(parent_hash)
        except ValueError as exc:
            raise ValueError("parent chain hash must be hex") from exc
        item_json = canonical_prefix(item)
        child = hashlib.sha256(parent + b"\x00" + item_json.encode("utf-8")).hexdigest()
        return child, item_json

    def append(self, parent: PrefixCursor, item: Any) -> PrefixCursor:
        """Append one exact trajectory item and return the new shared cursor."""
        if parent.depth < 0:
            raise ValueError("parent depth must be >= 0")
        root = self.root()
        if parent.depth == 0:
            if parent.chain_hash != root.chain_hash:
                raise ValueError("depth-0 parent must be the prefix root")
        else:
            row = self._db.execute(
                "SELECT depth FROM prefix_nodes WHERE chain_hash=?", (parent.chain_hash,)
            ).fetchone()
            if row is None or int(row["depth"]) != parent.depth:
                raise KeyError("parent prefix cursor is not present in this index")

        chain_hash, item_json = self._extend_hash(parent.chain_hash, item)
        depth = parent.depth + 1
        self._db.execute(
            """
            INSERT OR IGNORE INTO prefix_nodes (chain_hash, parent_hash, depth, item_json)
            VALUES (?, ?, ?, ?)
            """,
            (chain_hash, parent.chain_hash, depth, item_json),
        )
        row = self._db.execute(
            "SELECT parent_hash, depth, item_json FROM prefix_nodes WHERE chain_hash=?",
            (chain_hash,),
        ).fetchone()
        if (
            row is None
            or row["parent_hash"] != parent.chain_hash
            or int(row["depth"]) != depth
            or row["item_json"] != item_json
        ):
            raise RuntimeError("prefix hash collision or corrupt prefix node")
        self._db.commit()
        return PrefixCursor(chain_hash=chain_hash, depth=depth)

    def cursor_for_prefix(self, trajectory_prefix: Any) -> PrefixCursor:
        """Insert only missing DAG nodes for a complete prefix and return its cursor."""
        value = json.loads(canonical_prefix(trajectory_prefix))
        items = value if isinstance(value, list) else [value]
        cursor = self.root()
        hashes = trajectory_prefix_chain_hashes(value)
        for depth, (item, expected_hash) in enumerate(zip(items, hashes), start=1):
            row = self._db.execute(
                "SELECT parent_hash, depth FROM prefix_nodes WHERE chain_hash=?",
                (expected_hash,),
            ).fetchone()
            if row is not None:
                if row["parent_hash"] != cursor.chain_hash or int(row["depth"]) != depth:
                    raise RuntimeError("corrupt shared prefix DAG")
                cursor = PrefixCursor(expected_hash, depth)
                continue
            cursor = self.append(cursor, item)
            if cursor.chain_hash != expected_hash:
                raise RuntimeError("incremental prefix hash disagrees with canonical chain")
        return cursor

    def register(
        self,
        *,
        task_id: str,
        cursor: PrefixCursor,
        reference: str,
        env_fingerprint: str = "",
        step_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> PrefixTarget:
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if not reference:
            raise ValueError("reference must be non-empty")
        if step_id < 0:
            raise ValueError("step_id must be >= 0")
        if cursor.depth > 0:
            row = self._db.execute(
                "SELECT depth FROM prefix_nodes WHERE chain_hash=?", (cursor.chain_hash,)
            ).fetchone()
            if row is None or int(row["depth"]) != cursor.depth:
                raise KeyError("target cursor is not present in this index")
        elif cursor != self.root():
            raise ValueError("depth-0 target must use the root cursor")
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = metadata or {}
        self._db.execute(
            """
            INSERT INTO prefix_targets (
                task_id, env_fingerprint, chain_hash, depth, reference, step_id,
                created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, env_fingerprint, chain_hash) DO UPDATE SET
                reference=excluded.reference,
                step_id=excluded.step_id,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json
            """,
            (
                task_id,
                env_fingerprint,
                cursor.chain_hash,
                cursor.depth,
                reference,
                step_id,
                created_at,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._db.commit()
        return PrefixTarget(
            task_id=task_id,
            env_fingerprint=env_fingerprint,
            cursor=cursor,
            reference=reference,
            step_id=step_id,
            created_at=created_at,
            metadata=metadata,
        )

    def put(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        reference: str,
        env_fingerprint: str = "",
        step_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> PrefixTarget:
        cursor = self.cursor_for_prefix(trajectory_prefix)
        return self.register(
            task_id=task_id,
            cursor=cursor,
            reference=reference,
            env_fingerprint=env_fingerprint,
            step_id=step_id,
            metadata=metadata,
        )

    def longest_match(
        self,
        *,
        task_id: str,
        trajectory_prefix: Any,
        env_fingerprint: str = "",
    ) -> PrefixTarget | None:
        value = json.loads(canonical_prefix(trajectory_prefix))
        items = value if isinstance(value, list) else [value]
        hashes = trajectory_prefix_chain_hashes(value)
        hashes_by_depth = {i + 1: h for i, h in enumerate(hashes)}
        hashes_by_depth[0] = self.root().chain_hash
        # Probe candidate query-prefix hashes from longest to shortest.  The
        # prefix_targets primary key is (task_id, env_fingerprint, chain_hash),
        # so a normal near-prefix hit requires only one or a few indexed point
        # lookups instead of materializing every target for the task and then
        # scanning them in Python.
        for depth in range(len(items), -1, -1):
            chain_hash = hashes_by_depth[depth]
            row = self._db.execute(
                """
                SELECT * FROM prefix_targets
                WHERE task_id=? AND env_fingerprint=? AND chain_hash=? AND depth=?
                LIMIT 1
                """,
                (task_id, env_fingerprint, chain_hash, depth),
            ).fetchone()
            if row is not None:
                return self._target(row)
        return None

    def reconstruct(self, cursor: PrefixCursor) -> list[Any]:
        if cursor.depth == 0:
            if cursor != self.root():
                raise ValueError("invalid root cursor")
            return []
        items: list[Any] = []
        chain_hash = cursor.chain_hash
        expected_depth = cursor.depth
        while expected_depth > 0:
            row = self._db.execute(
                "SELECT parent_hash, depth, item_json FROM prefix_nodes WHERE chain_hash=?",
                (chain_hash,),
            ).fetchone()
            if row is None or int(row["depth"]) != expected_depth:
                raise KeyError("prefix cursor cannot be reconstructed")
            items.append(json.loads(row["item_json"]))
            chain_hash = row["parent_hash"]
            expected_depth -= 1
        if chain_hash != self.root().chain_hash:
            raise RuntimeError("prefix DAG parent chain does not terminate at root")
        items.reverse()
        return items

    def stats(self) -> dict[str, int]:
        node_count = int(self._db.execute("SELECT COUNT(*) FROM prefix_nodes").fetchone()[0])
        target_count = int(self._db.execute("SELECT COUNT(*) FROM prefix_targets").fetchone()[0])
        item_bytes = int(
            self._db.execute("SELECT COALESCE(SUM(LENGTH(item_json)), 0) FROM prefix_nodes").fetchone()[0]
        )
        return {"node_count": node_count, "target_count": target_count, "item_json_bytes": item_bytes}

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ExactPrefixIndex":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def _target(row: sqlite3.Row) -> PrefixTarget:
        return PrefixTarget(
            task_id=row["task_id"],
            env_fingerprint=row["env_fingerprint"],
            cursor=PrefixCursor(row["chain_hash"], int(row["depth"])),
            reference=row["reference"],
            step_id=int(row["step_id"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
