"""Append-only JSONL journal (v2 envelope).

Same envelope discipline as swebench/agent/trace.py (v1) -- monotonic ``seq``,
RFC3339 UTC ``ts``, flush per line -- but standalone so harness/* has no import
edge into benchmark code, and thread-safe because CLI slots emit from reader
threads.

The journal is the canonical state: replay/export/resume all read this file.
Writers never rewrite history; corrections are new events.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Union

from harness.core.events import JOURNAL_SCHEMA_VERSION


def new_run_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class JournalWriter:
    """Single-writer, thread-safe JSONL emitter."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        run_id: Optional[str] = None,
        agent_id: str = "agent",
        sandbox_id: str = "default",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_run_id()
        self.agent_id = agent_id
        self.sandbox_id = sandbox_id
        self._seq = 0
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event_type: str, **payload) -> dict:
        """Append one event; returns the full record (including seq)."""
        with self._lock:
            self._seq += 1
            record = {
                "v": JOURNAL_SCHEMA_VERSION,
                "type": event_type,
                "ts": _now(),
                "seq": self._seq,
                "run_id": self.run_id,
                "agent_id": self.agent_id,
                "sandbox_id": self.sandbox_id,
            }
            record.update(payload)
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
            return record

    @property
    def seq(self) -> int:
        return self._seq

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JournalWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def iter_journal(path: Union[str, Path]) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_journal(path: Union[str, Path]) -> List[dict]:
    return list(iter_journal(path))
