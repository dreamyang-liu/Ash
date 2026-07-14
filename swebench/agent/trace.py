"""Structured JSONL events for agent-to-runtime tool calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


TRACE_SCHEMA_VERSION = 1


def new_run_id() -> str:
    """Return a globally unique identity for one harness run or rollout."""
    return uuid4().hex


class ToolTraceWriter:
    """Append ordered tool events for one agent run."""

    def __init__(self, path: Path, *, run_id: str, agent_id: str,
                 sandbox_id: str) -> None:
        self.run_id = run_id
        self.agent_id = agent_id
        self.sandbox_id = sandbox_id
        self._seq = 0
        self._file = path.open("w", encoding="utf-8")

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        self._seq += 1
        event = {
            "v": TRACE_SCHEMA_VERSION,
            "type": event_type,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "seq": self._seq,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "sandbox_id": self.sandbox_id,
            **payload,
        }
        self._file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._file.flush()
        return event

    def close(self) -> None:
        self._file.close()
