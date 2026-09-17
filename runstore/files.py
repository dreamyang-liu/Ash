"""Atomic, fsynced receipts and append-only trajectory ingestion."""

from __future__ import annotations

import json
import os
from pathlib import Path

from runstore.specs import canonical


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def journal_events(path: Path) -> list[dict]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.endswith(b"\n") and line.strip()]


class JournalFrame:
    """Parse only appended complete lines, while checking the observed prefix."""

    def __init__(self) -> None:
        self.data = b""
        self.events: list[dict] = []
        self.persisted = 0

    def read(self, path: Path) -> list[dict]:
        from runstore.store import Conflict

        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = b""
        if not data.startswith(self.data):
            raise Conflict("Observed journal prefix was rewritten or truncated")
        end = data.rfind(b"\n") + 1
        complete = data[:end]
        added = [json.loads(line) for line in complete[len(self.data):].splitlines() if line.strip()]
        self.events.extend(added)
        self.data = complete
        return self.events
