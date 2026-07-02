"""Waggle — topology-agnostic write arbitration for agents sharing one workspace.

Optimistic concurrency control (OCC) over the sandbox filesystem, enforced at
tool-call granularity:

- **read**  (``read_file`` / ``text_editor view``) records a per-agent snapshot
  of the file version that was seen.
- **write** (``text_editor str_replace/insert/write``) is arbitrated: a write
  based on a stale snapshot is rejected with a unified diff of what changed,
  and the rejected agent is granted a time-limited *reservation* so it can
  re-read and re-apply without being overtaken (no starvation). Writers that
  hit a foreign reservation wait; on release they re-arbitrate in FIFO order.
- **shell** cannot be arbitrated up front. Instead its *effects* are detected
  by fingerprinting registered files after each call, so out-of-band writes
  still bump versions (post-hoc accounting: detect effects, don't guess intent).

Conflict *resolution* is delegated to the calling LLM: a rejection is just a
failed tool result carrying the diff and instructions to re-read.

Design rules:
- Topology-agnostic — no manager/worker/subtask concepts, only flat agent ids.
- The 7-tool schema is never changed; all semantics live in tool results.
- History stores full file content per version (self-evident, diff on demand).
"""

from __future__ import annotations

import difflib
import re
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..models import ToolResult

DEFAULT_TTL = 120.0          # reservation lifetime (also bounds one write call)
DIFF_LIMIT = 4_000           # max chars of diff shown in a rejection
CONTENT_LIMIT = 1_000_000    # files larger than this are tracked by hash only
ESCALATE_AFTER = 3           # consecutive stale rejections before extra hint

_MD5_LINE = re.compile(r"^([0-9a-f]{32})\s+(.+)$", re.MULTILINE)

Executor = Callable[[str, dict], ToolResult]


# --------------------------------------------------------------------------- #
#  Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChangeRecord:
    """One committed version of a file (full content, self-evident history)."""
    version: int
    author: str
    op: str                  # baseline | write | create | shell | delete
    timestamp: float
    content: str             # "" when deleted or larger than CONTENT_LIMIT


@dataclass
class Reservation:
    """Exclusive write intent granted to a conflict loser, bounded by TTL."""
    agent: str
    expires_at: float

    def active(self, now: float) -> bool:
        return now < self.expires_at


class _FileRecord:
    """All coordination state for one (sandbox, path). Guarded by ``cond``."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.version = 0                       # 0 = not registered yet
        self.history: list[ChangeRecord] = []
        self.reservation: Optional[Reservation] = None
        self.digest = ""                       # sandbox-side md5 of latest version
        self.rejects: dict[str, int] = {}      # agent -> consecutive stale rejects

    def content_at(self, version: int) -> Optional[str]:
        for record in self.history:
            if record.version == version:
                return record.content
        return None

    def authors_since(self, version: int) -> list[str]:
        return [r.author for r in self.history if r.version > version]


class WorkspaceCoordinator:
    """Shared bookkeeper: file records + per-agent snapshots.

    One instance per shared workspace. Thread-safe; lock ordering is always
    ``record.cond`` -> ``self._lock`` (never the reverse).
    """

    def __init__(self, ttl: float = DEFAULT_TTL) -> None:
        self.ttl = ttl
        self._lock = threading.Lock()
        self._files: dict[tuple[str, str], _FileRecord] = {}
        self._snapshots: dict[tuple[str, str, str], int] = {}
        self.scan_lock = threading.Lock()      # serializes shell effect scans

    def file(self, sandbox_id: str, path: str) -> _FileRecord:
        with self._lock:
            key = (sandbox_id, path)
            if key not in self._files:
                self._files[key] = _FileRecord()
            return self._files[key]

    def registered_paths(self, sandbox_id: str) -> list[str]:
        with self._lock:
            return [path for (sbx, path), rec in self._files.items()
                    if sbx == sandbox_id and rec.version > 0]

    def snapshot(self, agent_id: str, sandbox_id: str, path: str) -> Optional[int]:
        with self._lock:
            return self._snapshots.get((agent_id, sandbox_id, path))

    def set_snapshot(self, agent_id: str, sandbox_id: str, path: str, version: int) -> None:
        with self._lock:
            self._snapshots[(agent_id, sandbox_id, path)] = version

    @staticmethod
    def record_change(rec: _FileRecord, author: str, op: str,
                      content: str, digest: str) -> None:
        """Commit a new version. Caller must hold ``rec.cond``."""
        rec.version += 1
        if len(content) > CONTENT_LIMIT:
            content = ""
        rec.history.append(ChangeRecord(rec.version, author, op, time.time(), content))
        rec.digest = digest

    def dump(self) -> dict:
        """JSON-friendly audit of every file's version history."""
        with self._lock:
            files = dict(self._files)
        return {
            f"{sbx}:{path}": [
                {"version": r.version, "author": r.author, "op": r.op,
                 "timestamp": r.timestamp, "bytes": len(r.content)}
                for r in rec.history
            ]
            for (sbx, path), rec in files.items() if rec.history
        }


# --------------------------------------------------------------------------- #
#  Executor middleware (incarnation 1: in-process wrapper)
# --------------------------------------------------------------------------- #

class CoordinatedExecutor:
    """Wraps an executor and enforces read-versioned OCC on its tool calls.

    Auxiliary state (content, digests, existence) is fetched through the same
    inner executor, so the middleware needs no transport of its own.
    """

    WRITE_COMMANDS = frozenset({"str_replace", "insert", "write"})

    def __init__(self, inner: Executor, state: WorkspaceCoordinator, agent_id: str,
                 sandbox_id: str = "default", require_read: bool = True) -> None:
        self._inner = inner
        self._state = state
        self._agent = agent_id
        self._sbx = sandbox_id
        self._require_read = require_read

    # -- dispatch ----------------------------------------------------------- #

    def __call__(self, tool_name: str, args: dict) -> ToolResult:
        command = args.get("command", "")
        if tool_name == "read_file" or (tool_name == "text_editor" and command == "view"):
            return self._read(tool_name, args)
        if tool_name == "text_editor" and command in self.WRITE_COMMANDS:
            return self._write(args)
        if tool_name == "shell":
            return self._shell(args)
        return self._inner(tool_name, args)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close:
            close()

    # -- read path ----------------------------------------------------------- #

    def _read(self, tool_name: str, args: dict) -> ToolResult:
        result = self._inner(tool_name, args)
        path = args.get("path")
        if result.success and path:
            rec = self._state.file(self._sbx, path)
            with rec.cond:
                if rec.version == 0:
                    self._register_baseline(rec, path)
                self._state.set_snapshot(self._agent, self._sbx, path, rec.version)
                rec.rejects.pop(self._agent, None)
        return result

    def _register_baseline(self, rec: _FileRecord, path: str) -> None:
        """First sighting of an existing file: store v1. Holds ``rec.cond``."""
        content, digest = self._fetch_state(path)
        WorkspaceCoordinator.record_change(rec, author="baseline", op="baseline",
                                 content=content, digest=digest)

    # -- write path ----------------------------------------------------------- #

    def _write(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        rec = self._state.file(self._sbx, path)
        deadline = time.monotonic() + self._state.ttl
        with rec.cond:
            if not self._await_reservation(rec, deadline):
                return self._reject_contended(rec, path)
            if rec.version == 0:
                return self._write_unregistered(rec, path, args)
            snapshot = self._state.snapshot(self._agent, self._sbx, path)
            if snapshot is None:
                return self._reject_unread(path)
            if snapshot != rec.version:
                self._grant_reservation(rec)
                return self._reject_stale(rec, path, snapshot)
            return self._commit(rec, path, args, op="write")

    def _await_reservation(self, rec: _FileRecord, deadline: float) -> bool:
        """Wait out a foreign reservation. Woken waiters re-arbitrate; an
        expired reservation is cleared so a still-consistent waiter may win.
        Returns False if the call deadline passes first. Holds ``rec.cond``."""
        while True:
            reservation, now = rec.reservation, time.monotonic()
            if reservation is None or reservation.agent == self._agent:
                return True
            if not reservation.active(now):
                rec.reservation = None
                return True
            if now >= deadline:
                return False
            rec.cond.wait(timeout=min(reservation.expires_at, deadline) - now)

    def _write_unregistered(self, rec: _FileRecord, path: str, args: dict) -> ToolResult:
        """Write to a never-seen path: allow creation, refuse blind overwrite."""
        if self._require_read and self._path_exists(path):
            return self._reject_unread(path)
        result = self._inner("text_editor", args)
        if result.success:
            self._commit_bookkeeping(rec, path, op="create")
        return result

    def _commit(self, rec: _FileRecord, path: str, args: dict, op: str) -> ToolResult:
        """Snapshot is current; execute the write while holding ``rec.cond``
        (same-file writes are serialized, so no check-then-write race)."""
        result = self._inner("text_editor", args)
        if result.success:
            self._commit_bookkeeping(rec, path, op)
        return result

    def _commit_bookkeeping(self, rec: _FileRecord, path: str, op: str) -> None:
        content, digest = self._fetch_state(path)
        WorkspaceCoordinator.record_change(rec, self._agent, op, content, digest)
        if rec.reservation and rec.reservation.agent == self._agent:
            rec.reservation = None
        rec.rejects.pop(self._agent, None)
        self._state.set_snapshot(self._agent, self._sbx, path, rec.version)
        rec.cond.notify_all()

    def _grant_reservation(self, rec: _FileRecord) -> None:
        rec.reservation = Reservation(self._agent, time.monotonic() + self._state.ttl)

    # -- shell path (effect detection) ----------------------------------------- #

    def _shell(self, args: dict) -> ToolResult:
        result = self._inner("shell", args)
        self._scan_effects()
        return result

    def _scan_effects(self) -> None:
        """Fingerprint registered files; version-bump anything a shell changed.

        Attribution is approximate under concurrent shells (scans are
        serialized, the shells themselves are not) — correctness only needs
        the version bump, the author string is informational.
        """
        with self._state.scan_lock:
            paths = self._state.registered_paths(self._sbx)
            if not paths:
                return
            digests = self._fetch_digests(paths)
            for path in paths:
                rec = self._state.file(self._sbx, path)
                with rec.cond:
                    current = digests.get(path)
                    if current == rec.digest:
                        continue
                    if current is None:                      # deleted out of band
                        WorkspaceCoordinator.record_change(
                            rec, f"shell({self._agent})", "delete", "", "")
                    else:
                        content, _ = self._fetch_state(path)
                        WorkspaceCoordinator.record_change(
                            rec, f"shell({self._agent})", "shell", content, current)
                    rec.cond.notify_all()

    # -- sandbox probes (via the same inner executor) --------------------------- #

    def _fetch_state(self, path: str) -> tuple[str, str]:
        """Return (content, digest) of a file as the sandbox sees it now."""
        content_result = self._inner("shell", {"command": shlex.join(["cat", "--", path])})
        content = content_result.output if content_result.success else ""
        digests = self._fetch_digests([path])
        return content, digests.get(path, "")

    def _fetch_digests(self, paths: list[str]) -> dict[str, str]:
        command = shlex.join(["md5sum", "--", *paths])
        result = self._inner("shell", {"command": command})
        if not result.output:
            return {}
        return {path: digest for digest, path in _MD5_LINE.findall(result.output)}

    def _path_exists(self, path: str) -> bool:
        quoted = shlex.quote(path)
        result = self._inner(
            "shell", {"command": f"test -f {quoted} && echo EXISTS || echo MISSING"})
        return "EXISTS" in result.output

    # -- rejection messages ------------------------------------------------------ #

    def _reject(self, message: str) -> ToolResult:
        return ToolResult(success=False, output=message, error="coordination conflict")

    def _reject_unread(self, path: str) -> ToolResult:
        return self._reject(
            f"[WAGGLE] Write rejected: you have not read {path} in its current "
            f"state.\nRead it with read_file first, then apply your edit."
        )

    def _reject_contended(self, rec: _FileRecord, path: str) -> ToolResult:
        holder = rec.reservation.agent if rec.reservation else "another agent"
        return self._reject(
            f"[WAGGLE] Write rejected: {path} is still reserved by \"{holder}\" "
            f"after waiting.\nRe-read the file and retry shortly."
        )

    def _reject_stale(self, rec: _FileRecord, path: str, snapshot: int) -> ToolResult:
        rec.rejects[self._agent] = rec.rejects.get(self._agent, 0) + 1
        authors = ", ".join(rec.authors_since(snapshot)) or "unknown"
        lines = [
            f"[WAGGLE] Write rejected: {path} has changed since you read it.",
            f"  your snapshot : v{snapshot}",
            f"  current       : v{rec.version}  (changed by: {authors})",
            "",
            self._diff_since(rec, path, snapshot),
            "",
            "You now hold a temporary reservation on this file. Re-read it with",
            "read_file, then re-apply YOUR change on top of the latest version.",
            "Do not blindly retry the same edit.",
        ]
        if rec.rejects[self._agent] >= ESCALATE_AFTER:
            lines.append(
                "NOTE: this is rejection "
                f"#{rec.rejects[self._agent]} on this file — you are contending "
                "with another agent. Consider limiting your edit to the part "
                "only you are responsible for."
            )
        return self._reject("\n".join(lines))

    def _diff_since(self, rec: _FileRecord, path: str, snapshot: int) -> str:
        old, new = rec.content_at(snapshot), rec.content_at(rec.version)
        if not old and not new:
            return "(diff unavailable — file too large or content untracked)"
        diff = "\n".join(difflib.unified_diff(
            (old or "").splitlines(), (new or "").splitlines(),
            fromfile=f"{path} (v{snapshot}, as you read it)",
            tofile=f"{path} (v{rec.version}, current)", lineterm="",
        ))
        if len(diff) > DIFF_LIMIT:
            diff = diff[:DIFF_LIMIT] + "\n... (diff truncated)"
        return "--- what changed since your read ---\n" + diff
