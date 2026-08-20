"""Waggle — topology-agnostic write arbitration for agents sharing one workspace.

Optimistic concurrency control (OCC) over the sandbox filesystem, enforced at
tool-call granularity:

- **read**  (``text_editor view``) records a per-agent snapshot
  of the file version that was seen.
- **write** (``text_editor str_replace/insert/write``) is arbitrated: a write
  based on a stale snapshot is rejected with a unified diff of what changed,
  and the rejected agent is granted a time-limited *reservation* so it can
  re-read and re-apply without being overtaken (no starvation). Writers that
  hit a foreign reservation wait; on release they re-arbitrate in FIFO order.
- **shell** cannot be arbitrated up front. Instead its *effects* are detected
  by fingerprinting registered files after each call: any drift from the last
  coordinated state is recorded as an ``external`` version (post-hoc
  accounting: detect effects, don't guess intent). Attribution is deliberately
  honest — the record names the agent whose scan *detected* the drift, not a
  claimed culprit (concurrent shells are indistinguishable).

Deliberate trade-off: same-file operations are serialized on purpose, and the
tool I/O of a commit (write + content/fingerprint fetch) happens while holding
the file's condition — the fingerprint must be updated atomically with the
write, or drift detection could not tell coordinated writes from external
ones. Sandbox calls are localhost HTTP (milliseconds), so the serialization
cost is bounded; different files never contend.

Conflict *resolution* is delegated to the calling LLM: a rejection is just a
failed tool result carrying the diff and instructions to re-read.

Design rules:
- Topology-agnostic — no manager/worker/subtask concepts, only flat agent ids.
- The 7-tool schema is never changed; all semantics live in tool results.
- History stores full file content per version (self-evident, diff on demand).
- Mechanism vs policy: the OCC kernel is fixed; decisions are ``WagglePolicy``
  hooks (all-``Defer`` defaults == stock behavior). Two mountings share the
  kernel: ``CoordinatedExecutor`` (in-process wrapper, transitional / test
  fixture) and ``WaggleInterceptor`` (the MCP-proxy pipeline element).
"""

from __future__ import annotations

import difflib
import logging
import re
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

from ..models import ToolResult
from .pipeline import CallContext, Continue, ShortCircuit, ToolInterceptor, Verdict
from .tools import EDIT_COMMANDS

logger = logging.getLogger("ash.waggle")  # unconfigured -> WARNING+ to stderr

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
    op: str                  # baseline | write | create | external | delete
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
                for r in list(rec.history)     # snapshot: appends may race the dump
            ]
            for (sbx, path), rec in files.items() if rec.history
        }


# --------------------------------------------------------------------------- #
#  Policy surface (mechanism vs policy — ARCHITECTURE.md L2)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Allow:
    """Let the write commit now, bypassing further OCC checks."""


@dataclass(frozen=True)
class Reject:
    """Refuse the write; ``message`` (if any) is shown to the agent."""
    message: str = ""


@dataclass(frozen=True)
class Wait:
    """Block until the file's state changes hands, then re-ask the policy
    (bounded by the write call's TTL deadline)."""


@dataclass(frozen=True)
class Defer:
    """Fall through to the default OCC behavior (the stock kernel decision)."""


@dataclass(frozen=True)
class Ignore:
    """``on_drift`` only: skip recording the detected drift."""


PolicyDecision = Union[Allow, Reject, Wait, Defer, Ignore]
_DECISION_TYPES = (Allow, Reject, Wait, Defer, Ignore)


@dataclass(frozen=True)
class PolicyContext:
    """Read-only, kernel-computed view handed to policy hooks.

    Carries values, not the ledger: policies can decide but have no API to
    mutate coordination state (account integrity is unreachable from policy
    code). ``history`` items are frozen ``ChangeRecord``s.
    """
    event: str                       # write | conflict | drift | commit
    agent_id: str
    sandbox_id: str
    path: str
    tool_name: str
    args: dict
    snapshot_version: Optional[int]  # this agent's snapshot (None = never read)
    current_version: int
    authors_since: tuple[str, ...]   # authors of versions after the snapshot
    diff: str                        # unified diff (conflict events only)
    history: tuple[ChangeRecord, ...]


class WagglePolicy:
    """Decision hooks for Waggle. Mechanism is fixed; these choose decisions.

    Hooks run INSIDE the file's critical section (policy authors never reason
    about concurrency), receive a read-only ``PolicyContext``, and any hook
    exception falls back to default OCC and is logged (fail-safe). The default
    implementations reproduce stock OCC behavior exactly: everything defers.
    """

    def on_write(self, ctx: PolicyContext) -> PolicyDecision:
        """Gate a write before arbitration. ``Allow`` = commit without
        snapshot checks; ``Reject`` = refuse; ``Wait`` = block until the file
        changes hands, then re-ask; ``Defer`` = default OCC arbitration."""
        return Defer()

    def on_conflict(self, ctx: PolicyContext) -> PolicyDecision:
        """Choose the response to a stale write (``ctx.diff`` is populated).
        ``Allow`` = last-writer-wins; ``Reject(message)`` = custom rejection
        (the loser still gets a reservation); ``Defer`` = reject with diff +
        reservation grant (default)."""
        return Defer()

    def on_drift(self, ctx: PolicyContext) -> PolicyDecision:
        """React to confirmed out-of-band drift. ``Ignore`` = don't record
        it; ``Defer`` = record an ``external`` version (default)."""
        return Defer()

    def on_commit(self, ctx: PolicyContext) -> None:
        """Observe a committed version. Observe-only; return value ignored."""
        return None


# --------------------------------------------------------------------------- #
#  Executor middleware (incarnation 1: in-process wrapper)
# --------------------------------------------------------------------------- #

class CoordinatedExecutor:
    """Wraps an executor and enforces read-versioned OCC on its tool calls.

    Auxiliary state (content, digests, existence) is fetched through the same
    inner executor, so the middleware needs no transport of its own.
    """

    #: Alias of the shared contract (tools.EDIT_COMMANDS); kept as a class
    #: attribute because callers reach it as CoordinatedExecutor.WRITE_COMMANDS.
    WRITE_COMMANDS = EDIT_COMMANDS

    def __init__(self, inner: Executor, state: WorkspaceCoordinator, agent_id: str,
                 sandbox_id: str = "default", require_read: bool = True,
                 policy: Optional[WagglePolicy] = None) -> None:
        self._inner = inner
        self._state = state
        self._agent = agent_id
        self._sbx = sandbox_id
        self._require_read = require_read
        self._policy = policy            # None = pure OCC (no hook overhead)

    # -- dispatch ----------------------------------------------------------- #

    def __call__(self, tool_name: str, args: dict) -> ToolResult:
        command = args.get("command", "")
        if tool_name == "text_editor" and command == "view":
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
            self._record_read(path)
        return result

    def _record_read(self, path: str) -> None:
        """Register/refresh this agent's snapshot of ``path`` after any
        successful read (shared by both mountings)."""
        rec = self._state.file(self._sbx, path)
        with rec.cond:
            if rec.version == 0:
                self._register_baseline(rec, path)
            self._state.set_snapshot(self._agent, self._sbx, path, rec.version)
            rec.rejects.pop(self._agent, None)

    def _register_baseline(self, rec: _FileRecord, path: str) -> None:
        """First sighting of an existing file: store v1. Holds ``rec.cond``."""
        content, digest = self._fetch_state(path)
        WorkspaceCoordinator.record_change(rec, author="baseline", op="baseline",
                                 content=content, digest=digest)

    # -- write path ----------------------------------------------------------- #

    def _write(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        rec = self._state.file(self._sbx, path)
        with rec.cond:
            # Deadline starts once the lock is held — time spent waiting to
            # acquire it must not burn the budget (spurious contended errors).
            deadline = time.monotonic() + self._state.ttl
            gate = self._policy_gate(rec, path, args, deadline)
            if isinstance(gate, ToolResult):
                return gate                          # deadline passed while waiting
            if isinstance(gate, Reject):
                return self._reject_policy(path, gate)
            if isinstance(gate, Allow):
                return self._commit_forced(rec, path, args)
            return self._arbitrate(rec, path, args)  # Defer -> default OCC

    def _policy_gate(self, rec: _FileRecord, path: str, args: dict,
                     deadline: float) -> "PolicyDecision | ToolResult":
        """Wait out foreign reservations, then ask ``on_write``. ``Wait``
        decisions block on the file's condition and re-arbitrate on wake-up.
        Returns Allow | Reject | Defer, or a contended-rejection ToolResult if
        the deadline passes first. Holds ``rec.cond``."""
        while True:
            if not self._await_reservation(rec, deadline):
                return self._reject_contended(rec, path)
            if self._policy is None:
                return Defer()
            decision = self._safe_hook(
                "on_write", self._policy_ctx("write", rec, path, args))
            if not isinstance(decision, Wait):
                return decision if isinstance(decision, (Allow, Reject)) else Defer()
            now = time.monotonic()
            if now >= deadline:
                return self._reject(
                    f"[WAGGLE] Write rejected: policy kept {path} on hold past "
                    f"the deadline.\nRe-read the file and retry shortly.")
            rec.cond.wait(timeout=min(1.0, deadline - now))

    def _arbitrate(self, rec: _FileRecord, path: str, args: dict) -> ToolResult:
        """Default OCC arbitration (snapshot currency). Holds ``rec.cond``."""
        if rec.version == 0:
            return self._write_unregistered(rec, path, args)
        snapshot = self._state.snapshot(self._agent, self._sbx, path)
        if snapshot is None:
            return self._reject_unread(path)
        if snapshot != rec.version:
            return self._conflict(rec, path, args, snapshot)
        return self._commit(rec, path, args, op="write")

    def _conflict(self, rec: _FileRecord, path: str, args: dict,
                  snapshot: int) -> ToolResult:
        """Stale snapshot: ``on_conflict`` may override; the default (and any
        policy fallback) is reject + diff + reservation grant. Holds ``rec.cond``."""
        if self._policy is not None:
            ctx = self._policy_ctx("conflict", rec, path, args, snapshot=snapshot)
            decision = self._safe_hook("on_conflict", ctx)
            if isinstance(decision, Allow):
                return self._commit(rec, path, args, op="write")
            if isinstance(decision, Reject) and decision.message:
                self._grant_reservation(rec)
                return self._reject(
                    f"[WAGGLE] Write rejected by policy: {decision.message}")
        self._grant_reservation(rec)
        return self._reject_stale(rec, path, snapshot)

    def _commit_forced(self, rec: _FileRecord, path: str, args: dict) -> ToolResult:
        """Policy ``Allow``: bypass snapshot checks and commit now. Holds ``rec.cond``."""
        op = "create" if rec.version == 0 else "write"
        return self._commit(rec, path, args, op=op)

    def _reject_policy(self, path: str, decision: Reject) -> ToolResult:
        message = decision.message or f"policy denied writing {path}"
        return self._reject(f"[WAGGLE] Write rejected by policy: {message}")

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
            # Floor of 1s: near-expiry micro-timeouts would otherwise busy-spin.
            # A commit still wakes us instantly via notify_all; the floor only
            # coarsens how often we poll for TTL expiry.
            rec.cond.wait(timeout=max(1.0, min(reservation.expires_at, deadline) - now))

    def _write_unregistered(self, rec: _FileRecord, path: str, args: dict) -> ToolResult:
        """Write to a never-seen path: allow creation, refuse blind overwrite."""
        if self._require_read and self._path_exists(path):
            return self._reject_unread(path)
        result = self._inner("text_editor", args)
        if result.success:
            self._commit_bookkeeping(rec, path, op="create", args=args)
        return result

    def _commit(self, rec: _FileRecord, path: str, args: dict, op: str) -> ToolResult:
        """Snapshot is current; execute the write while holding ``rec.cond``
        (same-file writes are serialized, so no check-then-write race)."""
        result = self._inner("text_editor", args)
        if result.success:
            self._commit_bookkeeping(rec, path, op, args=args)
        return result

    def _commit_bookkeeping(self, rec: _FileRecord, path: str, op: str,
                            args: Optional[dict] = None) -> None:
        content, digest = self._fetch_state(path)
        WorkspaceCoordinator.record_change(rec, self._agent, op, content, digest)
        if rec.reservation and rec.reservation.agent == self._agent:
            rec.reservation = None
        rec.rejects.pop(self._agent, None)
        self._state.set_snapshot(self._agent, self._sbx, path, rec.version)
        rec.cond.notify_all()
        self._observe_commit(rec, path, args or {})

    def _observe_commit(self, rec: _FileRecord, path: str, args: dict) -> None:
        """``on_commit`` (observe-only), inside the critical section. A hook
        exception is logged and ignored — it can never undo a commit."""
        if self._policy is None:
            return
        try:
            self._policy.on_commit(self._policy_ctx("commit", rec, path, args))
        except Exception as exc:  # noqa: BLE001 — policy code must not break the kernel
            logger.warning("waggle policy on_commit failed (ignored): %s",
                           exc, exc_info=True)

    # -- policy plumbing --------------------------------------------------------- #

    def _safe_hook(self, hook: str, ctx: PolicyContext) -> PolicyDecision:
        """Run one policy hook. Exceptions and junk returns fall back to
        default OCC (``Defer``) and are logged — fail-safe by construction."""
        try:
            decision = getattr(self._policy, hook)(ctx)
        except Exception as exc:  # noqa: BLE001 — policy code must not break the kernel
            logger.warning("waggle policy %s failed; deferring to OCC: %s",
                           hook, exc, exc_info=True)
            return Defer()
        if isinstance(decision, _DECISION_TYPES):
            return decision
        logger.warning("waggle policy %s returned %r; deferring to OCC",
                       hook, decision)
        return Defer()

    def _policy_ctx(self, event: str, rec: _FileRecord, path: str, args: dict,
                    snapshot: Optional[int] = None) -> PolicyContext:
        """Kernel-computed, read-only context for policy hooks. Holds ``rec.cond``."""
        snap = snapshot if snapshot is not None else \
            self._state.snapshot(self._agent, self._sbx, path)
        diff = self._diff_since(rec, path, snap) \
            if event == "conflict" and snap is not None else ""
        return PolicyContext(
            event=event, agent_id=self._agent, sandbox_id=self._sbx, path=path,
            tool_name="shell" if event == "drift" else "text_editor",
            args=dict(args), snapshot_version=snap, current_version=rec.version,
            authors_since=tuple(rec.authors_since(snap)) if snap is not None else (),
            diff=diff, history=tuple(rec.history),
        )

    def _grant_reservation(self, rec: _FileRecord) -> None:
        rec.reservation = Reservation(self._agent, time.monotonic() + self._state.ttl)

    # -- shell path (effect detection) ----------------------------------------- #

    def _shell(self, args: dict) -> ToolResult:
        result = self._inner("shell", args)
        self._scan_effects()
        return result

    def _scan_effects(self) -> None:
        """Fingerprint registered files; record any drift from the last
        coordinated state as an ``external`` version.

        Two-phase safety: the bulk fingerprint is taken WITHOUT the per-file
        lock, so it may predate a coordinated commit that lands before the
        comparison (phantom-version risk). Any suspected drift is therefore
        re-fingerprinted while holding ``rec.cond`` before being believed.

        Attribution is deliberately honest: the record names the agent whose
        scan detected the drift (concurrent shells are indistinguishable);
        correctness only needs the version bump.
        """
        detector = f"external (detected by {self._agent})"
        with self._state.scan_lock:
            paths = self._state.registered_paths(self._sbx)
            if not paths:
                return
            digests = self._fetch_digests(paths)             # bulk, pre-lock photo
            for path in paths:
                rec = self._state.file(self._sbx, path)
                with rec.cond:
                    if digests.get(path) == rec.digest:
                        continue
                    # Suspected drift — confirm with a fresh fingerprint under
                    # the lock, so a raced coordinated commit is not misread.
                    fresh = self._fetch_digests([path]).get(path)
                    if fresh == rec.digest:
                        continue                             # coordinated write raced the scan
                    if self._drift_ignored(rec, path):
                        continue                             # policy chose not to record it
                    if fresh is None:                        # deleted out of band
                        WorkspaceCoordinator.record_change(rec, detector, "delete", "", "")
                    else:
                        content = self._fetch_content(path)
                        WorkspaceCoordinator.record_change(rec, detector, "external",
                                                           content, fresh)
                    rec.cond.notify_all()

    def _drift_ignored(self, rec: _FileRecord, path: str) -> bool:
        """Ask ``on_drift`` about confirmed drift; ``Ignore`` skips recording
        it (default records an ``external`` version). Holds ``rec.cond``."""
        if self._policy is None:
            return False
        decision = self._safe_hook("on_drift", self._policy_ctx("drift", rec, path, {}))
        return isinstance(decision, Ignore)

    # -- sandbox probes (via the same inner executor) --------------------------- #

    def _fetch_state(self, path: str) -> tuple[str, str]:
        """Return (content, digest) of a file as the sandbox sees it now."""
        return self._fetch_content(path), self._fetch_digests([path]).get(path, "")

    def _fetch_content(self, path: str) -> str:
        result = self._inner("shell", {"command": shlex.join(["cat", "--", path])})
        return result.output if result.success else ""

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
            f"state.\nRead it with text_editor view first, then apply your edit."
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
            "text_editor view, then re-apply YOUR change on top of the latest version.",
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


# --------------------------------------------------------------------------- #
#  Pipeline mounting (incarnation 2: MCP-proxy interceptor)
# --------------------------------------------------------------------------- #

class WaggleInterceptor(ToolInterceptor):
    """Mounts the Waggle kernel on the tool-interceptor pipeline.

    Adapts ``CoordinatedExecutor`` (kept as the proxy-less lite mode / test
    fixture) to before/after hooks:

    - **write** (``before``): arbitration AND the write itself happen inside
      the file's critical section (ADR-5 — the commit's write and fingerprint
      update must be atomic), so the verdict is a ``ShortCircuit`` carrying
      the committed or rejected result; the framework's own inner call is
      skipped for writes.
    - **read** (``after``): a successful read records this agent's snapshot.
    - **shell** (``after``): registered files are scanned for drift.

    The proxy must supply the raw per-sandbox executor in
    ``ctx.metadata["executor"]``; Waggle uses it for the arbitrated write
    itself and for probe traffic (content/digest fetches) that must not
    re-enter the pipeline. Coordination state is shared across sessions via
    one ``WorkspaceCoordinator``; the per-call adapter objects are stateless.

    ``fail_mode`` is closed: a broken coordinator must reject rather than
    silently allow lost updates.
    """

    tools = {"text_editor", "shell"}
    fail_mode = "closed"

    def __init__(self, state: Optional[WorkspaceCoordinator] = None,
                 policy: Optional[WagglePolicy] = None,
                 ttl: float = DEFAULT_TTL, require_read: bool = True,
                 opaque_writers: "set[str] | None" = None) -> None:
        self.state = state or WorkspaceCoordinator(ttl=ttl)
        self.policy = policy
        self.require_read = require_read
        # Tools that touch the workspace without Waggle being able to see how.
        # `shell` always does; name any other here -- a manifest-defined tool
        # that edits files, say, whose dispatch happens below this seat and so
        # reaches it as one opaque call. Each name costs a fingerprint scan per
        # call (~10 ms, ADR-1) and buys drift detection for it; leaving one out
        # means an agent can edit a file version it never read, unwarned.
        self.opaque_writers = {"shell"} | set(opaque_writers or ())
        self.tools = {"text_editor"} | self.opaque_writers

    def before(self, ctx: CallContext) -> Verdict:
        if ctx.tool_name == "text_editor" and \
                ctx.args.get("command", "") in CoordinatedExecutor.WRITE_COMMANDS:
            return ShortCircuit(self._adapter(ctx)._write(dict(ctx.args)))
        return Continue()

    def after(self, ctx: CallContext, result: ToolResult) -> ToolResult:
        if ctx.tool_name in self.opaque_writers:
            self._adapter(ctx)._scan_effects()
        elif result.success and self._is_read(ctx):
            path = ctx.args.get("path")
            if path:
                self._adapter(ctx)._record_read(path)
        return result

    def dump(self) -> dict:
        """JSON-friendly coordination audit (see ``WorkspaceCoordinator.dump``)."""
        return self.state.dump()

    def _adapter(self, ctx: CallContext) -> CoordinatedExecutor:
        """Per-call kernel adapter bound to this call's agent/sandbox/executor."""
        executor = ctx.metadata.get("executor")
        if executor is None:
            raise RuntimeError(
                "WaggleInterceptor needs the raw sandbox executor in "
                "ctx.metadata['executor'] for writes and probe traffic")
        return CoordinatedExecutor(executor, self.state, agent_id=ctx.agent_id,
                                   sandbox_id=ctx.sandbox_id,
                                   require_read=self.require_read, policy=self.policy)

    @staticmethod
    def _is_read(ctx: CallContext) -> bool:
        return ctx.tool_name == "text_editor" and ctx.args.get("command") == "view"
