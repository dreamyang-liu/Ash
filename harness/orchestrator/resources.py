"""Resource ledger: what a run allocated, so it can be reclaimed after a crash.

The problem this solves is recorded in AGENTS.md ("There is no GC yet") and
reproduced trivially: a handful of fork-demo runs left 17 orphaned snapshots on
the server. ``finally: session.destroy()`` covers a clean exit; SIGKILL, OOM and
a hard Ctrl-C cover nothing, and the backend has no idea which run a sandbox or
snapshot belonged to (the snapshot API returns no owner field).

Design: a **write-ahead ledger**. Record the intent to allocate *before*
allocating, mark it released after freeing. A crash therefore leaves a claim
behind, never a silent leak -- the failure mode is a stale entry (harmless,
``reap`` re-checks the backend) rather than an unreachable resource.

    ledger = ResourceLedger()                      # runs/resources.jsonl
    with ledger.run("run-42") as claim:
        claim.sandbox(sandbox_id)
        claim.snapshot(snapshot_id, keep=True)     # keep => not reaped
    # normal exit marks the run released

    harness reap --dry-run                          # what would go
    harness reap                                    # actually free it

Entries are append-only JSONL for the same reason the journal is: a partially
written line is skipped, and two processes appending concurrently cannot corrupt
each other's records.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Union

DEFAULT_LEDGER = "runs/resources.jsonl"

CLAIM = "claim"        # {run_id, kind, id, pid, keep}
RELEASE = "release"    # {run_id, kind, id}
RUN_DONE = "run_done"  # {run_id}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Resource:
    kind: str            # "sandbox" | "snapshot"
    id: str
    run_id: str
    pid: Optional[int] = None
    keep: bool = False
    released: bool = False
    run_done: bool = False
    created: Optional[str] = None

    def orphaned(self) -> bool:
        """Allocated, never released, and its owning process is gone."""
        if self.released or self.keep:
            return False
        if self.run_done:
            # The run finished but this resource was never released: a bug or a
            # kill between the two writes. Still an orphan.
            return True
        return not _pid_alive(self.pid)


class RunClaim:
    """Records what one run allocates. Obtained from ``ResourceLedger.run``."""

    def __init__(self, ledger: "ResourceLedger", run_id: str) -> None:
        self._ledger = ledger
        self.run_id = run_id
        self.sandboxes: List[str] = []
        self.snapshots: List[str] = []

    def sandbox(self, sandbox_id: str, *, keep: bool = False) -> str:
        self.sandboxes.append(sandbox_id)
        self._ledger._append(CLAIM, run_id=self.run_id, kind="sandbox",
                             id=sandbox_id, pid=os.getpid(), keep=keep)
        return sandbox_id

    def snapshot(self, snapshot_id: str, *, keep: bool = False) -> str:
        """Record a snapshot. ``keep=True`` for ones a fork will need later."""
        self.snapshots.append(snapshot_id)
        self._ledger._append(CLAIM, run_id=self.run_id, kind="snapshot",
                             id=snapshot_id, pid=os.getpid(), keep=keep)
        return snapshot_id

    def released(self, kind: str, resource_id: str) -> None:
        self._ledger._append(RELEASE, run_id=self.run_id, kind=kind, id=resource_id)

    def attach(self, journal) -> "RunClaim":
        """Record snapshots automatically from ``checkpoint.captured`` events."""
        from harness.core.events import CHECKPOINT_CAPTURED

        seen: Set[str] = set()

        def observer(record: dict) -> None:
            if record.get("type") != CHECKPOINT_CAPTURED:
                return
            snapshot_id = record.get("snapshot_id")
            if snapshot_id and snapshot_id not in seen:
                seen.add(snapshot_id)
                # keep=False: a checkpoint is reclaimable unless a fork pins it.
                self.snapshot(snapshot_id)

        journal.subscribe(observer)
        return self


class ResourceLedger:
    """Append-only record of allocations, and the queries ``reap`` needs."""

    def __init__(self, path: Union[str, Path] = DEFAULT_LEDGER) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # --- writing -----------------------------------------------------------
    def _append(self, event: str, **payload) -> None:
        record = {"event": event, "ts": _now()}
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    def run(self, run_id: str) -> "_RunContext":
        return _RunContext(self, run_id)

    def mark_run_done(self, run_id: str) -> None:
        self._append(RUN_DONE, run_id=run_id)

    # --- reading -----------------------------------------------------------
    def entries(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # torn write: skip, never abort the reap

    def resources(self) -> List[Resource]:
        """Replay the ledger into current resource state."""
        table: Dict[tuple, Resource] = {}
        done_runs: Set[str] = set()

        for entry in self.entries():
            event = entry.get("event")
            if event == RUN_DONE:
                done_runs.add(entry.get("run_id"))
                continue
            key = (entry.get("kind"), entry.get("id"))
            if not key[1]:
                continue
            if event == CLAIM:
                table[key] = Resource(
                    kind=key[0],
                    id=key[1],
                    run_id=entry.get("run_id") or "",
                    pid=entry.get("pid"),
                    keep=bool(entry.get("keep")),
                    created=entry.get("ts"),
                )
            elif event == RELEASE and key in table:
                table[key].released = True

        for resource in table.values():
            resource.run_done = resource.run_id in done_runs
        return list(table.values())

    def orphans(self) -> List[Resource]:
        return [r for r in self.resources() if r.orphaned()]

    def compact(self) -> int:
        """Rewrite the ledger keeping only live entries. Returns lines dropped."""
        live = [r for r in self.resources() if not r.released]
        before = sum(1 for _ in self.entries())
        lines = [
            json.dumps(
                {
                    "event": CLAIM,
                    "ts": r.created or _now(),
                    "run_id": r.run_id,
                    "kind": r.kind,
                    "id": r.id,
                    "pid": r.pid,
                    "keep": r.keep,
                },
                ensure_ascii=False,
            )
            for r in live
        ]
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            tmp.replace(self.path)
        return before - len(lines)


class _RunContext:
    """Context manager marking the run done on exit (even on exception)."""

    def __init__(self, ledger: ResourceLedger, run_id: str) -> None:
        self._ledger = ledger
        self.claim = RunClaim(ledger, run_id)

    def __enter__(self) -> RunClaim:
        return self.claim

    def __exit__(self, *exc) -> None:
        self._ledger.mark_run_done(self.claim.run_id)


def _pid_alive(pid: Optional[int]) -> bool:
    """True if the process still exists. Unknown pid counts as alive.

    Treating an unknown pid as alive is the safe direction: reaping something a
    live run still needs is worse than leaving a stale entry for the next pass.
    """
    if not pid:
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, owned by someone else
    except (TypeError, ValueError):
        return True
