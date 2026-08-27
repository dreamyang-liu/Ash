"""Reclaim sandboxes and snapshots that outlived their run.

Two independent sources, because either alone misses cases:

- the **ledger** (harness/resources.py) knows what each run allocated, so it can
  attribute an orphan to a dead pid. It cannot see resources created before the
  ledger existed, or by other tools.
- the **backend** knows everything that exists. It has no owner field, so age is
  the only signal -- hence ``--older-than`` and an explicit opt-in
  (``--include-unknown``) before touching anything the ledger never saw.

Default behaviour is deliberately narrow: reap only ledger-attributed orphans.
Wider sweeps must be asked for, because "delete every snapshot older than N
hours" is exactly the command that eats a colleague's in-progress experiment.

    python -m harness reap --dry-run
    python -m harness reap
    python -m harness reap --include-unknown --older-than 24h   # server sweep
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from harness.orchestrator.resources import Resource, ResourceLedger

_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")


def parse_duration(text: str) -> timedelta:
    """``30m`` / ``24h`` / ``7d`` -> timedelta."""
    match = _DURATION.match(text.strip().lower())
    if not match:
        raise ValueError("bad duration %r (use 30m, 24h, 7d)" % text)
    value, unit = float(match.group(1)), match.group(2)
    return timedelta(**{{"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]: value})


@dataclass
class ReapPlan:
    sandboxes: List[str] = field(default_factory=list)
    snapshots: List[str] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    #: Resources identified as reclaimable that the backend offers no way to
    #: delete. Reported separately: "cannot" is not "failed", and telling a user
    #: their leak is unfixable from here is more useful than 17 error lines.
    unsupported: List[str] = field(default_factory=list)

    def total(self) -> int:
        return len(self.sandboxes) + len(self.snapshots)


class Reaper:
    """Builds and applies a reap plan against an AgentENV-style backend."""

    def __init__(self, client, ledger: Optional[ResourceLedger] = None) -> None:
        self.client = client
        self.ledger = ledger or ResourceLedger()

    # --- planning ----------------------------------------------------------
    def plan(
        self,
        *,
        include_unknown: bool = False,
        older_than: Optional[timedelta] = None,
    ) -> ReapPlan:
        plan = ReapPlan()

        live_sandboxes = {s["id"]: s for s in self.client.list_sandboxes()}
        live_snapshots = {s["id"]: s for s in self.client.list_snapshots()}

        ledger_ids = {(r.kind, r.id) for r in self.ledger.resources()}
        for resource in self.ledger.resources():
            if resource.keep:
                plan.kept.append(resource.id)
                continue
            if not resource.orphaned():
                continue
            pool = live_sandboxes if resource.kind == "sandbox" else live_snapshots
            if resource.id not in pool:
                continue  # already gone; the ledger entry is just stale
            target = plan.sandboxes if resource.kind == "sandbox" else plan.snapshots
            target.append(resource.id)
            plan.reasons[resource.id] = "orphan of run %s (pid %s gone)" % (
                resource.run_id or "?", resource.pid,
            )

        if include_unknown:
            cutoff = _utcnow() - older_than if older_than else None
            for kind, pool, target in (
                ("sandbox", live_sandboxes, plan.sandboxes),
                ("snapshot", live_snapshots, plan.snapshots),
            ):
                for resource_id, meta in pool.items():
                    if (kind, resource_id) in ledger_ids or resource_id in target:
                        continue
                    if meta.get("names"):
                        # A named snapshot is someone's deliberate artifact.
                        plan.kept.append(resource_id)
                        continue
                    created = _parse_ts(meta.get("created_at"))
                    if cutoff is not None and (created is None or created > cutoff):
                        continue
                    target.append(resource_id)
                    plan.reasons[resource_id] = "not in ledger, created %s" % (
                        meta.get("created_at") or "unknown",
                    )

        probe = getattr(self.client, "supports_snapshot_delete", None)
        if plan.snapshots and probe is not None and not probe():
            plan.unsupported = plan.snapshots
            plan.snapshots = []
        return plan

    # --- applying ----------------------------------------------------------
    def apply(self, plan: ReapPlan) -> Dict[str, List[str]]:
        """Delete planned resources. Failures are reported, never raised."""
        done: Dict[str, List[str]] = {"sandboxes": [], "snapshots": [], "failed": []}
        # Sandboxes first: a running sandbox can hold a snapshot chain open.
        for sandbox_id in plan.sandboxes:
            if self.client.delete_sandbox(sandbox_id):
                done["sandboxes"].append(sandbox_id)
                self.ledger._append("release", kind="sandbox", id=sandbox_id, run_id="reap")
            else:
                done["failed"].append(sandbox_id)
        for snapshot_id in plan.snapshots:
            if self.client.delete_snapshot(snapshot_id):
                done["snapshots"].append(snapshot_id)
                self.ledger._append("release", kind="snapshot", id=snapshot_id, run_id="reap")
            else:
                done["failed"].append(snapshot_id)
        return done


class AgentEnvClient:
    """Minimal AgentENV REST client for listing and deleting.

    Kept separate from ``ash_sandbox`` on purpose: reaping must work when no
    pool, session or event loop exists -- including from a cron job after the
    process that leaked the resources is long gone.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        import os

        self.base_url = (base_url or os.environ.get("AENV_SERVER_URL")
                         or "http://127.0.0.1:8000").rstrip("/")
        self.api_key = api_key or os.environ.get("AENV_API_KEY") or ""
        self._snapshot_delete: Optional[bool] = None
        self._unsupported: set = set()

    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _get(self, path: str) -> list:
        import httpx

        try:
            response = httpx.get(self.base_url + path, headers=self._headers(), timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception:  # noqa: BLE001 - an unreachable backend reaps nothing
            return []
        return payload if isinstance(payload, list) else []

    #: 405 on a delete means the route exists but the verb does not, i.e. the
    #: backend has no deletion API for that resource -- distinct from a failure.
    UNSUPPORTED_STATUS = 405

    def _delete(self, path: str) -> bool:
        import httpx

        try:
            response = httpx.delete(self.base_url + path, headers=self._headers(), timeout=60)
        except Exception:  # noqa: BLE001
            return False
        if response.status_code == self.UNSUPPORTED_STATUS:
            self._unsupported.add(path.split("/")[1])
            return False
        return response.status_code < 300 or response.status_code == 404

    def supports_snapshot_delete(self) -> bool:
        """Probe once whether the backend can delete snapshots at all.

        AgentENV (as of this writing) exposes DELETE for /sandboxes and
        /templates only -- ``/snapshots/{id}`` answers 405. Snapshot GC therefore
        needs a backend change; a reaper cannot fix it from outside, and
        pretending otherwise turns one clear message into a wall of failures.
        """
        if self._snapshot_delete is None:
            import httpx

            try:
                response = httpx.request(
                    "OPTIONS", self.base_url + "/snapshots/probe",
                    headers=self._headers(), timeout=15,
                )
                allow = (response.headers.get("allow") or "").upper()
                if allow:
                    self._snapshot_delete = "DELETE" in allow
                else:
                    # No Allow header: fall back to asking for a real delete on a
                    # non-existent id, which is a no-op either way.
                    probe = httpx.delete(
                        self.base_url + "/snapshots/00000000-0000-0000-0000-000000000000",
                        headers=self._headers(), timeout=15,
                    )
                    self._snapshot_delete = probe.status_code != self.UNSUPPORTED_STATUS
            except Exception:  # noqa: BLE001 - assume unsupported if unreachable
                self._snapshot_delete = False
        return bool(self._snapshot_delete)

    def list_sandboxes(self) -> List[dict]:
        return [
            {
                "id": item.get("sandboxID") or item.get("sandbox_id") or "",
                "created_at": item.get("startedAt") or item.get("createdAt"),
                "names": item.get("names") or [],
            }
            for item in self._get("/sandboxes")
        ]

    def list_snapshots(self) -> List[dict]:
        return [
            {
                "id": item.get("snapshotID") or item.get("snapshot_id") or "",
                "created_at": item.get("createdAt"),
                "names": item.get("names") or [],
                "chain_size_mb": item.get("chainSizeMB"),
            }
            for item in self._get("/snapshots")
        ]

    def delete_sandbox(self, sandbox_id: str) -> bool:
        return self._delete("/sandboxes/%s" % sandbox_id)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        return self._delete("/snapshots/%s" % snapshot_id)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(text) -> Optional[datetime]:
    if not text or not isinstance(text, str):
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
