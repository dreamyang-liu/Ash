"""Batch runner: N tasks, bounded concurrency, per-task isolation.

Deliberately thin. It is a worker pool plus a resource ledger, not a persistent
orchestrator: if the batch dies you re-run it, and correctness of *cleanup* comes
from harness/resources.py + ``harness reap`` rather than from a durable state
machine. That trade is the whole design -- a durable scheduler is only worth
building once RL rollout defines what it must recover, and guessing now would get
the shape wrong.

What it does provide, because a batch of a few hundred tasks cannot work without
it:

- **bounded concurrency** -- AgentENV's Firecracker pool, host memory and provider
  rate limits all cap out well before "one VM per task".
- **failure isolation** -- one task's exception must not take the batch down, and
  a task that hangs must free its slot (per-task timeout).
- **retry classification** -- a provider rate limit is worth retrying; a context
  overflow or a safety refusal is not. Retrying everything corrupts scores by
  turning environment noise into apparent agent failure.
- **resumability by skipping** -- a re-run skips tasks whose journal already
  reports a terminal status, so an interrupted batch continues instead of
  restarting.

    python -m harness batch tasks.jsonl --slot codex --workers 8 --out runs/b1

``tasks.jsonl``: one JSON object per line, keys matching TaskSpec plus ``id``::

    {"id": "django-11848", "prompt": "fix ...", "cwd": "/tmp/w1"}
    {"id": "sympy-2100",   "prompt": "fix ...", "cwd": "/tmp/w2", "model": "..."}
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from harness.core.events import RUN_FINISHED
from harness.core.journal import JournalWriter, read_journal
from harness.core.slot import TaskSpec
from harness.orchestrator.resources import ResourceLedger
from harness.slots import load_slot

#: Error substrings worth another attempt. Everything else is the agent's own
#: outcome and must be reported, not retried.
RETRYABLE = (
    "rate limit",
    "ratelimit",
    "429",
    "overloaded",
    "503",
    "502",
    "timeout",
    "connection",
    "temporarily unavailable",
    # Concurrency contention in an agent's own store, not an agent outcome:
    # opencode keeps sessions in SQLite and parallel lanes hit this immediately.
    "database is locked",
    "resource temporarily unavailable",
)

TERMINAL_STATUSES = ("completed", "error", "timeout", "killed")


def is_retryable(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = error.lower()
    if "context" in lowered and ("window" in lowered or "length" in lowered):
        return False   # will fail identically next time
    if "refus" in lowered or "safety" in lowered:
        return False   # a real outcome, not noise
    return any(token in lowered for token in RETRYABLE)


@dataclass
class TaskOutcome:
    task_id: str
    status: str
    attempts: int = 1
    journal: Optional[str] = None
    error: Optional[str] = None
    final_text: str = ""
    usage: Dict[str, object] = field(default_factory=dict)
    seconds: float = 0.0
    skipped: bool = False

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "attempts": self.attempts,
            "journal": self.journal,
            "error": self.error,
            "usage": self.usage,
            "seconds": round(self.seconds, 1),
            "skipped": self.skipped,
        }


def load_tasks(path: Union[str, Path]) -> List[dict]:
    tasks = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        payload.setdefault("id", "task-%d" % (index + 1))
        tasks.append(payload)
    return tasks


class BatchRunner:
    """In-memory registry + worker pool. One instance per batch."""

    def __init__(
        self,
        slot_name: str,
        out_dir: Union[str, Path],
        *,
        workers: int = 4,
        max_attempts: int = 2,
        timeout_s: float = 1800.0,
        ledger: Optional[ResourceLedger] = None,
        mcp_factory: Optional[Callable[[dict], object]] = None,
        on_update: Optional[Callable[[str, TaskOutcome], None]] = None,
        resume: bool = True,
    ) -> None:
        self.slot_name = slot_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, int(workers))
        self.max_attempts = max(1, int(max_attempts))
        self.timeout_s = timeout_s
        self.ledger = ledger or ResourceLedger(self.out_dir / "resources.jsonl")
        self.mcp_factory = mcp_factory
        self.on_update = on_update
        self.resume = resume

        #: task_id -> state. The registry: pending | running | <terminal status>
        self.states: Dict[str, str] = {}
        self.outcomes: Dict[str, TaskOutcome] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # --- registry ----------------------------------------------------------
    def _set_state(self, task_id: str, state: str) -> None:
        with self._lock:
            self.states[task_id] = state

    def snapshot_states(self) -> Dict[str, str]:
        with self._lock:
            return dict(self.states)

    def stop(self) -> None:
        """Ask workers to stop claiming new tasks (in-flight ones finish)."""
        self._stop.set()

    # --- execution ---------------------------------------------------------
    def journal_path(self, task_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in task_id)
        return self.out_dir / ("%s.jsonl" % safe)

    def already_done(self, task_id: str) -> Optional[str]:
        """Terminal status from a previous attempt's journal, if any."""
        path = self.journal_path(task_id)
        if not self.resume or not path.exists():
            return None
        try:
            for record in reversed(read_journal(path)):
                if record.get("type") == RUN_FINISHED:
                    status = record.get("status")
                    return status if status in TERMINAL_STATUSES else None
        except Exception:  # noqa: BLE001 - a torn journal is simply re-run
            return None
        return None

    def run_one(self, task: dict) -> TaskOutcome:
        task_id = str(task["id"])
        existing = self.already_done(task_id)
        if existing:
            outcome = TaskOutcome(
                task_id=task_id, status=existing, attempts=0,
                journal=str(self.journal_path(task_id)), skipped=True,
            )
            self._finish(task_id, outcome)
            return outcome

        started = time.time()
        outcome = TaskOutcome(task_id=task_id, status="error", attempts=0)

        for attempt in range(1, self.max_attempts + 1):
            if self._stop.is_set():
                outcome.status = "killed"
                outcome.error = "batch stopped before attempt %d" % attempt
                break
            self._set_state(task_id, "running" if attempt == 1 else "retry-%d" % attempt)
            outcome.attempts = attempt
            try:
                result = self._attempt(task, task_id, attempt)
            except Exception as exc:  # noqa: BLE001 - isolate task failures
                outcome.status = "error"
                outcome.error = "%s: %s" % (type(exc).__name__, exc)
            else:
                outcome.status = result.status
                outcome.error = result.error
                outcome.final_text = result.final_text
                outcome.usage = result.usage
                outcome.journal = str(self.journal_path(task_id))

            if outcome.status == "completed":
                break
            if attempt < self.max_attempts and is_retryable(outcome.error):
                continue
            break

        outcome.seconds = time.time() - started
        self._finish(task_id, outcome)
        return outcome

    def _attempt(self, task: dict, task_id: str, attempt: int):
        slot = load_slot(self.slot_name)()
        extra = dict(task.get("extra") or {})
        if self.slot_name == "claude-code":
            extra.setdefault("setting_sources", [])   # ignore local .claude config
        if self.slot_name.startswith("opencode"):
            # Both opencode drivers (server and CLI) share the SQLite session
            # store, so both need the isolation -- matching on the exact name
            # silently dropped it when the default driver was renamed.
            # Per-task, not per-attempt: a retry or a later fork must find the
            # session the first attempt wrote, and that lives in this db.
            extra.setdefault("data_home", str(self.out_dir / "state" / task_id))

        spec = TaskSpec(
            prompt=task["prompt"],
            cwd=task.get("cwd") or str(self.out_dir),
            model=task.get("model"),
            env=dict(task.get("env") or {}),
            timeout_s=float(task.get("timeout_s") or self.timeout_s),
            extra=extra,
        )
        mcp = self.mcp_factory(task) if self.mcp_factory else None
        run_id = "%s-a%d" % (task_id, attempt)

        with self.ledger.run(run_id) as claim:
            # Append: a retry keeps the previous attempt's events, and the
            # terminal status of the last RUN_FINISHED is what `already_done`
            # reads, so a successful retry supersedes a failed attempt.
            with JournalWriter(
                self.journal_path(task_id), run_id=run_id, agent_id=task_id
            ) as journal:
                claim.attach(journal)     # snapshots recorded for reaping
                return slot.run(spec, journal, mcp)

    def _finish(self, task_id: str, outcome: TaskOutcome) -> None:
        self._set_state(task_id, outcome.status)
        with self._lock:
            self.outcomes[task_id] = outcome
        if self.on_update:
            try:
                self.on_update(task_id, outcome)
            except Exception:  # noqa: BLE001 - a reporter cannot break the batch
                pass

    def run(self, tasks: List[dict]) -> List[TaskOutcome]:
        for task in tasks:
            self.states.setdefault(str(task["id"]), "pending")

        results: List[TaskOutcome] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures: Dict[Future, str] = {
                pool.submit(self.run_one, task): str(task["id"]) for task in tasks
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - should not happen
                    outcome = TaskOutcome(
                        task_id=task_id, status="error",
                        error="worker crashed: %s" % exc,
                    )
                    self._finish(task_id, outcome)
                    results.append(outcome)

        summary_path = self.out_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "slot": self.slot_name,
                    "workers": self.workers,
                    "tasks": len(tasks),
                    "counts": self.counts(),
                    "outcomes": [o.as_dict() for o in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return results

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for outcome in self.outcomes.values():
            key = "skipped" if outcome.skipped else outcome.status
            counts[key] = counts.get(key, 0) + 1
        return counts
