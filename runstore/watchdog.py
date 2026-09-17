"""Per-attempt deadline and lease supervision with no database or journal reads."""

from __future__ import annotations

import logging
from threading import Event, Lock, Thread
import time
from typing import Callable


class AttemptStopped(RuntimeError):
    pass


class Watchdog:
    def __init__(self, stop_process: Callable, stop: Event, lease_s: float) -> None:
        self.stop_process = stop_process
        self.stop = stop
        self.lease_s = lease_s
        self.lease_deadline = time.monotonic() + lease_s
        self.deadline: float | None = None
        self.identity: dict | None = None
        self.running: Callable | None = None
        self.reason: str | None = None
        self.lock = Lock()
        self.finished = Event()
        self.thread = Thread(target=self._run, name="attempt-watchdog", daemon=True)
        self.thread.start()

    def renew(self, started: float) -> None:
        with self.lock:
            self.lease_deadline = max(self.lease_deadline, started + self.lease_s)
        self.check()

    def arm(self, identity: dict, deadline: float, *, running: Callable | None = None) -> None:
        with self.lock:
            self.identity = identity
            self.deadline = deadline
            self.running = running
            stopped = self.reason is not None
        if stopped:
            self.stop_process(identity)
        self.check()

    def check(self) -> None:
        if self.reason is not None:
            raise AttemptStopped(self.reason)
        if self.stop.is_set():
            raise AttemptStopped("worker_shutdown")

    def close(self) -> None:
        self.finished.set()
        self.thread.join(timeout=16)

    def _run(self) -> None:
        while not self.finished.wait(0.05):
            now = time.monotonic()
            with self.lock:
                if self.stop.is_set():
                    reason = "worker_shutdown"
                elif (self.deadline is not None and now >= self.deadline
                      and (self.running is None or self.running())):
                    reason = "timeout"
                elif now >= self.lease_deadline:
                    reason = "lease_expired"
                else:
                    continue
                self.reason = reason
                identity = self.identity
            try:
                self.stop_process(identity)
            except Exception:
                logging.getLogger(__name__).exception("Attempt watchdog could not stop its process")
            return


class LeaseKeeper:
    """Renew independently of ingestion, native indexing and final collection.

    A failed/blocked renewal never extends the local watchdog deadline.
    Database fencing remains authoritative on every write.
    """

    def __init__(self, store, job: dict, lease_s: float, watchdog: Watchdog) -> None:
        self.finished = Event()

        def run() -> None:
            while not self.finished.wait(min(10, lease_s / 4)):
                try:
                    watchdog.check()
                    started = time.monotonic()
                    store.heartbeat(job["id"], job["lease_token"], lease_s=lease_s)
                    watchdog.renew(started)
                except Exception:
                    # A final publication can fence this last heartbeat too.
                    # In either case, do not renew a lease we cannot prove.
                    return

        self.thread = Thread(target=run, name="attempt-lease", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.finished.set()
        self.thread.join(timeout=16)
