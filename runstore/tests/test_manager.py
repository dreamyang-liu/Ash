from collections import deque
from pathlib import Path
import multiprocessing
import subprocess
from threading import Event, Lock, Thread, get_ident
import time
from types import SimpleNamespace

import pytest

from runstore.manager import WorkerManager
from runstore.tests.test_payload import envelope
from runstore.tests.test_store import request, store
from runstore.worker import Worker, process_identity, stop_process


def eventually(predicate, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert predicate()


class Queue:
    def __init__(self, count: int) -> None:
        self.jobs = deque({"id": str(index)} for index in range(count))
        self.claimers = []

    def expire(self) -> list:
        return []

    def recoverable_jobs(self, **kwargs) -> list:
        return []

    def claim(self, *args, **kwargs) -> dict | None:
        self.claimers.append(get_ident())
        return self.jobs.popleft() if self.jobs else None


def test_one_claim_loop_bounds_concurrency_and_refills_after_failure():
    queue = Queue(30)
    stop = Event()
    lock = Lock()
    first_wave = Event()
    released = Event()
    active = set()
    started = []
    completed = []
    peaks = []

    def execute(job, **kwargs):
        with lock:
            active.add(job["id"])
            started.append(job["id"])
            peaks.append(len(active))
            if len(active) == 4:
                first_wave.set()
        released.wait(10)
        try:
            if job["id"] == "0":
                raise ValueError("One attempt fails without stopping the manager")
            time.sleep(0.02)
        finally:
            with lock:
                active.remove(job["id"])
                completed.append(job["id"])
                if len(completed) == 30:
                    stop.set()

    worker = SimpleNamespace(store=queue, worker_id="one-manager", lease_s=60, run_claimed=execute)
    manager = WorkerManager(worker, concurrency=4, poll_s=0.01)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert first_wave.wait(5)
        assert len(started) == 4
        released.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert len(set(started)) == len(completed) == 30
        assert max(peaks) == 4
        assert set(queue.claimers) == {thread.ident}
    finally:
        released.set()
        stop.set()
        thread.join(timeout=5)


def test_manager_backs_off_queue_errors_and_keeps_running():
    queue = Queue(1)
    original = queue.claim
    calls = []

    def flaky(*args, **kwargs):
        calls.append(time.monotonic())
        if len(calls) <= 2:
            raise OSError("Database temporarily unavailable")
        return original(*args, **kwargs)

    queue.claim = flaky
    executed = []
    worker = SimpleNamespace(store=queue, worker_id="fixture", lease_s=60,
                             run_claimed=lambda job, **kwargs: executed.append(job["id"]))
    WorkerManager(worker, poll_s=0.03).run(once=True)
    assert executed == ["0"] and len(calls) == 3
    assert calls[1] - calls[0] >= 0.025
    assert calls[2] - calls[1] >= 0.05


def test_once_claims_at_most_one_new_job_even_with_large_capacity():
    queue = Queue(20)
    executed = []
    worker = SimpleNamespace(store=queue, worker_id="fixture", lease_s=60,
                             run_claimed=lambda job, **kwargs: executed.append(job["id"]))
    WorkerManager(worker, concurrency=24, poll_s=0.01).run(once=True)
    assert executed == ["0"] and len(queue.jobs) == 19


def test_drain_before_start_admits_nothing_and_preserves_caller_stop():
    queue = Queue(3)
    stop = Event()
    manager = WorkerManager(SimpleNamespace(store=queue, worker_id="drained"))
    manager.request_drain()
    manager.request_drain()
    manager.run(stop=stop)
    assert manager.draining and not manager.active
    assert not stop.is_set() and not queue.claimers and len(queue.jobs) == 3


def drain_signal_inside_event_lock() -> None:
    import signal

    manager = WorkerManager(SimpleNamespace(worker_id="signal-lock"))
    signal.signal(signal.SIGUSR1, lambda *_: manager.request_drain())
    with manager.wake._cond:
        signal.raise_signal(signal.SIGUSR1)
    assert manager.draining


def test_drain_signal_cannot_deadlock_an_interrupted_manager_event():
    process = multiprocessing.get_context("spawn").Process(target=drain_signal_inside_event_lock)
    process.start()
    try:
        process.join(timeout=5)
        assert not process.is_alive() and process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)


def test_drain_finishes_active_work_without_refilling_or_cancelling():
    queue = Queue(4)
    entered, release, stop = Event(), Event(), Event()
    completed = []

    def execute(job, *, stop):
        entered.set()
        assert release.wait(5)
        assert not stop.is_set()
        completed.append(job["id"])

    manager = WorkerManager(SimpleNamespace(store=queue, worker_id="drain", lease_s=60,
                                            run_claimed=execute), poll_s=0.01, shutdown_s=0.01)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert entered.wait(5)
        manager.request_drain()
        thread.join(timeout=0.1)  # Drain must outlive the cancellation grace.
        assert thread.is_alive() and not stop.is_set()
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and completed == ["0"]
        assert len(queue.jobs) == 3 and not stop.is_set()
    finally:
        release.set()
        stop.set()
        thread.join(timeout=5)


def test_drain_during_claim_supervises_the_returned_job():
    queue = Queue(2)
    entered, release, stop = Event(), Event(), Event()
    original_claim = queue.claim
    completed = []

    def claim(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_claim(*args, **kwargs)

    queue.claim = claim
    manager = WorkerManager(SimpleNamespace(store=queue, worker_id="claim", lease_s=60,
        run_claimed=lambda job, **kw: completed.append((job["id"], kw["stop"].is_set()))), poll_s=0.01)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert entered.wait(5)
        manager.request_drain()
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and completed == [("0", False)]
        assert len(queue.jobs) == 1 and not stop.is_set()
    finally:
        release.set()
        stop.set()
        thread.join(timeout=5)


@pytest.mark.parametrize("during", ["expire", "recoverable_jobs"])
def test_drain_during_queue_scan_stops_recovery_and_new_claims(during):
    queue = Queue(2)
    recovered = []
    manager = WorkerManager(SimpleNamespace(store=queue, worker_id="scan", lease_s=60,
                                            reconcile=lambda job, **kw: recovered.append(job)))

    def scan(**kwargs):
        manager.request_drain()
        return [{"id": "expired-1"}, {"id": "expired-2"}]

    setattr(queue, during, scan)
    manager.run()
    assert not recovered and not queue.claimers and len(queue.jobs) == 2


def test_stop_during_drain_still_cancels_owned_work():
    queue = Queue(2)
    entered, stop = Event(), Event()

    def execute(job, *, stop):
        entered.set()
        assert stop.wait(5)

    manager = WorkerManager(SimpleNamespace(store=queue, worker_id="stop", lease_s=60,
                                            run_claimed=execute), poll_s=0.01)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert entered.wait(5)
        manager.request_drain()
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(queue.jobs) == 1
    finally:
        stop.set()
        thread.join(timeout=5)


def test_shutdown_does_not_wait_forever_for_blocked_io():
    queue = Queue(2)
    entered, release, stop = Event(), Event(), Event()

    def execute(job, **kwargs):
        entered.set()
        release.wait(10)

    manager = WorkerManager(SimpleNamespace(store=queue, worker_id="fixture", lease_s=60,
                                            run_claimed=execute), poll_s=0.01, shutdown_s=0.1)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert entered.wait(5)
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(manager.active) == 1
        assert len(queue.jobs) == 1
    finally:
        release.set()
        stop.set()
        thread.join(timeout=5)
        eventually(lambda: all(future.done() for future in manager.active.values()))


@pytest.fixture
def fixture_children(monkeypatch):
    popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        assert command[2] == "runstore.child"
        process = popen([*command[:2], "runstore.tests.payload_child", *command[3:]], **kwargs)
        processes.append((process, process_identity(process.pid)))
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    yield processes
    for process, identity in processes:
        stop_process(identity)
        process.wait(timeout=5)


def config(root: Path) -> dict:
    return {"artifact_root": str(root), "profiles": {"default": {"run_defaults": {
        "sandbox_image": "fixture", "backend": {"backend": "microvm"}}}}}


def attempt_directory(store, root: Path, job: dict) -> Path:
    return root / job["id"] / store.get(job["id"])["active_attempt"]


def test_blocked_indexing_does_not_block_sibling_heartbeat_refill_or_deadline(store, tmp_path, fixture_children):
    blocked = store.submit(request(prompt="hold-fixture", timeout_s=2), "blocked")
    sibling = store.submit(request(prompt="gate-fixture", timeout_s=30), "sibling")
    next_job = store.submit(request(), "next")
    worker = Worker(store, config(tmp_path), lease_s=5)
    ingest = worker._ingest
    entered, release, stop = Event(), Event(), Event()

    def blocked_ingest(job, directory, payload):
        if job["id"] == blocked["id"]:
            entered.set()
            release.wait(20)
        return ingest(job, directory, payload)

    worker._ingest = blocked_ingest
    manager = WorkerManager(worker, concurrency=2, poll_s=0.02)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert entered.wait(10)
        eventually(lambda: store.get(sibling["id"])["active_attempt"] is not None)
        sibling_directory = attempt_directory(store, tmp_path, sibling)
        eventually(lambda: (sibling_directory / "executed.json").exists())
        assert store.get(next_job["id"])["state"] == "queued"
        sibling_lease = store.get(sibling["id"])["lease_until"]
        blocked_attempt = store.attempts(blocked["id"])[0]
        blocked_identity = blocked_attempt["execution"]["process"]
        eventually(lambda: process_identity(blocked_identity["pid"]) is None, timeout=6)
        assert not release.is_set()
        eventually(lambda: store.get(sibling["id"])["lease_until"] > sibling_lease)
        assert not manager.active[blocked["id"]].done()
        (sibling_directory / "release").touch()
        eventually(lambda: store.get(next_job["id"])["state"] == "succeeded")
        assert store.get(sibling["id"])["state"] == "succeeded"
        release.set()
        eventually(lambda: blocked["id"] not in manager.active)
        assert len(store.attempts(next_job["id"])) == 1
    finally:
        release.set()
        stop.set()
        thread.join(timeout=25)
        assert not thread.is_alive()


def test_malformed_recovery_is_isolated_and_other_jobs_run(store, tmp_path, fixture_children):
    broken = store.submit(request(), "broken-recovery")
    claim = store.claim("dead-manager")
    store.freeze_payload(claim["id"], claim["lease_token"], envelope(claim))
    directory = tmp_path / broken["id"] / claim["active_attempt"]
    directory.mkdir(parents=True)
    (directory / "outcome.json").write_text("malformed outcome")
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                       (broken["id"],))
    healthy = store.submit(request(), "healthy")
    worker = Worker(store, config(tmp_path))
    WorkerManager(worker, poll_s=0.01).run(once=True)
    assert store.get(broken["id"])["state"] == "quarantined"
    assert "JSONDecodeError" in store.get(broken["id"])["error"]
    assert store.get(healthy["id"])["state"] == "succeeded"


def test_attempt_index_exception_does_not_stop_manager_or_siblings(store, tmp_path, fixture_children):
    broken = store.submit(request(prompt="hold-fixture"), "broken-index")
    healthy = store.submit(request(), "healthy")
    worker = Worker(store, config(tmp_path))
    ingest = worker._ingest

    def invalid_index(job, directory, payload):
        if job["id"] == broken["id"]:
            raise ValueError("Malformed trajectory fixture")
        return ingest(job, directory, payload)

    worker._ingest = invalid_index
    stop = Event()
    manager = WorkerManager(worker, concurrency=2, poll_s=0.01)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        eventually(lambda: store.get(healthy["id"])["state"] == "succeeded")
        eventually(lambda: store.get(broken["id"])["state"] == "quarantined")
        assert "Malformed trajectory" in store.get(broken["id"])["error"]
        assert thread.is_alive()
    finally:
        stop.set()
        thread.join(timeout=25)
        assert not thread.is_alive()


def test_cli_routes_concurrency_and_once_to_manager(tmp_path, monkeypatch):
    import json
    import sys

    from runstore import __main__ as cli

    configuration = tmp_path / "worker.json"
    configuration.write_text(json.dumps(config(tmp_path / "artifacts")))
    captured = {}
    handlers = {}

    class Manager:
        def __init__(self, worker, *, concurrency):
            captured["concurrency"] = concurrency

        def request_drain(self):
            captured["drain_requests"] = captured.get("drain_requests", 0) + 1

        def run(self, *, stop, once):
            captured["once"] = once
            assert not stop.is_set()
            handlers[cli.signal.SIGUSR1](None, None)
            assert not stop.is_set()
            handlers[cli.signal.SIGTERM](None, None)
            assert stop.is_set()

    monkeypatch.setenv("ASH_RUNSTORE_DSN", "fixture-not-connected")
    monkeypatch.setattr(sys, "argv", ["runstore", "worker", "--config", str(configuration),
                                    "--concurrency", "24", "--once"])
    monkeypatch.setattr(cli, "volatile_reason", lambda root: None)
    monkeypatch.setattr(cli.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    monkeypatch.setattr("runstore.manager.WorkerManager", Manager)
    cli.main()
    assert captured == {"concurrency": 24, "once": True, "drain_requests": 1}


def test_drain_preserves_live_child_lease_and_sibling_worker(store, tmp_path, fixture_children):
    gated = store.submit(request(prompt="gate-fixture", timeout_s=30), "draining-job")
    stop = Event()
    manager = WorkerManager(Worker(store, config(tmp_path), worker_id="draining", lease_s=3),
                            poll_s=0.02, shutdown_s=0.1)
    peer = WorkerManager(Worker(store, config(tmp_path), worker_id="peer", lease_s=3), poll_s=0.02)
    thread = Thread(target=lambda: manager.run(stop=stop))
    peer_thread = Thread(target=lambda: peer.run(stop=stop))
    thread.start()
    directory = None
    try:
        eventually(lambda: bool(store.get(gated["id"])["active_attempt"]))
        directory = attempt_directory(store, tmp_path, gated)
        eventually(lambda: (directory / "executed.json").exists())
        attempt = store.attempts(gated["id"])[0]
        identity = attempt["execution"]["process"]
        lease = store.get(gated["id"])["lease_until"]
        manager.request_drain()
        queued = store.submit(request(), "peer-job")
        peer_thread.start()
        eventually(lambda: store.get(queued["id"])["state"] == "succeeded")
        assert store.attempts(queued["id"])[0]["worker_id"] == "peer"
        time.sleep(3.2)  # Exceed the draining worker's original lease duration.
        assert store.get(gated["id"])["lease_until"] > lease and store.expire() == []
        assert process_identity(identity["pid"]) == identity
        assert thread.is_alive() and not stop.is_set()
        (directory / "release").touch()
        thread.join(timeout=10)
        assert not thread.is_alive() and not stop.is_set()
        assert store.get(gated["id"])["state"] == "succeeded"
        following = store.submit(request(), "after-drain")
        eventually(lambda: store.get(following["id"])["state"] == "succeeded")
        assert store.attempts(following["id"])[0]["worker_id"] == "peer"
        assert peer_thread.is_alive()
        assert len(store.attempts(gated["id"])) == 1
    finally:
        if directory is not None:
            (directory / "release").touch()
        stop.set()
        thread.join(timeout=15)
        if peer_thread.ident is not None:
            peer_thread.join(timeout=15)
        assert not thread.is_alive() and not peer_thread.is_alive()


def run_drain_cli_fixture(dsn: str, root: str) -> None:
    import os
    import sys
    from runstore import __main__ as cli

    popen = subprocess.Popen

    def launch(command, **kwargs):
        return popen([*command[:2], "runstore.tests.payload_child", *command[3:]], **kwargs)

    subprocess.Popen = launch
    cli.volatile_reason = lambda _: None  # Only these disposable test artifacts use /tmp.
    os.environ["ASH_RUNSTORE_DSN"] = dsn
    sys.argv = ["runstore", "worker", "--config", str(Path(root) / "cli-config.json")]
    cli.main()


def test_cli_sigusr1_drains_real_worker_without_interrupting_child(store, tmp_path):
    import json
    import os
    import signal

    gated = store.submit(request(prompt="gate-fixture", timeout_s=30), "signal-job")
    queued = store.submit(request(), "queued-after-signal")
    (tmp_path / "cli-config.json").write_text(json.dumps(config(tmp_path)))
    process = multiprocessing.get_context("spawn").Process(
        target=run_drain_cli_fixture, args=(store.dsn, str(tmp_path)))
    identity = directory = None
    process.start()
    try:
        eventually(lambda: bool(store.get(gated["id"])["active_attempt"]))
        directory = attempt_directory(store, tmp_path, gated)
        eventually(lambda: (directory / "executed.json").exists())
        attempt = store.attempts(gated["id"])[0]
        identity = attempt["execution"]["process"]
        os.kill(process.pid, signal.SIGUSR1)
        time.sleep(0.2)
        assert process.is_alive() and process_identity(identity["pid"]) == identity
        (directory / "release").touch()
        process.join(timeout=15)
        assert not process.is_alive() and process.exitcode == 0
        assert store.get(gated["id"])["state"] == "succeeded"
        assert len(store.attempts(gated["id"])) == 1
        assert store.get(queued["id"])["state"] == "queued"
        assert process_identity(identity["pid"]) is None
    finally:
        if directory is not None:
            (directory / "release").touch()
        if process.is_alive():
            process.terminate()
            process.join(timeout=15)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        stop_process(identity)


def test_shutdown_stops_all_owned_children_and_leaves_queued_jobs(store, tmp_path, fixture_children):
    jobs = [store.submit(request(prompt="hold-fixture", timeout_s=30), f"job-{index}") for index in range(3)]
    worker = Worker(store, config(tmp_path))
    manager = WorkerManager(worker, concurrency=2, poll_s=0.02)
    stop = Event()
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        eventually(lambda: len(fixture_children) == 2)
        for job in jobs[:2]:
            directory = attempt_directory(store, tmp_path, job)
            eventually(lambda: (directory / "executed.json").exists())
        stop.set()
        thread.join(timeout=25)
        assert not thread.is_alive()
        assert len(fixture_children) == 2
        assert all(process_identity(identity["pid"]) is None for process, identity in fixture_children)
        assert store.get(jobs[2]["id"])["state"] == "queued"
        assert all(store.get(job["id"])["state"] in {"failed", "quarantined"} for job in jobs[:2])
    finally:
        stop.set()
        thread.join(timeout=25)


def test_expiry_scan_skips_locked_job_instead_of_blocking_dispatch(store):
    locked = store.submit(request(), "locked")
    store.claim("worker")
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                       (locked["id"],))
    other = store.submit(request(), "other")
    with store.transaction() as cursor:
        cursor.execute("SELECT id FROM rs_jobs WHERE id=%s FOR UPDATE", (locked["id"],))
        started = time.monotonic()
        assert store.expire() == []
        assert store.claim("manager")["id"] == other["id"]
        assert time.monotonic() - started < 2
    assert store.expire() == [locked["id"]]


def test_store_sets_sql_and_connection_timeouts(store):
    with store.transaction() as cursor:
        cursor.execute("SHOW statement_timeout")
        assert cursor.fetchone()["statement_timeout"] == "5s"
        settings = cursor.connection.get_dsn_parameters()
        assert settings["connect_timeout"] == "5"
        assert settings["tcp_user_timeout"] == "10000"
        assert settings["keepalives_count"] == "2"


def run_crash_fixture(dsn: str, root: str) -> None:
    from runstore.store import Store

    popen = subprocess.Popen

    def launch(command, **kwargs):
        return popen([*command[:2], "runstore.tests.payload_child", *command[3:]], **kwargs)

    subprocess.Popen = launch
    WorkerManager(Worker(Store(dsn), config(Path(root))), concurrency=2, poll_s=0.02).run()


def test_manager_sigkill_reconciles_all_attempts_then_refills(store, tmp_path, fixture_children):
    jobs = [store.submit(request(prompt="hold-fixture", timeout_s=30), f"crash-{index}") for index in range(2)]
    following = store.submit(request(), "following")
    controller = multiprocessing.get_context("spawn").Process(target=run_crash_fixture, args=(store.dsn, str(tmp_path)))
    controller.start()
    identities = []
    try:
        for job in jobs:
            eventually(lambda: bool(store.attempts(job["id"])) and
                       bool(store.attempts(job["id"])[0]["execution"].get("process")))
            identities.append(store.attempts(job["id"])[0]["execution"]["process"])
            directory = attempt_directory(store, tmp_path, job)
            eventually(lambda: (directory / "executed.json").exists())
        controller.kill()
        controller.join(timeout=5)
        assert not controller.is_alive() and controller.exitcode == -9
        with store.transaction() as cursor:
            cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=ANY(%s)",
                           ([job["id"] for job in jobs],))
        WorkerManager(Worker(store, config(tmp_path)), concurrency=2, poll_s=0.02).run(once=True)
        assert all(store.get(job["id"])["state"] == "failed" for job in jobs)
        assert store.get(following["id"])["state"] == "succeeded"
        assert all(process_identity(identity["pid"]) is None for identity in identities)
        assert all(len(store.attempts(job["id"])) == 1 for job in [*jobs, following])
    finally:
        if controller.is_alive():
            controller.kill()
        controller.join(timeout=5)
        for identity in identities:
            stop_process(identity)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ValueError):
        WorkerManager(None, concurrency=capacity)
