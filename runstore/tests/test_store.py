"""Real PostgreSQL concurrency and fencing, not a SQLite approximation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import time
from uuid import uuid4

import pytest

from runstore.specs import JobSpec
from runstore.store import (
    Conflict,
    Fenced,
    Store,
    _COMPRESSED_EVENT,
    _loaded_event,
    _stored_event,
)


@pytest.fixture
def store():
    dsn = os.environ.get("ASH_RUNSTORE_TEST_DSN")
    if not dsn:
        pytest.skip("Set ASH_RUNSTORE_TEST_DSN to an isolated PostgreSQL database")
    pytest.importorskip("psycopg2")
    schema = "fixture_" + uuid4().hex
    admin = Store(dsn)
    with admin.transaction() as cursor:
        cursor.execute(f"CREATE SCHEMA {schema}")
    instance = Store(dsn + f" options='-c search_path={schema}'")
    instance.initialize()
    try:
        yield instance
    finally:
        with admin.transaction() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")


def request(**changes):
    return JobSpec("rollout", {"prompt": "fixture", **changes})


def test_idempotent_submit_and_restart(store):
    job = store.submit(request(), "fixture")
    assert Store(store.dsn).submit(request(), "fixture")["id"] == job["id"]
    with pytest.raises(Conflict):
        store.submit(request(prompt="different"), "fixture")


def test_concurrent_claimers_claim_each_job_once(store):
    jobs = [store.submit(request(), str(index))["id"] for index in range(24)]
    with ThreadPoolExecutor(max_workers=24) as pool:
        claims = list(pool.map(lambda number: store.claim(f"worker-{number}"), range(32)))
    assert sorted(claim["id"] for claim in claims if claim) == sorted(jobs)
    assert store.claim("late-worker") is None


def test_expiry_quarantines_instead_of_replaying_uncertain_execution(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker", lease_s=0.02)
    time.sleep(0.04)
    with pytest.raises(Fenced):
        store.heartbeat(job["id"], claim["lease_token"])
    assert store.expire() == [job["id"]]
    assert store.get(job["id"])["state"] == "quarantined"
    assert store.claim("other-worker") is None
    with pytest.raises(Fenced):
        store.finish(job["id"], claim["lease_token"], {"ok": True})


def test_cancel_intent_is_durable_for_queued_and_running_jobs(store):
    queued = store.submit(request(), "queued-cancel")
    store.request_cancel(queued["id"])
    assert store.get(queued["id"])["state"] == "cancelled"
    assert store.get(queued["id"])["cancel_requested"] is True

    running = store.submit(request(), "running-cancel")
    claim = store.claim("worker")
    assert claim["id"] == running["id"]
    store.request_cancel(running["id"])
    assert store.get(running["id"])["phase"] == "cancelling"
    assert store.heartbeat(running["id"], claim["lease_token"]) is True
    store.finish(
        running["id"], claim["lease_token"],
        {"status": "cancelled", "stop_reason": "cancelled"},
        state="cancelled",
    )
    assert store.get(running["id"])["state"] == "cancelled"


def test_cancel_uses_timeout_within_http_client_deadline():
    class Cursor:
        def __init__(self):
            self.queries = []

        def execute(self, query, parameters=None):
            self.queries.append((query, parameters))

        def fetchone(self):
            return {"state": "running"}

    cursor = Cursor()

    @contextmanager
    def transaction():
        yield cursor

    instance = Store("unused")
    instance.transaction = transaction
    instance.request_cancel("job")
    assert cursor.queries[0] == ("SET LOCAL statement_timeout = '20s'", None)
    assert "FOR UPDATE" in cursor.queries[1][0]


def test_cancel_waits_past_the_default_statement_timeout_for_a_row_lock(store):
    """Cancellation has its own timeout rather than inheriting the 5s default."""
    job = store.submit(request(), "cancel-lock-wait")
    claim = store.claim("worker")
    assert claim["id"] == job["id"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.transaction() as cursor:
            cursor.execute("SELECT id FROM rs_jobs WHERE id=%s FOR UPDATE", (job["id"],))
            cancellation = pool.submit(store.request_cancel, job["id"])
            time.sleep(5.5)
            assert not cancellation.done()
        cancellation.result(timeout=1)
    assert store.get(job["id"])["phase"] == "cancelling"


def test_large_event_encoding_happens_before_the_job_row_is_locked(monkeypatch):
    import runstore.store as store_module

    class Cursor:
        def __init__(self):
            self.queries = []

        def execute(self, query, parameters=None):
            self.queries.append((query, parameters))
            self.rowcount = 1

    cursor = Cursor()

    @contextmanager
    def transaction():
        yield cursor

    instance = Store("unused")
    instance.transaction = transaction
    observed = []

    def fence(_cursor, _job_id, _token):
        observed.append("locked")
        return {"active_attempt": "attempt"}

    event = {"seq": 1, "type": "rollout.session_state", "state": {"text": "x" * 1_100_000}}
    instance._fence = fence
    original = store_module._stored_event

    def encoded(value):
        observed.append("encoded")
        return original(value)

    monkeypatch.setattr(store_module, "_stored_event", encoded)
    instance.append_events("job", "lease", [event])
    assert observed == ["encoded", "locked"]


def test_result_is_durable_and_old_attempt_cannot_overwrite(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    store.finish(job["id"], claim["lease_token"], {"final_text": "fixture result"})
    assert Store(store.dsn).get(job["id"])["result"]["final_text"] == "fixture result"
    with pytest.raises(Fenced):
        store.finish(job["id"], claim["lease_token"], {"final_text": "stale"})


def test_journal_append_is_idempotent_but_not_rewritable(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    event = {"seq": 1, "type": "tool.started", "call_id": "first"}
    store.append_events(job["id"], claim["lease_token"], [event, event])
    assert store.events(claim["active_attempt"]) == [event]
    with pytest.raises(Conflict):
        store.append_events(job["id"], claim["lease_token"], [{**event, "call_id": "changed"}])


def test_large_events_are_compressed_losslessly():
    event = {"seq": 1, "type": "rollout.session_state", "state": {"text": "x" * 1_100_000}}
    stored = _stored_event(event)
    assert stored["encoding"] == "ash.runstore.zlib-json-v1"
    assert len(stored["data"]) < len(event["state"]["text"])
    assert _loaded_event(stored) == event


def test_large_event_round_trips_through_postgresql(store):
    job = store.submit(request(), "large-event")
    claim = store.claim("worker")
    assert claim["id"] == job["id"]
    event = {
        "seq": 1,
        "type": "rollout.session_state",
        "state": {"text": "repeated session state " * 100_000},
    }
    store.append_events(job["id"], claim["lease_token"], [event])
    assert store.events(claim["active_attempt"]) == [event]
    with store.transaction() as cursor:
        cursor.execute(
            "SELECT event FROM rs_events WHERE attempt_id=%s AND seq=1",
            (claim["active_attempt"],),
        )
        assert cursor.fetchone()["event"]["encoding"] == _COMPRESSED_EVENT


def test_compressed_event_marker_is_not_interpreted_without_all_fields():
    event = {"encoding": "ash.runstore.zlib-json-v1", "application": "payload"}
    assert _loaded_event(event) == event


def test_actor_retry_needs_verified_pair_and_preserves_cap(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    result = {"failure_kind": "infrastructure", "error": "transport"}
    with pytest.raises(ValueError, match="verified continuation"):
        store.finish(job["id"], claim["lease_token"], result, state="failed", retry=True)
    for number in range(3):
        store.finish(job["id"], claim["lease_token"], result, state="failed", retry=True,
                     recovery={"point_id": "paired", "timeout_s": 10})
        if number < 2:
            with store.transaction() as cursor:
                cursor.execute("UPDATE rs_jobs SET ready_at=clock_timestamp() WHERE id=%s", (job["id"],))
            claim = store.claim("worker")
    assert store.get(job["id"])["state"] == "failed"
    assert len(store.attempts(job["id"])) == 3


@pytest.mark.parametrize("updates", [
    {"session": "not-serializable-session"}, {"journal_path": "/tmp/old.jsonl"},
    {"slot": "codex-cli"}, {"cwd": "/some/host/repo"},
    {"backend": {"api_key": "should-not-be-persisted"}}, {"timeout_s": float("nan")},
])
def test_unsupported_or_unsafe_runs_rejected(updates):
    with pytest.raises(ValueError):
        request(**updates).validate()
