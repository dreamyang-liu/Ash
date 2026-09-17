"""Long collection must retain ownership without weakening fencing."""

import json
from threading import Event
import time
from types import SimpleNamespace

import pytest

from runstore.files import JournalFrame
from runstore.store import Conflict, Fenced
from runstore.tests.test_store import request, store
from runstore.watchdog import LeaseKeeper, Watchdog
from runstore.worker import Worker


def test_cached_frame_detects_rewrite_and_preserves_partial_line(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    path.write_bytes(b'{"seq":1}\n{"seq":')
    frame = JournalFrame()
    assert frame.read(path) == [{"seq": 1}]
    path.write_bytes(b'{"seq":1}\n{"seq":2}\n')
    assert frame.read(path) == [{"seq": 1}, {"seq": 2}]
    path.write_bytes(b'{"seq":9}\n{"seq":2}\n')
    with pytest.raises(Conflict, match="rewritten"):
        frame.read(path)


def test_worker_persists_only_new_events_in_bounded_transactions(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    events = [{"seq": i, "type": "raw.fixture"} for i in range(1, 1026)]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    batches = []
    worker = Worker(SimpleNamespace(append_events=lambda j, t, batch: batches.append(batch)),
                    {"artifact_root": str(tmp_path)}, snapshot_valid=lambda point: True)
    worker.index = SimpleNamespace(project=lambda *args: None)
    job = {"id": "job", "lease_token": "token", "request": {"context": {}, "spec": {}}}
    for _ in range(3):
        worker._ingest(job, tmp_path, {"effective_spec": {}})
    assert max(map(len, batches)) <= 256
    assert [event for batch in batches for event in batch] == events


def test_keeper_renews_during_collection_and_cannot_revive_fenced_owner(store):
    store.submit(request(), "long-collection")
    job = store.claim("worker", lease_s=0.8)
    watchdog = Watchdog(lambda identity: None, Event(), 0.8)
    keeper = LeaseKeeper(store, job, 0.8, watchdog)
    try:
        time.sleep(2)
        assert store.expire() == []
        assert watchdog.reason is None
        store.finish(job["id"], job["lease_token"], {}, state="quarantined")
        adopted = store.adopt_quarantined(job["id"], "replacement", 10)
        time.sleep(0.3)
        with pytest.raises(Fenced):
            store.append_events(job["id"], job["lease_token"], [{"seq": 1}])
        assert store.get(job["id"])["lease_token"] == adopted["lease_token"]
    finally:
        keeper.close()
        watchdog.close()


def test_reconcile_renews_through_slow_final_ingestion(store, tmp_path, monkeypatch):
    store.submit(request(), "reconcile")
    job = store.claim("old")
    envelope = {"version": 1, "job_id": job["id"], "attempt_id": job["active_attempt"],
                "kind": "rollout", "effective_spec": {}, "profile_config": {}, "recovery": None}
    store.freeze_payload(job["id"], job["lease_token"], envelope)
    store.finish(job["id"], job["lease_token"], {}, state="quarantined")
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {}}, lease_s=0.8,
                    snapshot_valid=lambda point: True)

    def slow_ingest(*args):
        time.sleep(2)
        assert store.expire() == []
        return []

    monkeypatch.setattr(worker, "_ingest", slow_ingest)
    monkeypatch.setattr(worker, "_cleanup", lambda *args: True)
    assert worker.reconcile(job["id"])
    assert store.get(job["id"])["state"] == "failed"
