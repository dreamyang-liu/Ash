import hashlib

from runstore.index import Index
from runstore.native import NativePoint
from runstore.tests.test_store import request, store


def test_prefix_response_and_ancestor_fallback_never_mix_states(store, tmp_path):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    events = []
    scope = {"base_image": "immutable-image", "task": "fixture"}
    for depth in range(1, 4):
        events.extend([
            {"seq": depth * 2 - 1, "type": "tool.started", "call_id": str(depth),
             "name": "shell", "args": {"command": f"step-{depth}"}},
            {"seq": depth * 2, "type": "tool.finished", "call_id": str(depth),
             "output": f"response-{depth}"},
        ])
    prefix = tmp_path / "native.jsonl"
    prefix.write_bytes(b'{"fixture":true}\n')
    native = {"path": str(prefix), "byte_length": prefix.stat().st_size,
              "sha256": hashlib.sha256(prefix.read_bytes()).hexdigest()}
    available = {"snapshot-1", "snapshot-3"}
    index = Index(store, lambda point: point["snapshot_id"] in available)
    index.project(job["id"], claim["lease_token"], scope, events, [
        NativePoint(1, 1, "snapshot-1", native), NativePoint(2, 3, "snapshot-3", native)])
    calls = [{"name": "shell", "arguments": {"command": f"step-{depth}"}} for depth in range(1, 4)]
    exact = index.query(scope, calls)
    assert exact["exact"] and exact["matched_depth"] == 3
    assert exact["matches"][0]["recovery_depth"] == 3
    assert [result["output"] for result in exact["matches"][0]["responses"]] == [
        "response-1", "response-2", "response-3"]
    partial = index.query(scope, [*calls[:2], {"name": "shell", "arguments": {"command": "different"}}])
    assert partial["matched_depth"] == 2
    assert partial["matches"][0]["recovery_depth"] == 1
    assert partial["matches"][0]["replay_suffix"][0] == calls[1]
    available.remove("snapshot-3")
    assert index.query(scope, calls)["matches"][0]["recovery"]["snapshot_id"] == "snapshot-1"
    prefix.write_bytes(b'corrupted\n')
    assert index.query(scope, calls)["matches"][0]["recovery"] is None
    assert index.query({**scope, "task": "other"}, calls)["matched_depth"] == 0


def test_later_boundary_invalidation_revokes_published_recovery(store, tmp_path):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    prefix = tmp_path / "native.jsonl"
    prefix.write_bytes(b'{}\n')
    native = {"path": str(prefix), "byte_length": 3,
              "sha256": hashlib.sha256(prefix.read_bytes()).hexdigest()}
    events = [{"type": "tool.started", "call_id": "call", "name": "shell", "args": {}},
              {"type": "tool.finished", "call_id": "call", "output": "response"}]
    index = Index(store, lambda point: True)
    index.project(job["id"], claim["lease_token"], {}, events, [NativePoint(1, 1, "snapshot", native)])
    assert index.points(claim["active_attempt"])[0]["valid"]
    index.project(job["id"], claim["lease_token"], {}, events, [])
    assert not index.points(claim["active_attempt"])[0]["valid"]
