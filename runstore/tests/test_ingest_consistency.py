"""Concurrent journal/native growth must not combine different observation frames."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from runstore.native import index_native
from runstore.worker import Worker


def tool_events(depth: int) -> list[dict]:
    return [
        {"seq": depth * 3, "type": "tool.started", "call_id": f"call-{depth}",
         "name": "ash__shell", "args": {}},
        {"seq": depth * 3 + 1, "type": "checkpoint.captured", "step": depth,
         "call_id": f"call-{depth}", "pairing": "call-id-v1", "prefix_complete": True,
         "snapshot_id": f"snapshot-{depth}", "session_ckpt": "session", "reason": "captured"},
        {"seq": depth * 3 + 2, "type": "tool.finished", "call_id": f"call-{depth}", "output": "ok"},
    ]


def native_call(depth: int, output: bool = False) -> dict:
    return {"type": "response_item", "payload": {
        "type": "function_call_output" if output else "function_call", "call_id": f"call-{depth}"}}


def append(path: Path, entries: list[dict]) -> None:
    with path.open("a") as stream:
        for entry in entries:
            stream.write(json.dumps(entry) + "\n")


def fixture_files(directory: Path) -> tuple[Path, Path]:
    journal = directory / "trajectory.jsonl"
    native = directory / "native-home/sessions/rollout-session.jsonl"
    native.parent.mkdir(parents=True)
    append(journal, [{"seq": 1, "type": "checkpoint.policy", "pairing": "call-id-v1"},
                     {"seq": 2, "type": "session.ref", "native_session_id": "session"}, *tool_events(1)])
    append(native, [{"type": "session_meta", "payload": {"id": "session"}}])
    return journal, native


def test_ingestion_uses_one_journal_frame_when_writer_advances(tmp_path):
    journal, native = fixture_files(tmp_path)
    completed = {"type": "event_msg", "payload": {"type": "token_count"}}
    append(native, [native_call(1), native_call(1, True), completed])
    observations = []
    advanced = False

    def persist(job_id, token, events):
        nonlocal advanced
        if not advanced:
            append(journal, tool_events(2))
            append(native, [native_call(2), native_call(2, True), completed])
            advanced = True

    def project(job_id, token, scope, events, points):
        tool_count = sum(event.get("type") == "tool.started" for event in events)
        assert all(point.tool_depth <= tool_count for point in points)
        observations.append([point.tool_depth for point in points])

    worker = Worker(SimpleNamespace(append_events=persist), {
        "artifact_root": str(tmp_path), "profiles": {}}, snapshot_valid=lambda point: True)
    worker.index = SimpleNamespace(project=project)
    job = {"id": "job", "lease_token": "lease", "request": {
        "spec": {"slot": "codex"}, "context": {}, "profile_hash": None}}
    envelope = {"effective_spec": {"slot": "codex"}, "recovery": None}
    (tmp_path / "request.json").write_text("invalid obsolete file")
    worker._ingest(job, tmp_path, envelope)
    worker._ingest(job, tmp_path, envelope)
    assert observations == [[1], [1, 2]]


def test_unobserved_tool_in_same_native_response_is_not_a_complete_prefix(tmp_path):
    journal, native = fixture_files(tmp_path)
    append(native, [native_call(1), native_call(2), native_call(1, True), native_call(2, True),
                    {"type": "event_msg", "payload": {"type": "token_count"}}])
    assert index_native(journal, native, "codex", "session") == []
    append(journal, tool_events(2))
    points = index_native(journal, native, "codex", "session")
    assert [(point.message_step, point.tool_depth) for point in points] == [(1, 2)]


def test_inherited_tools_require_an_explicit_pinned_native_prefix(tmp_path):
    journal, native = fixture_files(tmp_path)
    completed = {"type": "event_msg", "payload": {"type": "token_count"}}
    append(native, [native_call(0), native_call(0, True), completed])
    data = native.read_bytes()
    inherited = {"slot": "codex", "path": str(native), "byte_length": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
    append(native, [native_call(1), native_call(1, True), completed])
    assert index_native(journal, native, "codex", "session") == []
    points = index_native(journal, native, "codex", "session", inherited_native=inherited)
    assert [point.tool_depth for point in points] == [1]


def test_frozen_frame_does_not_reopen_the_growing_journal(tmp_path, monkeypatch):
    from harness import rollback

    journal, native = fixture_files(tmp_path)
    append(native, [native_call(1), native_call(1, True),
                    {"type": "event_msg", "payload": {"type": "token_count"}}])
    events = [json.loads(line) for line in journal.read_text().splitlines()]

    def refuse(*args, **kwargs):
        raise AssertionError("Reopened journal after selecting the observation frame")

    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(rollback, "read_journal", refuse)
    assert len(index_native(journal, native, "codex", "session", events=events)) == 1
