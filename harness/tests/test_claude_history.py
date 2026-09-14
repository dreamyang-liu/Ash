import hashlib
import json
from pathlib import Path

import pytest

from harness.slots.claude_history import find_prefix_source, prepare_prefix


def transcript(tmp_path, extra=None):
    path = tmp_path / "projects/source/original.jsonl"
    path.parent.mkdir(parents=True)
    entries = [
        {"type": "user", "uuid": "prompt", "sessionId": "original", "cwd": "/source",
         "message": {"role": "user", "content": "original task\u2028unchanged"}},
        {"type": "assistant", "uuid": "call", "parentUuid": "prompt", "sessionId": "original",
         "message": {"content": [{"type": "tool_use", "id": "call-1", "name": "shell", "input": {}}]}},
        {"type": "user", "uuid": "cut", "parentUuid": "call", "sessionId": "original",
         "message": {"content": [{"type": "tool_result", "tool_use_id": "call-1",
                                  "content": extra or "earlier output"}]}},
        {"type": "system", "uuid": "compact", "subtype": "compact_boundary"},
        {"type": "user", "uuid": "summary", "isCompactSummary": True,
         "message": {"content": "FUTURE_SUMMARY"}},
    ]
    path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n")
    return path, entries


def test_prefix_registers_unique_sessions_and_preserves_message_payloads(tmp_path):
    from claude_agent_sdk._internal.sessions import _sanitize_path

    original, entries = transcript(tmp_path)
    original_bytes = original.read_bytes()
    projects = tmp_path / "projects"
    source = find_prefix_source(projects, "original", "cut")
    sessions = []
    for name in ("branch-one", "branch-two"):
        cwd = tmp_path / name
        manifest = prepare_prefix(source, cwd, tmp_path / "receipts" / name, projects)
        saved = [json.loads(line) for line in Path(manifest["saved_prefix"]).read_bytes().split(b"\n") if line]
        assert [entry.get("message") for entry in saved] == [entry.get("message") for entry in entries[:3]]
        assert [entry.get("uuid") for entry in saved] == ["prompt", "call", "cut"]
        assert all(entry["sessionId"] == manifest["resume_session_id"] for entry in saved)
        assert Path(manifest["native_path"]).parent == projects / _sanitize_path(str(cwd.resolve()))
        assert Path(manifest["native_path"]).read_bytes() == Path(manifest["saved_prefix"]).read_bytes()
        assert hashlib.sha256(Path(manifest["native_path"]).read_bytes()).hexdigest() == manifest["prefix_sha256"]
        sessions.append(manifest["resume_session_id"])
    assert sessions[0] != sessions[1]
    assert original.read_bytes() == original_bytes


def test_changed_source_refuses_preparation_before_registering_a_session(tmp_path):
    original, _ = transcript(tmp_path)
    source = find_prefix_source(tmp_path / "projects", "original", "cut")
    original.write_text(original.read_text() + '{}\n')
    with pytest.raises(ValueError, match="changed"):
        prepare_prefix(source, tmp_path / "actor", tmp_path / "receipt", tmp_path / "projects")
    assert not (tmp_path / "actor").exists()


def test_missing_or_malformed_native_prefix_is_not_admitted(tmp_path):
    original, _ = transcript(tmp_path)
    assert find_prefix_source(tmp_path / "projects", "original", "absent") is None
    original.write_bytes(b"not-json\n" + original.read_bytes())
    assert find_prefix_source(tmp_path / "projects", "original", "cut") is None


def test_persisted_output_must_exist_and_is_pinned_without_rewriting_chat(tmp_path):
    output = tmp_path / "projects/source/original/tool-results/output.txt"
    original, _ = transcript(tmp_path, "<persisted-output>\nFull output saved to: %s\n</persisted-output>" % output)
    assert find_prefix_source(tmp_path / "projects", "original", "cut") is None
    output.parent.mkdir(parents=True)
    output.write_text("preserved long output")
    source = find_prefix_source(tmp_path / "projects", "original", "cut")
    manifest = prepare_prefix(source, tmp_path / "actor", tmp_path / "receipt", tmp_path / "projects")
    assert manifest["referenced_outputs"] == [{"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}]
    assert str(output) in Path(manifest["saved_prefix"]).read_text()
    assert original.exists() and output.exists()
    output.write_text("changed output")
    with pytest.raises(ValueError, match="referenced native output changed"):
        prepare_prefix(source, tmp_path / "another-actor", tmp_path / "another-receipt", tmp_path / "projects")


def test_ancestor_project_output_references_remain_available(tmp_path):
    output = tmp_path / "projects/ancestor/earlier-session/tool-results/output.txt"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"ancestor output")
    transcript(tmp_path, "<persisted-output>\nFull output saved to: %s\n</persisted-output>" % output)
    source = find_prefix_source(tmp_path / "projects", "original", "cut")
    assert source is not None
    manifest = prepare_prefix(source, tmp_path / "actor", tmp_path / "receipt", tmp_path / "projects")
    assert manifest["referenced_outputs"][0]["path"] == str(output)


def test_reference_outside_native_projects_is_rejected(tmp_path):
    output = tmp_path / "outside.txt"
    output.write_text("unrelated data")
    transcript(tmp_path, "<persisted-output>\nFull output saved to: %s\n</persisted-output>" % output)
    assert find_prefix_source(tmp_path / "projects", "original", "cut") is None
