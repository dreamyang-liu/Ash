"""A branch's conversation ends where its filesystem does: the fork-step cut."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swebench import fork_eval
from swebench.fork_eval import conversation_cut
from swebench.tests.test_parent_from import write_journal


def write_transcript(projects: Path, session_id: str, call_ids: list) -> Path:
    """A Claude Code transcript: per tool call, an assistant tool_use entry and a
    user tool_result entry, each with its own uuid."""
    path = projects / "-tmp" / ("%s.jsonl" % session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"type": "user", "uuid": "u-prompt", "message": {"role": "user", "content": "task"}}]
    for n, cid in enumerate(call_ids, 1):
        entries.append({"type": "assistant", "uuid": "a-%d" % n, "message": {
            "role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": "mcp__ash__shell",
                                             "input": {"command": "ls"}}]}})
        entries.append({"type": "user", "uuid": "r-%d" % n, "message": {
            "role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": "ok"}]}})
    entries.append({"type": "assistant", "uuid": "a-final", "message": {
        "role": "assistant", "content": [{"type": "text", "text": "done"}]}})
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return path


def journal_with_session(path: Path, steps: int, session_id: str) -> Path:
    write_journal(path, steps=steps)
    records = [json.loads(l) for l in path.read_text().splitlines()]
    for record in records:
        if record.get("type") == "checkpoint.captured":
            record["session_ckpt"] = session_id
    records.insert(1, {"v": 2, "type": "session.ref", "ts": "2026-09-04T00:00:00Z", "seq": 0,
                       "run_id": "parent", "native_session_id": session_id})
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_cut_is_the_tool_result_entry_of_the_fork_step(tmp_path, monkeypatch):
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", tmp_path / "projects")
    journal = journal_with_session(tmp_path / "parent.jsonl", steps=4, session_id="sess-1")
    calls = [json.loads(l)["call_id"] for l in journal.read_text().splitlines()
             if '"tool.started"' in l]
    write_transcript(tmp_path / "projects", "sess-1", calls)
    assert conversation_cut(journal, 2) == "r-2"      # after step 2's result
    assert conversation_cut(journal, 4) == "r-4"      # the last step: nothing after it
    assert conversation_cut(journal, 5) is None       # no such step
    assert conversation_cut(journal, 0) is None


def test_cut_is_none_when_the_transcript_is_missing_or_disagrees(tmp_path, monkeypatch):
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", tmp_path / "projects")
    journal = journal_with_session(tmp_path / "parent.jsonl", steps=3, session_id="sess-2")
    assert conversation_cut(journal, 1) is None                       # no transcript on disk
    write_transcript(tmp_path / "projects", "sess-2", ["other-1", "other-2", "other-3"])
    assert conversation_cut(journal, 1) is None                       # ids do not match


def test_cut_can_use_the_checkpoint_session_instead_of_the_last_session(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", steps=1, session_id="latest")
    write_transcript(projects, "paired", ["c1"])
    assert conversation_cut(journal, 1) is None
    assert conversation_cut(journal, 1, "paired") == "r-1"


@pytest.mark.parametrize("combined", [False, True])
def test_parallel_native_calls_cannot_be_cut_with_unresolved_or_future_results(tmp_path, monkeypatch, combined):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "parallel")
    path = write_transcript(projects, "parallel", ["c1", "c2"])
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    # Both tool uses belong to the same assistant message.
    entries[1]["message"]["content"] += entries[3]["message"]["content"]
    del entries[3]
    if combined:
        entries[2]["message"]["content"] += entries[3]["message"]["content"]
        del entries[3]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    assert conversation_cut(journal, 1) is None
    assert conversation_cut(journal, 2) == ("r-1" if combined else "r-2")


def test_explicit_checkpoint_call_id_must_agree_with_step(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "wrong")
    write_transcript(projects, "wrong", ["c1", "c2"])
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    for record in records:
        if record.get("type") == "checkpoint.captured" and record.get("step") == 1:
            record.update(call_id="c2", pairing="call-id-v1", prefix_complete=True)
    journal.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert conversation_cut(journal, 1) is None


def test_one_response_split_around_tool_result_is_still_one_turn(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "split")
    path = write_transcript(projects, "split", ["c1", "c2"])
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    # A, result A, B, result B: an empty pending set after A is NOT sufficient.
    entries[1]["message"]["id"] = entries[3]["message"]["id"] = "one-response"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    assert conversation_cut(journal, 1) is None
    assert conversation_cut(journal, 2) == "r-2"


def test_new_policy_preserves_tool_storage_but_only_offers_closed_turns(tmp_path, monkeypatch):
    from harness.core.journal import JournalWriter, read_journal
    from harness.normalize.claude_turns import ModelTurnTracker
    from harness.rollback import branch_checkpoints, turn_branch_checkpoints
    from harness.tests.test_model_turns import assistant, result, feed
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    path = tmp_path / "parent.jsonl"
    with JournalWriter(path) as journal:
        journal.emit("checkpoint.policy", pairing="call-id-v1")
        tracker = ModelTurnTracker(journal)
        feed(journal, tracker, assistant("t1", "c1", "c2"))
        for step in (1, 2):
            journal.emit("checkpoint.captured", step=step, call_id="c%d" % step,
                         snapshot_id="s%d" % step, session_ckpt="sess", reason="captured",
                         pairing="call-id-v1", prefix_complete=True)
            feed(journal, tracker, result("c%d" % step))
        assert list(branch_checkpoints(path)) == [1, 2]
        assert turn_branch_checkpoints(path) == {}
        feed(journal, tracker, assistant("t2"))
    write_transcript(projects, "sess", ["c1", "c2"])
    assert list(turn_branch_checkpoints(path)) == [2]
    assert conversation_cut(path, 1, "sess") is None
    assert list(fork_eval.available_branch_points(path)) == [2]
    # A turn-complete message cannot bless an uncertain execution snapshot.
    records = read_journal(path)
    for e in records:
        if e.get("type") == "checkpoint.captured" and e["step"] == 2:
            e.update(reason="execution_uncertain", prefix_complete=False)
    path.write_text("\n".join(json.dumps(e) for e in records) + "\n")
    assert not fork_eval.available_branch_points(path)


def test_a_cut_before_a_compaction_boundary_is_not_loadable(tmp_path, monkeypatch):
    """Pre-compaction cuts without explicit preservation evidence stay rejected."""
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", tmp_path / "projects")
    journal = journal_with_session(tmp_path / "parent.jsonl", steps=4, session_id="sess-c")
    calls = [json.loads(l)["call_id"] for l in journal.read_text().splitlines() if '"tool.started"' in l]
    path = write_transcript(tmp_path / "projects", "sess-c", calls)
    rows = path.read_text().splitlines()
    # boundary after step 2's result: steps 1-2 are unloadable, 3-4 are fine
    idx = next(i for i, l in enumerate(rows) if '"r-2"' in l) + 1
    rows.insert(idx, json.dumps({"type": "system", "subtype": "compact_boundary",
                                 "uuid": "cb", "content": "Conversation compacted"}))
    rows.insert(idx + 1, json.dumps({"type": "user", "uuid": "cs", "isCompactSummary": True,
                                     "message": {"role": "user", "content": "summary"}}))
    path.write_text("\n".join(rows) + "\n")
    assert conversation_cut(journal, 1) is None
    assert conversation_cut(journal, 2) is None
    assert conversation_cut(journal, 3) == "r-3"
    assert conversation_cut(journal, 4) == "r-4"


def compact_transcript(path, preserved, *, before=None, second_compaction=False):
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    final = entries.pop()
    entries.extend(before or [])
    boundary = {"type": "system", "subtype": "compact_boundary", "uuid": "compact",
                "compactMetadata": {"preservedMessages": {"uuids": preserved}}}
    entries.extend([boundary, {"type": "user", "uuid": "summary", "isCompactSummary": True,
                              "message": {"role": "user", "content": "compressed prefix"}}, final])
    if second_compaction:
        entries.append({**boundary, "uuid": "compact-again"})
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")


def test_only_preserved_frontier_cut_survives_compaction(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "preserved")
    transcript = write_transcript(projects, "preserved", ["c1", "c2"])
    compact_transcript(transcript, ["a-1", "r-1", "a-2", "r-2"],
                       before=[{"type": "attachment", "uuid": "bookkeeping",
                                "attachment": {"type": "total_tokens_reminder"}}])
    assert conversation_cut(journal, 1) is None
    assert conversation_cut(journal, 2) == "r-2"
    assert list(fork_eval.available_branch_points(journal)) == [1, 2]
    assert fork_eval.conversation_restore(journal, 1, "preserved")[1] is not None


@pytest.mark.parametrize("preserved", [None, "r-2", [], ["r-1"], ["r-2", None]])
def test_missing_or_malformed_preserved_cut_is_rejected(tmp_path, monkeypatch, preserved):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "missing-preserved")
    transcript = write_transcript(projects, "missing-preserved", ["c1", "c2"])
    compact_transcript(transcript, preserved)
    assert conversation_cut(journal, 2) is None


@pytest.mark.parametrize("role,content", [
    ("assistant", [{"type": "text", "text": "later discovery"}]),
    ("assistant", [{"type": "thinking", "thinking": "later reasoning"}]),
    ("user", "later instruction"),
    ("assistant", [{"type": "tool_use", "id": "later-call", "name": "shell", "input": {}}]),
])
def test_preserved_cut_cannot_inherit_future_content_via_summary(tmp_path, monkeypatch, role, content):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "future-summary")
    transcript = write_transcript(projects, "future-summary", ["c1", "c2"])
    compact_transcript(transcript, ["r-2"], before=[{
        "type": role, "uuid": "future", "message": {"role": role, "content": content}}])
    assert conversation_cut(journal, 2) is None


def test_repeated_compaction_does_not_implicitly_certify_old_cut(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "repeated-compact")
    transcript = write_transcript(projects, "repeated-compact", ["c1", "c2"])
    compact_transcript(transcript, ["r-2"], second_compaction=True)
    assert conversation_cut(journal, 2) is None


@pytest.mark.parametrize("attachment", [
    {"type": "file", "content": "later file observation"}, {}, "unknown", ["unknown"],
])
def test_unproven_post_cut_attachments_are_not_absorbed_into_summary(tmp_path, monkeypatch, attachment):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    journal = journal_with_session(tmp_path / "parent.jsonl", 2, "attachment-summary")
    transcript = write_transcript(projects, "attachment-summary", ["c1", "c2"])
    compact_transcript(transcript, ["r-2"], before=[{
        "type": "attachment", "uuid": "future-attachment", "attachment": attachment}])
    assert conversation_cut(journal, 2) is None


@pytest.mark.parametrize("full_conversation", [False, True])
def test_missing_cut_requires_explicit_full_conversation_mode(tmp_path, monkeypatch, full_conversation):
    """Default mode refuses an unavailable cut; full mode must be explicit."""
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", tmp_path / "projects")
    seen = []

    def fake_run_attempt(orch, args, instance, **kw):
        seen.append(kw)
        jp = tmp_path / "out" / "t" / ("%s.jsonl" % kw["name"])
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text("{}\n")
        return SimpleNamespace(status="completed", error=None, checkpoints=0, journal_path=jp)
    monkeypatch.setattr(fork_eval, "run_attempt", fake_run_attempt)
    monkeypatch.setattr(fork_eval, "ask_analyst", lambda model, prompt, **k: json.dumps(
        {"failure_reason": "x", "lesson": "y", "salvage": "z",
         "branch_candidates": [{"step": 2, "why": "w"}],
         "synthesis": "s",
         "branches": [{"name": "b", "base": "parent", "branch_step": 2, "hint": "h"}]}))
    journal_with_session(tmp_path / "base" / "shard-0" / "t" / "parent.jsonl", 3, "sess-3")

    class Bench(fork_eval.Benchmark):
        name = "fake"
        def instance(self, raw):
            return {"instance_id": raw, "repo": "r", "image": "img", "problem": "p", "f2p": [], "p2p": []}
        def prompt(self, instance): return "p"
        def branch_prompt(self, instance, verdict, hint, **ctx):
            return "TRUNCATED" if ctx.get("truncated") else "FULL"
        def grade(self, snapshot_id, instance, backend): return fork_eval.Grade(patch="d")

    args = SimpleNamespace(rounds=1, slot="claude-code", model="m", analyst_model="m",
                           analyst_tokens=1000, timeout=10.0, runtime_bin="runtime/ash-runtime",
                           parent_from=str(tmp_path / "base"), fork_full_conversation=full_conversation)
    fork_eval.run_one(None, args, "t", [1], tmp_path / "out" / "t", Bench())
    if not full_conversation:
        assert not seen
        plan = json.loads((tmp_path / "out/t/plan-round1.json").read_text())
        assert "no native conversation cut" in plan["validation_error"]
        return
    branch = [k for k in seen if k["name"] != "parent"][0]
    assert branch["resume_at"] is None and branch["prompt"] == "FULL"
    assert branch["origin"]["cut_note"] == "explicit-full-conversation"


def test_a_cut_the_cli_refuses_is_reported_without_untruncated_retry(tmp_path, monkeypatch):
    """A refused cut must not silently change the selected conversation state."""
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", tmp_path / "projects")
    journal = journal_with_session(tmp_path / "base" / "shard-0" / "t" / "parent.jsonl", 3, "sess-r")
    calls = [json.loads(l)["call_id"] for l in journal.read_text().splitlines() if '"tool.started"' in l]
    write_transcript(tmp_path / "projects", "sess-r", calls)
    seen = []

    def fake_run_attempt(orch, args, instance, **kw):
        seen.append(kw)
        jp = tmp_path / "out" / "t" / ("%s.jsonl" % kw["name"])
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text("{}\n")
        if kw.get("resume_at"):
            return SimpleNamespace(status="error", checkpoints=0, journal_path=jp,
                                   error="ResultError: No message found with message.uuid of: r-2")
        return SimpleNamespace(status="completed", error=None, checkpoints=0, journal_path=jp)
    monkeypatch.setattr(fork_eval, "run_attempt", fake_run_attempt)
    monkeypatch.setattr(fork_eval, "ask_analyst", lambda model, prompt, **k: json.dumps(
        {"failure_reason": "x", "lesson": "y", "salvage": "z",
         "branch_candidates": [{"step": 2, "why": "w"}],
         "synthesis": "s",
         "branches": [{"name": "b", "base": "parent", "branch_step": 2, "hint": "h"}]}))

    class Bench(fork_eval.Benchmark):
        name = "fake"
        def instance(self, raw):
            return {"instance_id": raw, "repo": "r", "image": "img", "problem": "p", "f2p": [], "p2p": []}
        def prompt(self, instance): return "p"
        def branch_prompt(self, instance, verdict, hint, **ctx):
            return "TRUNCATED" if ctx.get("truncated") else "FULL"
        def grade(self, snapshot_id, instance, backend): return fork_eval.Grade(patch="d")

    args = SimpleNamespace(rounds=1, slot="claude-code", model="m", analyst_model="m",
                           analyst_tokens=1000, timeout=10.0, runtime_bin="runtime/ash-runtime",
                           parent_from=str(tmp_path / "base"), fork_full_conversation=False)
    attempts = fork_eval.run_one(None, args, "t", [1], tmp_path / "out" / "t", Bench())
    branches = [k for k in seen if k["name"] != "parent"]
    assert [b["resume_at"] for b in branches] == ["r-2"]
    assert branches[0]["prompt"] == "TRUNCATED"
    assert attempts[-1].outcome.status == "error"
    assert "no full-conversation retry" in attempts[-1].grade.error
    assert (tmp_path / "out/t/r1b1-b.jsonl").exists()


def test_run_attempt_passes_the_cut_to_the_slot(tmp_path):
    seen = {}

    class Orch:
        def run(self, spec):
            seen["spec"] = spec
            return SimpleNamespace(status="completed", journal_path=None, checkpoints=0, error=None)

    fork_eval.run_attempt(Orch(), SimpleNamespace(slot="claude-code", model="m", timeout=1.0,
                                                  runtime_bin="runtime/ash-runtime"),
                          {"instance_id": "t"}, name="r1b1", prompt="p", image="snap",
                          out_dir=tmp_path, resume="sess", fork=True, resume_at="r-7")
    spec = seen["spec"]
    assert spec.resume_session_id == "sess" and spec.fork is True
    assert spec.extra["resume_session_at"] == "r-7"
