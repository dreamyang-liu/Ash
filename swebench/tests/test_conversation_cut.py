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


def test_a_cut_before_a_compaction_boundary_is_not_loadable(tmp_path, monkeypatch):
    """Claude Code resumes only entries after `compact_boundary`; a uuid before it
    is rejected ("No message found"). 11 of 82 DeepSWE parents were compacted."""
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


def test_branch_falls_back_to_the_full_conversation_only_when_no_cut_exists(tmp_path, monkeypatch):
    """A compacted parent cannot be cut before its boundary: the branch then
    runs with the full conversation and SAYS SO in its origin (cut_note)."""
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
         "base": "parent", "branch_step": 2, "why": "w", "synthesis": "s",
         "branches": [{"name": "b", "hint": "h"}]}))
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
                           parent_from=str(tmp_path / "base"), fork_full_conversation=False)
    fork_eval.run_one(None, args, "t", [1], tmp_path / "out" / "t", Bench())
    branch = [k for k in seen if k["name"] != "parent"][0]
    assert branch["resume_at"] is None and branch["prompt"] == "FULL"
    assert branch["origin"]["cut_note"] == "compacted-before-fork"


def test_a_cut_the_cli_refuses_is_retried_with_the_full_conversation(tmp_path, monkeypatch):
    """Belt and braces for the compaction case the transcript scan missed: if
    Claude Code answers "No message found with message.uuid", the branch is
    re-run without the cut, the refused start's journal is kept under another
    name, and the origin records cut-refused-by-cli."""
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
         "base": "parent", "branch_step": 2, "why": "w", "synthesis": "s",
         "branches": [{"name": "b", "hint": "h"}]}))

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
    fork_eval.run_one(None, args, "t", [1], tmp_path / "out" / "t", Bench())
    branches = [k for k in seen if k["name"] != "parent"]
    assert [b["resume_at"] for b in branches] == ["r-2", None]
    assert [b["prompt"] for b in branches] == ["TRUNCATED", "FULL"]
    assert branches[1]["origin"]["cut_note"] == "cut-refused-by-cli"
    assert list((tmp_path / "out" / "t").glob("*.cut-refused.jsonl")), "the refused start is kept"


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
