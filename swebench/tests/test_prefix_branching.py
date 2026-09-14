import json
from pathlib import Path
from types import SimpleNamespace

from swebench import fork_eval
from swebench.tests.test_conversation_cut import journal_with_session, write_transcript
from swebench.tests.test_parent_from import write_journal


def test_reviewer_selects_pre_compaction_point_and_launcher_uses_independent_prefix(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setattr(fork_eval, "CLAUDE_PROJECTS_DIR", projects)
    parent = journal_with_session(tmp_path / "base/task/parent.jsonl", 4, "original")
    native = write_transcript(projects, "original", ["c1", "c2", "c3", "c4"])
    entries = [json.loads(line) for line in native.read_bytes().split(b"\n") if line]
    position = next(index for index, entry in enumerate(entries) if entry.get("uuid") == "r-3") + 1
    entries[position:position] = [
        {"type": "system", "uuid": "compact", "subtype": "compact_boundary"},
        {"type": "user", "uuid": "summary", "isCompactSummary": True,
         "message": {"content": "FUTURE_SUMMARY_THROUGH_STEP3"}},
    ]
    native.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    original_bytes = native.read_bytes()
    prompts, calls = [], []

    def ask(model, prompt):
        prompts.append(prompt)
        if "## Every attempt so far" in prompt:
            return json.dumps({"branches": [
                {"name": name, "base": "parent", "branch_step": 2, "hint": "Check retained work."}
                for name in ["one", "two"]]})
        return json.dumps({"failure_reason": "probe", "branch_candidates": [{"step": 2, "why": "retained work"}]})

    def run(orch, args, instance, **kwargs):
        calls.append(kwargs)
        manifest = json.loads(Path(kwargs["origin"]["conversation_prefix_manifest"]).read_text())
        assert kwargs["image"] == "snap-2" and kwargs["resume_at"] == "r-2"
        assert kwargs["resume"] == manifest["resume_session_id"] != "original"
        assert str(kwargs["cwd"]) == manifest["cwd"]
        assert "FUTURE_SUMMARY_THROUGH_STEP3" not in Path(manifest["saved_prefix"]).read_text()
        prefix = [json.loads(line) for line in Path(manifest["saved_prefix"]).read_bytes().split(b"\n") if line]
        result_ids = [block["tool_use_id"] for entry in prefix
                      for block in ((entry.get("message") or {}).get("content") or [])
                      if isinstance(block, dict) and block.get("type") == "tool_result"]
        assert result_ids == ["c1", "c2"]
        journal = write_journal(kwargs["out_dir"] / (kwargs["name"] + ".jsonl"))
        return SimpleNamespace(status="completed", error=None, checkpoints=3, journal_path=journal)

    class Bench(fork_eval.Benchmark):
        def instance(self, raw):
            return {"instance_id": raw, "repo": "r", "image": "image", "problem": "task", "f2p": [], "p2p": []}

        def grade(self, *args):
            return fork_eval.Grade(patch="patch")

        def branch_prompt(self, instance, verdict, hint, **kwargs):
            return hint

    monkeypatch.setattr(fork_eval, "ask_analyst", ask)
    monkeypatch.setattr(fork_eval, "run_attempt", run)
    args = SimpleNamespace(rounds=1, slot="claude-code", model="m", analyst_model="m",
                           analyst_tokens=1000, timeout=10, runtime_bin="runtime/ash-runtime",
                           parent_from=str(tmp_path / "base"), fork_full_conversation=False)
    fork_eval.run_one(None, args, "task", [2], tmp_path / "out", Bench())
    assert len(calls) == 2 and calls[0]["resume"] != calls[1]["resume"]
    assert "[1, 2, 3, 4]" in prompts[0]
    assert '"available_steps":[1,2,3,4]' in "".join(prompts[1].split())
    assert native.read_bytes() == original_bytes and parent.exists()


def test_prefix_launch_preserves_registered_cwd_and_disables_local_settings(tmp_path):
    seen = []

    class Orch:
        def run(self, spec):
            seen.append(spec)
            return SimpleNamespace(status="completed")

    cwd = tmp_path / "prefix-actor"
    fork_eval.run_attempt(Orch(), SimpleNamespace(slot="claude-code", model="m", timeout=10,
                                                  runtime_bin="runtime/ash-runtime"), {},
                          name="branch", prompt="hint", image="snapshot-40", out_dir=tmp_path,
                          resume="prefix-session", fork=True, resume_at="cut-40", cwd=cwd,
                          origin={"conversation_restore": "original-prefix"})
    assert seen[0].cwd == str(cwd)
    assert seen[0].resume_session_id == "prefix-session"
    assert seen[0].extra["resume_session_at"] == "cut-40"
    assert seen[0].extra["setting_sources"] == []
