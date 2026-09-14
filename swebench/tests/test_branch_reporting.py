import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from scripts import deepswe_export_resolved as exporter
from scripts.deepswe_branch_report import fork_positions
from scripts.trajectory_view import fork_metadata, lineage_journals, load, render_with_ancestry


@pytest.mark.parametrize("name", ["trajectory_view.py", "deepswe_export_resolved.py", "deepswe_branch_report.py"])
def test_reporting_scripts_still_run_without_pythonpath(tmp_path, name):
    script = Path(__file__).resolve().parents[2] / "scripts" / name
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def journal(path, steps, origin=None):
    events = []
    if origin:
        events.append({"type": "fork.origin", **origin})
    events.append({"type": "run.started", "task_prompt": "PROMPT_" + path.stem})
    for step in range(1, steps + 1):
        call_id = "%s-%d" % (path.stem, step)
        events += [
            {"type": "tool.started", "name": "shell", "call_id": call_id,
             "args": {"command": "ACTION_%s_%d" % (path.stem, step)}},
            {"type": "tool.finished", "call_id": call_id, "output": "RESULT_" + call_id},
            {"type": "agent.message", "text": "AFTER_%s_%d" % (path.stem, step)},
        ]
    events.append({"type": "run.finished", "status": "completed", "usage": {}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
    return path


def family(directory):
    parent = journal(directory / "parent.jsonl", 3)
    first = journal(directory / "r1b1-first.jsonl", 3,
                    {"parent_run_id": "parent", "branch_step": 1, "actor_hint": "FIRST_HINT"})
    winner = journal(directory / "r2b1-winner.jsonl", 2,
                     {"parent_run_id": first.stem, "branch_step": 2, "actor_hint": "WINNER_HINT"})
    return parent, first, winner


def test_lineage_follows_the_actual_base_and_stops_at_the_tool_result(tmp_path):
    parent, first, winner = family(tmp_path)
    assert lineage_journals(winner) == [parent, first, winner]
    text = "\n".join(render_with_ancestry(winner, max_output=1000, full=True))
    assert "RESULT_parent-1" in text and "RESULT_r1b1-first-2" in text
    assert "AFTER_parent_1" not in text and "ACTION_parent_2" not in text
    assert "AFTER_r1b1-first_2" not in text and "ACTION_r1b1-first_3" not in text
    assert "ACTION_r2b1-winner_1" in text and "### step 4" in text


def test_origin_overrides_planned_step_and_hint(tmp_path):
    path = journal(tmp_path / "r1b1-first.jsonl", 1,
                   {"parent_run_id": "other", "branch_step": 2, "actor_hint": "ACTUAL",
                    "branch_count_mode": "fixed"})
    (tmp_path / "plan-round1.json").write_text(json.dumps({"review": {
        "branches": [{"name": "first", "base": "parent", "branch_step": 5, "hint": "PLANNED"}]}}))
    info = fork_metadata(path)
    assert info["base"] == "other" and info["branch_step"] == 2 and info["hint"] == "ACTUAL"
    assert info["branch_count_mode"] == "fixed"


@pytest.mark.parametrize("independent", [True, False])
def test_plan_fallback_reads_both_schema_versions(tmp_path, independent):
    path = journal(tmp_path / "r1b1-first.jsonl", 1)
    branch = {"name": "first", "hint": "H"}
    review = {"branches": [branch]}
    (branch if independent else review).update(base="parent", branch_step=2)
    (tmp_path / "plan-round1.json").write_text(json.dumps({"review": review}))
    assert fork_metadata(path)["branch_step"] == 2


def test_full_conversation_view_is_labelled_and_keeps_parent_suffix(tmp_path):
    journal(tmp_path / "parent.jsonl", 3)
    child = journal(tmp_path / "r1b1-full.jsonl", 1,
                    {"parent_run_id": "parent", "branch_step": 1,
                     "cut_note": "explicit-full-conversation"})
    text = "\n".join(render_with_ancestry(child, max_output=1000, full=True))
    assert "ACTION_parent_3" in text and "Full-conversation resume" in text


def test_loader_keeps_unicode_line_separators_inside_a_json_string(tmp_path):
    path = tmp_path / "unicode.jsonl"
    path.write_text(json.dumps({"type": "agent.message", "text": "a\u2028b"}, ensure_ascii=False) + "\n")
    assert load(path)[0]["text"] == "a\u2028b"


def test_fork_position_report_does_not_mix_rounds_with_individual_branches():
    attempts = [{"name": "r1b1-a", "resolved": False}, {"name": "r1b2-b", "resolved": True}]
    old = {"reports": {"parent": {"steps": 100}},
           "review": {"base": "parent", "branch_step": 50,
                      "branches": [{"name": "a"}, {"name": "b"}]}}
    new = {"reports": old["reports"], "branch_policy": "adaptive-per-branch",
           "review": {"branches": [{"name": "a", "base": "parent", "branch_step": 25},
                                   {"name": "b", "base": "parent", "branch_step": 75},
                                   {"name": "unstarted", "base": "parent", "branch_step": 90}]}}
    assert fork_positions(old, attempts, 1) == ("rounds", [(0.5, True)])
    assert fork_positions(new, attempts, 1) == ("branches", [(0.25, False), (0.75, True)])


@pytest.mark.parametrize("policy", ["adaptive-per-branch", "per-branch"])
def test_rejected_raw_plan_is_not_interpreted_as_executed_branches(tmp_path, policy):
    plan = {"branch_policy": policy, "review": {"branches": [42]},
            "validation_error": "branch 1 is not an object"}
    assert fork_positions(plan, [], 1) == ("branches", [])
    path = journal(tmp_path / "r1b1-first.jsonl", 1,
                   {"parent_run_id": "parent", "branch_step": 1, "actor_hint": "ACTUAL"})
    (tmp_path / "plan-round1.json").write_text(json.dumps(plan))
    assert fork_metadata(path)["hint"] == "ACTUAL"


def test_export_keeps_the_full_ancestry_and_correct_selected_point(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    parent, first, winner = family(root / "shard-0/task")
    summary = {"instances": [{"instance": "task", "attempts": [
        {"name": parent.stem, "resolved": False, "journal": str(parent)},
        {"name": first.stem, "resolved": False, "journal": str(first)},
        {"name": winner.stem, "resolved": True, "journal": str(winner)}]}]}
    (root / "shard-0/summary.json").write_text(json.dumps(summary))
    single = tmp_path / "single.json"
    single.write_text(json.dumps({"tasks": []}))
    archive = tmp_path / "export.tar.gz"
    monkeypatch.setattr(sys, "argv", ["export", "--single", str(single), "--branch", str(root),
                                      "--no-atif", "-o", str(archive)])
    assert exporter.main() == 0
    with tarfile.open(archive) as tar:
        assert "task/r1b1-first.jsonl" in tar.getnames()
        manifest = json.load(tar.extractfile("task/manifest.json"))
        assert manifest["fork"]["base"] == first.stem
        assert manifest["fork"]["branch_step"] == 2
        text = tar.extractfile("task/TRAJECTORY.md").read().decode()
        assert "AFTER_parent_1" not in text and "AFTER_r1b1-first_2" not in text
        assert "ACTION_r2b1-winner_1" in text
