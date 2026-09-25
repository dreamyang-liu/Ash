import json
from types import SimpleNamespace

import pytest

from swebench import fork_eval
from swebench.branching import branch_count_rule, branch_run_name, planned_branch, review_branches
from swebench.tests.test_parent_from import write_journal


def direction(step, name="lead", base="parent", hint="  Check the boundary.  "):
    return {"name": name, "base": base, "branch_step": step,
            "why": "Relevant work is present.", "hint": hint}


def exercise(tmp_path, monkeypatch, plans, *, limits=None, alter_parent=None, count_mode="adaptive"):
    parent = write_journal(tmp_path / "source/task/parent.jsonl", steps=5)
    if alter_parent:
        records = [json.loads(line) for line in parent.read_text().splitlines()]
        parent.write_text("\n".join(json.dumps(e) for e in alter_parent(records)) + "\n")
    calls, analyst_inputs, reviewer_inputs = [], [], []

    def ask(model, prompt):
        if "## Every attempt so far" in prompt:
            index = len(reviewer_inputs)
            reviewer_inputs.append(prompt)
            return json.dumps(plans[index])
        analyst_inputs.append(prompt)
        return json.dumps({"failure_reason": "A boundary remains.", "lesson": "Inspect it.",
                           "salvage": "Existing implementation.",
                           "branch_candidates": [{"step": 1, "why": "Useful state."}]})

    def run(orch, args, instance, **kwargs):
        calls.append(kwargs)
        path = write_journal(kwargs["out_dir"] / (kwargs["name"] + ".jsonl"))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for event in records:
            if event.get("type") == "checkpoint.captured":
                event["snapshot_id"] = kwargs["name"] + "-" + event["snapshot_id"]
        path.write_text("\n".join(json.dumps(e) for e in records) + "\n")
        return SimpleNamespace(status="completed", error=None, checkpoints=3, journal_path=path)

    class Bench(fork_eval.Benchmark):
        name = "adaptive-test"

        def instance(self, raw):
            return {"instance_id": raw, "repo": "repo", "image": "image",
                    "problem": "Fix the public task.", "f2p": [], "p2p": []}

        def grade(self, snapshot_id, instance, backend):
            return fork_eval.Grade(patch="change")

        def branch_prompt(self, instance, verdict, hint, **context):
            return hint

    monkeypatch.setattr(fork_eval, "ask_analyst", ask)
    monkeypatch.setattr(fork_eval, "run_attempt", run)
    monkeypatch.setattr(fork_eval, "conversation_cut",
                        lambda path, step, session=None: "%s-cut-%d" % (path.stem, step))
    args = SimpleNamespace(rounds=len(plans), slot="claude-code", model="model",
                           analyst_model="model", analyst_tokens=1000, timeout=1,
                           runtime_bin="runtime/ash-runtime", parent_from=str(tmp_path / "source"),
                           fork_full_conversation=False, branch_count_mode=count_mode,
                           reviewer_max_attempts=1)
    attempts = fork_eval.run_one(None, args, "task", limits or [4], tmp_path / "out", Bench())
    return attempts, calls, analyst_inputs, reviewer_inputs


@pytest.mark.parametrize("steps", [[2], [2, 2], [1, 2, 3, 4]])
def test_reviewer_may_choose_fewer_repeated_or_distinct_points(tmp_path, monkeypatch, steps):
    branches = [direction(step, name="lead%d" % i, hint="  Hint %d.  " % i)
                for i, step in enumerate(steps)]
    attempts, calls, analyst, reviewer = exercise(
        tmp_path, monkeypatch, [{"synthesis": "Useful choices.", "branches": branches}])
    assert len(calls) == len(steps)
    assert [c["image"] for c in calls] == ["snap-%d" % step for step in steps]
    assert [c["resume_at"] for c in calls] == ["parent-cut-%d" % step for step in steps]
    assert [a.hint for a in attempts[1:]] == [b["hint"] for b in branches]
    assert len(analyst) == len(reviewer) == 1
    assert '"available_steps": [' in reviewer[0]
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert saved["branch_limit"] == 4 and saved["review"]["branches"] == branches
    assert saved["branch_count_mode"] == "adaptive"
    assert len(saved["selected_branches"]) == len(steps)


def test_branches_can_choose_different_base_attempts(tmp_path, monkeypatch):
    child = branch_run_name(1, 1, "first")
    plans = [{"synthesis": "Start.", "branches": [direction(1, "first")]},
             {"synthesis": "Two bases.", "branches": [direction(2), direction(3, base=child)]}]
    _, calls, _, _ = exercise(tmp_path, monkeypatch, plans, limits=[4, 3])
    assert len(calls) == 3
    assert calls[1]["image"] == "snap-2"
    assert calls[2]["image"] == child + "-snap-3"
    assert calls[2]["origin"]["parent_run_id"] == child
    assert calls[2]["resume_at"] == child + "-cut-3"


def test_analyst_and_reviewer_only_receive_complete_turn_steps(tmp_path, monkeypatch):
    def alter(records):
        records.insert(1, {"type": "branch.boundary.policy", "policy": "completed-model-turn-v1"})
        for step, ids in ((2, ["c1", "c2"]), (5, ["c3", "c4", "c5"])):
            records += [{"type": "model.turn.output_completed", "turn_id": str(step), "call_ids": ids},
                        {"type": "model.turn.completed", "turn_id": str(step), "call_ids": ids, "step": step}]
        return records
    _, calls, analysts, reviewers = exercise(tmp_path, monkeypatch,
        [{"branches": [direction(2)]}], alter_parent=alter)
    plan = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert plan["available_steps"]["parent"] == [2, 5]
    assert "[2, 5]" in analysts[0]
    assert len(reviewers) == len(calls) == 1
    assert calls[0]["image"] == "snap-2"


@pytest.mark.parametrize("branches,reason", [
    ([direction(1)] * 5, "above limit"),
    ([direction(1), direction(9)], "no eligible checkpoint"),
    ([direction(1), direction(2, base="unknown")], "unknown base"),
    ([direction(True)], "no eligible checkpoint"),
    ([direction(1, hint=" ")], "non-empty hint"),
])
def test_invalid_plan_is_rejected_before_any_branch_launch(tmp_path, monkeypatch, branches, reason):
    _, calls, _, _ = exercise(tmp_path, monkeypatch, [{"branches": branches}])
    assert not calls
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert reason in saved["validation_error"]


def test_empty_selection_stops_without_padding_or_reasking(tmp_path, monkeypatch):
    _, calls, _, reviews = exercise(tmp_path, monkeypatch,
                                    [{"synthesis": "No useful lead.", "branches": []},
                                     {"branches": [direction(1)]}])
    assert not calls and len(reviews) == 1


@pytest.mark.parametrize("missing", [False, True])
def test_failed_or_missing_snapshot_is_not_offered_or_silently_replaced(tmp_path, monkeypatch, missing):
    def alter(records):
        out = []
        for e in records:
            if e.get("type") == "checkpoint.captured" and e["step"] == 2:
                if missing:
                    continue
                e.update(reason="failed", snapshot_id="snap-1", captured=False)
            out.append(e)
        return out

    _, calls, analyst, reviews = exercise(tmp_path, monkeypatch,
                                         [{"branches": [direction(2)]}], alter_parent=alter)
    assert not calls
    assert "[1, 3, 4, 5]" in analyst[0]
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert saved["available_steps"]["parent"] == [1, 3, 4, 5]
    assert "no eligible checkpoint" in saved["validation_error"]


def test_readers_support_historical_shared_and_new_independent_points():
    old = {"base": "parent", "branch_step": 4, "why": "old",
           "branches": [{"name": "some_name", "hint": "H"}]}
    new = {"branches": [direction(2, "some_name"), direction(3, base="other")]}
    assert review_branches(old)[0]["branch_step"] == 4
    assert [b["branch_step"] for b in review_branches(new)] == [2, 3]
    assert planned_branch(new, "r1b1-some-name", 1)["base"] == "parent"
    assert planned_branch(old, "r1b1-some-name", 1)["hint"] == "H"


@pytest.mark.parametrize("count_mode", ["adaptive", "fixed"])
def test_reviewer_count_instructions_match_the_selected_mode(count_mode):
    prompt = fork_eval._REVIEW_PROMPT.format(
        problem="task", reports="[]", count_rule=branch_count_rule(count_mode, 4))
    assert "Positions do NOT need to be distinct" in prompt
    if count_mode == "fixed":
        assert "Return exactly 4 branches" in prompt
        assert "Return at most" not in prompt and "upper bound" not in prompt
        assert "return an empty branches list" not in prompt
    else:
        assert "Return at most 4 branches" in prompt
        assert "upper bound, not a quota" in prompt
        assert "Return exactly" not in prompt


@pytest.mark.parametrize("steps", [[2, 2, 2, 2], [1, 2, 3, 4]])
def test_fixed_mode_runs_exact_count_with_repeated_or_distinct_points(tmp_path, monkeypatch, steps):
    plan = {"branches": [direction(s, name="lead%d" % i) for i, s in enumerate(steps)]}
    _, calls, _, reviews = exercise(tmp_path, monkeypatch, [plan], count_mode="fixed")
    assert len(calls) == 4
    assert [c["origin"]["branch_step"] for c in calls] == steps
    assert all(c["origin"]["branch_count_mode"] == "fixed" for c in calls)
    assert "Return exactly 4 branches" in reviews[0]
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert saved["branch_count_mode"] == "fixed" and saved["branch_policy"] == "per-branch"


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5])
def test_fixed_mode_rejects_wrong_count_without_padding_or_extra_review(tmp_path, monkeypatch, count):
    plan = {"branches": [direction(2, name="lead%d" % i) for i in range(count)]}
    _, calls, _, reviews = exercise(tmp_path, monkeypatch, [plan], count_mode="fixed")
    assert not calls and len(reviews) == 1
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert "requires exactly 4 branches" in saved["validation_error"]
    assert len(saved["review"]["branches"]) == count


def test_fixed_count_follows_each_round_schedule(tmp_path, monkeypatch):
    plans = [{"branches": [direction(2, name="lead%d" % i) for i in range(n)]} for n in [4, 3]]
    _, calls, _, reviews = exercise(tmp_path, monkeypatch, plans, limits=[4, 3], count_mode="fixed")
    assert len(calls) == 7
    assert [c["origin"]["round"] for c in calls] == [1] * 4 + [2] * 3
    assert "Return exactly 4 branches" in reviews[0]
    assert "Return exactly 3 branches" in reviews[1]


def test_fixed_count_does_not_bypass_checkpoint_validation(tmp_path, monkeypatch):
    plan = {"branches": [direction(s, name="lead%d" % i) for i, s in enumerate([1, 2, 3, 9])]}
    _, calls, _, _ = exercise(tmp_path, monkeypatch, [plan], count_mode="fixed")
    assert not calls
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert "no eligible checkpoint" in saved["validation_error"]


@pytest.mark.parametrize("requested_mode", [None, "adaptive", "fixed"])
def test_cli_mode_reaches_runner_and_summary(tmp_path, monkeypatch, requested_mode):
    seen = {}
    bench = SimpleNamespace(name="fake", no_network=False, image_env=False, catalogue=lambda args: {"task": {}})
    monkeypatch.setattr(fork_eval, "select_benchmark", lambda args: bench)
    monkeypatch.setattr(fork_eval, "Orchestrator", lambda **kwargs: None)

    def run(orch, args, raw, schedule, out_dir, benchmark):
        seen.update(mode=args.branch_count_mode, schedule=schedule)
        return []

    monkeypatch.setattr(fork_eval, "run_one", run)
    argv = ["--instance", "task", "--branches", "4,3", "--out", str(tmp_path), "--volatile-ok"]
    if requested_mode:
        argv += ["--branch-count-mode", requested_mode]
    assert fork_eval.main(argv) == 1
    mode = requested_mode or "adaptive"
    assert seen == {"mode": mode, "schedule": [4, 3]}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["sandbox_ttl"] == 2400  # default1800s actor budget +600s margin
    assert summary["branch_count_mode"] == mode
    assert summary["branch_schedule_semantics"] == ("exact_counts" if mode == "fixed" else "upper_bounds")


def test_invalid_count_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown branch count mode"):
        branch_count_rule("unknown", 4)
    with pytest.raises(ValueError, match="unknown branch count mode"):
        fork_eval.prepare_branches({"branches": []}, limit=4, round_no=1,
                                   attempts={}, checkpoints={}, count_mode="unknown")
