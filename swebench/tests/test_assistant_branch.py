import json
from types import SimpleNamespace

import pytest

from harness.core.journal import read_journal
from harness.orchestrator.run import Orchestrator
from harness.slots.mini_history import read_entries
from harness.tests.test_assistant_turn import assistant_turn
from harness.tests.test_mini_swe import model_server, owned_filesystem, reply, pytestmark as mini_only
from runstore.tests.test_assistant_turn_branch import restore_files
from runstore.tests.test_mini_native import parent_run
from runstore.native import read_prefix
from swebench import fork_eval
from swebench.assistant_branch import ASSISTANT_REVIEW_PROMPT, reviewer_context
from swebench.branching import planned_branch


@pytest.mark.parametrize("arguments", [
    ["--slot", "claude-code"],
    ["--slot", "codex"],
    ["--slot", "mini-swe-agent", "--fork-full-conversation"],
])
@pytest.mark.parametrize("mode", ["assistant-turn", "none"])
def test_cli_rejects_incompatible_mode_before_any_run(tmp_path, arguments, mode):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as raised:
        fork_eval.main(["--branch-guidance", mode, "-o", str(output), *arguments])
    assert raised.value.code == 2 and not output.exists()


def test_assistant_prompt_and_report_schema_keep_the_synthetic_turn():
    prompt = ASSISTANT_REVIEW_PROMPT.format(problem="task", reports="[]", count_rule="At most 1.")
    assert "ordinary content" in prompt and "record the REAL observations" in prompt
    turn = assistant_turn()
    plan = {"branches": [{"name": "repair", "base": "parent", "branch_step": 3,
                          "why": "boundary", "assistant_turn": turn}]}
    assert planned_branch(plan, "r1b1-repair", 1)["assistant_turn"] == turn


@mini_only
@pytest.mark.parametrize("mode", ["assistant-turn", "none"])
@pytest.mark.parametrize("correct_first", [False, True])
def test_core_reviewer_selection_reaches_mini_without_a_user_hint(tmp_path, monkeypatch, mode, correct_first):
    memory, parent, native_points = parent_run(tmp_path / "parent", monkeypatch)
    cut = native_points[0]
    child_memory = restore_files(memory, cut.snapshot_id, tmp_path / "child-sandbox")
    wire_specs, reviewer_inputs = [], []

    def wire(self, run_spec, *args):
        wire_specs.append(run_spec)
        assert run_spec.sandbox_image == cut.snapshot_id
        return owned_filesystem(child_memory)

    monkeypatch.setattr(Orchestrator, "_wire_sandbox", wire)
    monkeypatch.setattr(fork_eval, "existing_parent", lambda *args: parent.journal_path)
    monkeypatch.setattr(fork_eval, "grade_attempt", lambda outcome, *args:
                        fork_eval.Grade(resolved=outcome.journal_path.name != "parent.jsonl"))
    turn = assistant_turn("printf reviewer > answer; cat answer") if mode == "assistant-turn" else None
    plan = {"synthesis": "Continue from the initial state.", "branches": [
        {"name": "repair", "base": "parent", "branch_step": cut.tool_depth,
         "why": "PRIVATE_REVIEWER_REASON_NOT_FOR_ACTOR",
         **({"assistant_turn": turn} if turn is not None else {})}]}

    def analyst(model, prompt):
        if "## Every attempt so far" in prompt:
            reviewer_inputs.append(prompt)
            if correct_first and len(reviewer_inputs) == 1:
                assert not wire_specs
                invalid = json.loads(json.dumps(plan))
                if mode == "assistant-turn":
                    invalid["branches"][0]["assistant_turn"]["tool_calls"][0]["function"]["arguments"] = '{"command":"bad\\escape"}'
                else:
                    invalid["branches"][0]["hint"] = "not allowed"
                return "```branch-plan\n" + json.dumps(invalid) + "\n```"
            return "```branch-plan\n" + json.dumps(plan) + "\n```"
        return json.dumps({"failure_reason": "Inspect the initial file.", "lesson": "Check it.",
                           "salvage": "file", "branch_candidates": [{"step": cut.tool_depth}]})

    monkeypatch.setattr(fork_eval, "ask_analyst", analyst)

    class Bench(fork_eval.Benchmark):
        name = "local-mini-fixture"

        def instance(self, raw):
            return {"instance_id": "fixture", "repo": "repo", "image": "unused",
                    "problem": "Write the answer.", "f2p": [], "p2p": []}

        def branch_prompt(self, *args, **kwargs):
            pytest.fail("This branch mode must never construct a user hint")

    args = SimpleNamespace(slot="mini-swe-agent", model="fixture", analyst_model="fixture",
                           rounds=1, timeout=30, analyst_tokens=10000, runtime_bin="runtime/ash-runtime",
                           parent_from="fixture", fork_full_conversation=False,
                           branch_count_mode="fixed", branch_guidance=mode)
    with model_server([reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        monkeypatch.setenv("OPENAI_BASE_URL", url)
        monkeypatch.setenv("OPENAI_API_KEY", "local-fixture")
        attempts = fork_eval.run_one(Orchestrator(), args, {}, [1], tmp_path / "run", Bench())
    assert len(attempts) == 2 and attempts[1].outcome.status == "completed", attempts[-1].outcome.error
    assert len(requests) == 2 and len(wire_specs) == 1
    assert wire_specs[0].prompt == ""
    if mode == "assistant-turn":
        assert wire_specs[0].extra["assistant_turn"] == turn
        assert requests[0]["messages"][-2] == turn
        assert json.loads(requests[0]["messages"][-1]["content"])["output"] == "reviewer"
    else:
        assert wire_specs[0].extra["resume_without_hint"] is True
        assert "assistant_turn" not in wire_specs[0].extra
        retained = [{k: v for k, v in e["message"].items() if k != "extra"}
                    for e in read_entries(read_prefix(cut.native)) if e["type"] == "mini.message"]
        assert requests[0]["messages"] == retained
        assert json.loads(requests[1]["messages"][-1]["content"])["output"] == "early\n"
        assert (child_memory.root / "answer").read_text() == "early\n"
        assert "Choose only each branch's base attempt and checkpoint" in reviewer_inputs[0]
    assert "echo late" not in json.dumps(requests)
    assert attempts[1].assistant_turn == turn and attempts[1].hint == ""
    recorded = json.loads((tmp_path / "run/plan-round1.json").read_text())
    assert "PRIVATE_REVIEWER_REASON_NOT_FOR_ACTOR" not in json.dumps(requests)
    assert recorded["branch_guidance"] == mode
    assert recorded["review"] == plan
    assert len(recorded["review_attempts"]) == (2 if correct_first else 1)
    if correct_first:
        assert recorded["review_attempts"][0]["validation_error"] in reviewer_inputs[1]
    if turn is not None:
        assert recorded["selected_branches"][0]["assistant_turn"] == turn
    else:
        assert "assistant_turn" not in recorded["selected_branches"][0]
    assert '"native_history":' in reviewer_inputs[0] and '"prefix_message_counts":' in reviewer_inputs[0]
    assert '"tools":' in reviewer_inputs[0] and '"name": "bash"' in reviewer_inputs[0]
    assert '"additionalProperties": false' in reviewer_inputs[0]
    assert '"workspace":' in reviewer_inputs[0] and '"/testbed"' in reviewer_inputs[0]
    assert "Return exactly 1 branches" in reviewer_inputs[0]
    events = read_journal(attempts[1].outcome.journal_path)
    origin = next(e for e in events if e["type"] == "fork.origin")
    assert origin["hint_delivery"] == mode
    if turn is not None:
        assert origin["assistant_turn"] == turn
    else:
        assert "assistant_turn" not in origin
        assert not any(e["type"] == "branch.assistant_turn" for e in events)
        first_response = next(i for i, e in enumerate(events) if e["type"] == "raw.mini-swe-agent")
        first_tool = next(i for i, e in enumerate(events) if e["type"] == "tool.started")
        assert first_response < first_tool


@mini_only
def test_review_context_and_selection_validate_the_exact_prefix(tmp_path, monkeypatch):
    _, parent, native_points = parent_run(tmp_path, monkeypatch)
    points = fork_eval.available_branch_points(parent.journal_path)
    context = reviewer_context(parent.journal_path, points)
    assert set(context["prefix_message_counts"]) == {"2", "3", "4"}
    early_count = context["prefix_message_counts"]["2"]
    early = context["messages"][:early_count]
    assert early[-1]["role"] == "tool" and "echo late" not in json.dumps(early)
    turn = assistant_turn()
    plan = {"branches": [{"base": "parent", "branch_step": 2, "assistant_turn": turn}]}
    kwargs = dict(limit=1, round_no=1, attempts={"parent": SimpleNamespace(outcome=parent)},
                  checkpoints={"parent": points}, guidance_mode="assistant-turn")
    choices = fork_eval.prepare_branches(plan, **kwargs)
    assert choices[0].assistant_turn == turn and choices[0].cut == native_points[0].native["cut"]
    plan["branches"][0]["hint"] = "must not leak"
    with pytest.raises(ValueError, match="must not include a user hint"):
        fork_eval.prepare_branches(plan, **kwargs)
    del plan["branches"][0]["hint"]
    turn["tool_calls"][0]["id"] = next(m["tool_calls"][0]["id"] for m in early if m.get("tool_calls"))
    with pytest.raises(ValueError, match="unique across history"):
        fork_eval.prepare_branches(plan, **kwargs)
    point = {"base": "parent", "branch_step": 2, "why": "Only a restart point."}
    kwargs["guidance_mode"] = "none"
    selected = fork_eval.prepare_branches({"branches": [point]}, **kwargs)[0]
    assert selected.hint == "" and selected.assistant_turn is None
    assert selected.cut == native_points[0].native["cut"]
    for forbidden in ["hint", "assistant_turn", "command", "content", "reasoning"]:
        with pytest.raises(ValueError, match="only point-selection fields"):
            fork_eval.prepare_branches({"branches": [{**point, forbidden: ""}]}, **kwargs)


def test_nonmini_parent_is_rejected_even_with_mini_actor(tmp_path):
    from swebench.assistant_branch import require_mini_parent

    journal = tmp_path / "claude.jsonl"
    journal.write_text(json.dumps({"type": "run.started", "slot": "claude-code"}) + "\n")
    with pytest.raises(ValueError, match="only mini-swe-agent"):
        require_mini_parent(journal)


@pytest.mark.parametrize("mode", [None, "user-hint", "assistant-turn", "none"])
def test_cli_guidance_flag_reaches_runner_and_summary(tmp_path, monkeypatch, mode):
    seen = []
    bench = SimpleNamespace(name="fixture", no_network=False, image_env=False, catalogue=lambda _: {"task": {}})
    monkeypatch.setattr(fork_eval, "select_benchmark", lambda _: bench)
    monkeypatch.setattr(fork_eval, "Orchestrator", lambda **_: None)

    def run(orch, args, raw, schedule, out_dir, benchmark):
        seen.append(args.branch_guidance)
        return []

    monkeypatch.setattr(fork_eval, "run_one", run)
    args = ["--slot", "mini-swe-agent", "--instance", "task", "--out", str(tmp_path), "--volatile-ok"]
    if mode:
        args.extend(["--branch-guidance", mode])
    assert fork_eval.main(args) == 1
    assert seen == [mode or "user-hint"]
    assert json.loads((tmp_path / "summary.json").read_text())["branch_guidance"] == (mode or "user-hint")
