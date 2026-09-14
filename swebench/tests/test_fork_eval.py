"""What the analyst actually gets to see."""
import json
from pathlib import Path
from string import Formatter
from types import SimpleNamespace

import pytest

from swebench import fork_eval
from swebench.fork_eval import RESULT_CHARS, _clip, render_transcript


def _journal(tmp_path, records):
    path = tmp_path / "j.jsonl"
    with open(path, "w") as fh:
        for i, r in enumerate(records, 1):
            fh.write(json.dumps(dict(r, seq=i, run_id="r", agent_id="a")) + "\n")
    return path


def test_grading_uses_saved_code_before_a_native_task_update(tmp_path, monkeypatch):
    path = _journal(tmp_path, [
        {"type": "checkpoint.policy", "pairing": "call-id-v1"},
        {"type": "tool.started", "step": 1, "call_id": "commit", "name": "mcp__ash__shell"},
        {"type": "checkpoint.captured", "step": 1, "snapshot_id": "saved-code",
         "reason": "captured", "captured": True},
        {"type": "tool.started", "step": 2, "call_id": "todo", "name": "TaskUpdate"},
        {"type": "checkpoint.captured", "step": 2, "call_id": "todo", "snapshot_id": None,
         "reason": "not_executed", "captured": False},
    ])
    graded = []

    def grade(snapshot_id, instance, backend):
        graded.append(snapshot_id)
        return fork_eval.Grade(resolved=False, detail="official test failure")

    monkeypatch.setattr(fork_eval, "backend_for", lambda *args: None)
    result = fork_eval.grade_attempt(SimpleNamespace(journal_path=path), {}, None,
                                     SimpleNamespace(grade=grade))
    assert graded == ["saved-code"]
    assert not result.resolved and result.detail == "official test failure"
    assert result.grading_snapshot["policy"] == "last_successful_snapshot"
    assert result.grading_snapshot["capture_step"] == 1
    assert result.grading_snapshot["later_tool_calls"] == [
        {"step": 2, "call_id": "todo", "name": "TaskUpdate"}]
    assert result.grading_snapshot["later_checkpoint_issues"][0]["reason"] == "not_executed"


def test_grading_ignores_old_backfills_and_failed_snapshot_references(tmp_path, monkeypatch):
    path = _journal(tmp_path, [
        {"type": "checkpoint.captured", "step": 1, "snapshot_id": "old"},
        {"type": "checkpoint.captured", "step": 2, "snapshot_id": "new", "reason": None},
        {"type": "checkpoint.captured", "step": 1, "snapshot_id": "old", "reason": "session_ref_backfill"},
        {"type": "checkpoint.captured", "step": 3, "snapshot_id": "unproven", "reason": "failed", "captured": False},
    ])
    graded = []

    def grade(snapshot_id, instance, backend):
        graded.append(snapshot_id)
        return fork_eval.Grade()

    monkeypatch.setattr(fork_eval, "backend_for", lambda *args: None)
    result = fork_eval.grade_attempt(SimpleNamespace(journal_path=path), {}, None,
                                     SimpleNamespace(grade=grade))
    assert graded == ["new"]
    assert result.grading_snapshot["capture_step"] == 2
    assert result.grading_snapshot["later_checkpoint_issues"] == [
        {"step": 3, "call_id": None, "reason": "failed"}]


@pytest.mark.parametrize("records", [[], [
    {"type": "checkpoint.captured", "step": 1, "snapshot_id": "unproven", "reason": "failed", "captured": False},
]])
def test_grading_without_a_successful_snapshot_still_fails(tmp_path, records):
    path = _journal(tmp_path, records)
    result = fork_eval.grade_attempt(SimpleNamespace(journal_path=path), {}, None)
    assert result.error == "no successful snapshot recorded -- nothing to grade"
    assert result.grading_snapshot is None


def test_regrade_persists_the_same_last_successful_snapshot_policy(tmp_path, monkeypatch):
    directory = tmp_path / "task-a"
    directory.mkdir()
    _journal(directory, [
        {"type": "checkpoint.captured", "step": 1, "snapshot_id": "saved"},
        {"type": "checkpoint.captured", "step": 2, "snapshot_id": "not-saved", "reason": "failed"},
    ])
    graded = []

    def grade(snapshot_id, instance, backend):
        graded.append(snapshot_id)
        return fork_eval.Grade(resolved=True)

    bench = SimpleNamespace(catalogue=lambda args: {"task-a": "raw"},
                            instance=lambda raw: {}, grade=grade)
    monkeypatch.setattr(fork_eval, "backend_for", lambda *args: None)
    assert fork_eval.regrade(SimpleNamespace(slot="test", model="test"), tmp_path, bench) == 0
    assert graded == ["saved"]
    recorded = json.loads((tmp_path / "regrade.json").read_text())["instances"][0]["attempts"][0]
    assert recorded["grading_snapshot"]["snapshot_id"] == "saved"
    assert recorded["grading_snapshot"]["later_checkpoint_issues"][0]["reason"] == "failed"


def test_a_long_tool_result_keeps_its_tail():
    """THE defect this budget exists to fix. Tool results on a real run have a
    median of ~1.9k characters and a max of ~17k; the old 300-character head-only
    cap fed the analyst a test run's banner and threw away the assertion that
    explains the failure -- which is the one thing it needs to diagnose."""
    body = "banner " * 2000 + "ASSERTION FAILED: the answer"
    clipped = _clip(body, 300)
    assert len(clipped) < len(body)
    assert "ASSERTION FAILED: the answer" in clipped, "the verdict is at the END"
    assert clipped.startswith("banner"), "and the head still identifies it"


def test_a_short_result_is_untouched():
    assert _clip("small", 300) == "small"


def test_step_numbers_match_the_snapshot_map(tmp_path):
    """The analyst names a step and fork_plan looks it up: both count one per
    exec call, in order. Two counting schemes would put every branch at the
    wrong snapshot."""
    path = _journal(tmp_path, [
        {"type": "run.started"},
        {"type": "tool.started", "name": "mcp__ash__shell", "args": {"command": "a"}},
        {"type": "tool.finished", "name": "mcp__ash__shell", "output": "out-a"},
        {"type": "checkpoint.captured", "step": 1, "snapshot_id": "s1"},
        {"type": "tool.started", "name": "mcp__ash__text_editor", "args": {"path": "p"}},
        {"type": "tool.finished", "name": "mcp__ash__text_editor", "output": "OK"},
        {"type": "checkpoint.captured", "step": 2, "snapshot_id": "s2"},
    ])
    body, lo, hi = render_transcript(path)
    assert (lo, hi) == (1, 2)
    assert body.startswith("[1] shell(")
    assert "[2] text_editor(" in body
    assert "out-a" in body and "OK" in body


def test_the_agents_own_words_are_included(tmp_path):
    """Its stated intent is how you tell a wrong turn from a wrong result."""
    path = _journal(tmp_path, [
        {"type": "agent.message", "text": "I will change only __eq__"},
    ])
    body, _, _ = render_transcript(path)
    assert "I will change only __eq__" in body


def test_the_token_budget_elides_the_middle_not_the_ends(tmp_path):
    """When even the generous budget overflows, the early steps (what was
    understood) and the late ones (the failure) are what must survive."""
    records = []
    for i in range(400):
        records.append({"type": "tool.started", "name": "mcp__ash__shell",
                        "args": {"command": "step-%d" % i}})
        records.append({"type": "tool.finished", "name": "mcp__ash__shell",
                        "output": "x" * 4000})
    path = _journal(tmp_path, records)
    body, _, hi = render_transcript(path, token_budget=2000)
    assert hi == 400, "the step COUNT is still the truth"
    assert "middle elided" in body
    assert "step-0" in body and "step-399" in body


# --- naming the regression, not just tailing the log --------------------------
def test_failing_test_names_are_extracted_from_every_runner_format():
    """The one thing an analyst most needs about a regression is WHICH test broke.
    Measured: two instances in an 8-run batch stalled at "target passes,
    regressions fail" across seven branches each, because the verdict carried
    1200 trailing characters of a 57-test run and the failing name was usually
    not in them -- so every branch guessed at what it had broken."""
    from swebench.fork_eval import _failing_tests

    assert _failing_tests(
        "FAILED lib/t_a.py::test_one - AssertionError\n"
        "FAILED lib/t_b.py::test_two\n=== 2 failed ==="
    ) == ["lib/t_a.py::test_one", "lib/t_b.py::test_two"]

    # sympy's own runner and the direct-call runner
    assert _failing_tests("PASS m.test_x\nFAIL m.test_y") == ["m.test_y"]
    # django's runner, and errors count too
    assert _failing_tests("FAIL: test_a (mod.Cls)\nERROR: test_b (mod.Cls)") == \
        ["test_a", "test_b"]
    assert _failing_tests("everything passed") == []


def test_extraction_deduplicates_and_is_bounded():
    """pytest names a failure twice (inline and in the summary banner), and a
    sweeping change can break hundreds -- neither should flood the prompt."""
    from swebench.fork_eval import _failing_tests

    doubled = "FAILED a.py::test_x\n" * 3
    assert _failing_tests(doubled) == ["a.py::test_x"]

    many = "".join("FAILED a.py::test_%d\n" % i for i in range(100))
    assert len(_failing_tests(many)) == 25


def test_a_verdict_with_no_regressions_says_nothing_about_them():
    """Silence is the correct output when nothing broke; inventing an empty
    'BROKEN:' line would read as a finding."""
    from swebench.fork_eval import Grade

    grade = Grade(f2p_pass=True, p2p_ran=True, p2p_pass=True, resolved=True)
    assert grade.broken == []
    assert "BROKEN" not in grade.summary()


# --- the two grading conventions that must not silently regress ---------------
class _RecordingSession:
    """Stands in for SandboxSession; records every command grade_snapshot runs."""

    def __init__(self, *a, **k):
        self.commands = []
        self.create_error = None

    def create(self, image):
        return True

    def destroy(self):
        pass

    def execute(self, tool, args):
        import json as _json
        from harness.core.result import ToolResult

        self.commands.append((tool, args))
        return ToolResult(success=True, output=_json.dumps(
            {"exit_code": 0, "stdout": "x ... ok", "stderr": ""}))


def test_agent_test_edits_are_reverted_before_the_test_patch_lands(monkeypatch):
    """Public-leaderboard convention: the model's patch excludes test files, so
    edits to graded tests are DISCARDED before grading, not graded as a fatal
    collision. 53 of the first full 500 were killed by the old behaviour; the
    first one re-graded under the convention was simply resolved -- its source
    fix had been right all along. The revert must come BEFORE `git apply` of the
    test_patch, or the collision it prevents still happens."""
    import harness.execution.session as session_module
    from swebench.fork_eval import grade_snapshot

    recorder = {}
    monkeypatch.setattr(session_module, "SandboxSession",
                        lambda *a, **k: recorder.setdefault("s", _RecordingSession()))
    grade_snapshot("snap-1", {
        "instance_id": "x", "repo": "some/repo",
        "f2p": ["tests/t.py::test_a"], "p2p": [],
        "test_patch": "--- a/tests/t.py\n+++ b/tests/t.py\n@@ -1 +1 @@\n-a\n+b\n",
    }, {"backend": "docker"})
    shell = [args["command"] for tool, args in recorder["s"].commands
             if tool == "shell"]
    revert = next((i for i, c in enumerate(shell) if "checkout HEAD" in c), None)
    apply_ = next((i for i, c in enumerate(shell) if "git apply" in c), None)
    assert revert is not None, "no revert of the graded test files"
    assert "tests/t.py" in shell[revert]
    assert apply_ is not None and revert < apply_, \
        "the revert must precede the test_patch application"


def test_the_django_run_forces_utf8_stdout():
    """--verbosity 2 makes django print "Creating tables…", and one ellipsis
    under the images' ascii locale killed the whole run with UnicodeEncodeError
    before any test executed. Found because the MUST-PASS validation case failed
    -- a previously-resolved instance graded as target-FAIL. Without the forced
    encoding every django verdict is fiction again."""
    import inspect

    from swebench import fork_eval

    source = inspect.getsource(fork_eval._grade_django)
    assert "PYTHONIOENCODING=utf-8" in source
    assert "--verbosity 2" in source, \
        "output parsing NEEDS verbosity 2 -- the docstring lines only exist there"


def test_verdict_text_names_broken_tests_and_survives_empty_grade():
    """The verdict is what both analyst stages and every branch prompt see; the
    BROKEN names are the single most useful fact in it (measured: without them,
    branches guessed for seven attempts what they had regressed)."""
    from swebench.fork_eval import Attempt, Grade

    grade = Grade(f2p_pass=True, p2p_ran=True, p2p_pass=False,
                  broken=["mod.test_a"], patch="+x\n", detail="d")
    attempt = Attempt("r1b1", outcome=None, grade=grade, hint="try X")
    text = attempt.verdict_text()
    assert "mod.test_a" in text and "BROKE" in text
    assert Attempt("p", None, Grade()).verdict_text()  # empty grade renders too


@pytest.mark.parametrize("template,input_fields,output_fields", [
    (fork_eval._CASE_PROMPT,
     {"problem", "verdict", "transcript", "lo", "hi", "checkpoint_steps", "candidate_limit"},
     {"failure_reason", "lesson", "salvage", "branch_candidates"}),
    (fork_eval._REVIEW_PROMPT,
     {"problem", "reports", "count_rule"},
     {"synthesis", "branches"}),
])
def test_analysis_prompts_preserve_fields_and_separate_private_evidence(
        template, input_fields, output_fields):
    fields = {entry[1] for entry in Formatter().parse(template) if entry[1]}
    assert fields == input_fields
    values = {field: "INPUT_%s" % field.upper() for field in input_fields}
    rendered = template.format(**values)
    assert all(value in rendered for value in values.values())
    schema = json.loads(rendered.rsplit("Return ONLY a JSON object, no prose:", 1)[1]
                        .replace("<int>", "1"))
    assert set(schema) == output_fields
    compact = " ".join(rendered.split())
    assert "private diagnostic evidence" in compact
    assert "natural and effective continuation" in compact


def test_analyst_translates_private_evidence_to_accessible_repair_directions():
    """Instruction coverage, not a claim that a model always follows it."""
    prompt = " ".join(fork_eval._CASE_PROMPT.split())
    assert "## Translate evidence into repair directions" in prompt
    assert "A path in verifier output is not evidence that the actor can open it" in prompt
    assert "In every returned field" in prompt
    assert "omit verifier-only filenames, test names/IDs and grader paths" in prompt
    assert "task or retained prefix establishes it is available" in prompt
    assert "Do not ask the actor to read, locate or recreate a hidden test" in prompt
    assert "compliance_test.go" not in fork_eval._CASE_PROMPT


def test_analyst_uncertainty_still_produces_a_testable_direction():
    prompt = " ".join(fork_eval._CASE_PROMPT.split())
    assert "strongest supported hypothesis" in prompt
    assert "what would support or rule it out" in prompt
    assert "Do not merely add 'possibly' to an invented explanation" in prompt
    assert "do not guess how an unseen test constructs its fixtures" in prompt
    assert "If the cause is established, state the correction directly" in prompt
    assert "caller-owned or handler-owned prefix" in prompt


def test_analysis_prompts_anchor_guidance_to_the_retained_prefix():
    analyst = " ".join(fork_eval._CASE_PROMPT.split())
    reviewer = " ".join(fork_eval._REVIEW_PROMPT.split())
    assert "Separate observed facts from inferences" in analyst
    assert "failure_reason and lesson are controller-only reports" in analyst
    assert "state after each candidate step" in analyst
    assert "LATER IS BETTER" in analyst
    assert "For EACH branch, choose its own base attempt and branch_step" in reviewer
    assert "Positions do NOT need to be distinct" in reviewer
    assert "state AFTER branch_step" in reviewer
    assert "already present or already verified" in reviewer
    assert "Prefer 2-5 concise sentences" in reviewer
    assert "through its own tools" in reviewer
    assert "not just the wording" in reviewer


def test_reviewer_prompt_includes_continuity_usefulness_and_no_source_narration():
    reviewer = " ".join(fork_eval._REVIEW_PROMPT.split())
    assert "delivered VERBATIM" in reviewer
    assert "mentally remove the hint" in reviewer
    assert "Do not make a precise diagnosis vague" in reviewer
    assert "Do not fabricate the agent's reasoning" in reviewer
    assert "do not include test names or IDs" in reviewer
    assert "pass/fail counts, scores, raw verifier output" in reviewer
    assert "Do not label the text as a reminder" in reviewer
    assert "useful diagnosis and behavioral constraints are still intact" in reviewer


@pytest.mark.parametrize("hint", [
    "Inspect subclass dispatch. Reproduce the operand case before editing.",
    "  The caller owns /environments/:id/compliance.\n"
    "RegisterRoutes registers only /baselines beneath that group.  ",
])
def test_branch_loop_consumes_both_analysis_prompts_and_preserves_the_hint(
        tmp_path, monkeypatch, hint):
    from swebench.tests.test_parent_from import write_journal

    write_journal(tmp_path / "base" / "shard-0" / "task-a" / "parent.jsonl")
    analysis = {
        "failure_reason": "Subclass overrides may bypass the shared method.",
        "lesson": "Trace dispatch and reproduce the affected operand case.",
        "salvage": "At step 2 the shared comparison implementation is visible.",
        "branch_candidates": [{"step": 2, "why": "Inspect subclass dispatch."}],
    }
    plan = {
        "synthesis": "An override may require a different comparison path.",
        "branches": [{"name": "dispatch", "base": "parent", "branch_step": 2,
                      "why": "Keep the inspected code as the starting point.", "hint": hint}],
    }
    replies = iter([analysis, plan])
    prompts = []
    branch_calls = []

    def ask_analyst(model, prompt):
        prompts.append(prompt)
        return json.dumps(next(replies))

    def run_attempt(orch, args, instance, **kwargs):
        branch_calls.append(kwargs)
        journal = write_journal(kwargs["out_dir"] / (kwargs["name"] + ".jsonl"))
        return SimpleNamespace(status="completed", error=None, checkpoints=3,
                               journal_path=journal)

    class Benchmark(fork_eval.Benchmark):
        name = "prompt-test"

        def instance(self, raw):
            return {"instance_id": raw, "repo": "repo", "image": "image",
                    "problem": "Fix comparison dispatch.", "f2p": [], "p2p": []}

        def grade(self, snapshot_id, instance, backend):
            return fork_eval.Grade(
                patch="diff", detail="tests/hidden.py::test_dispatch failed")

        def branch_prompt(self, instance, verdict, hint, **context):
            assert verdict == ""
            assert "analysis" not in context and "grade" not in context
            return hint

    monkeypatch.setattr(fork_eval, "ask_analyst", ask_analyst)
    monkeypatch.setattr(fork_eval, "run_attempt", run_attempt)
    monkeypatch.setattr(fork_eval, "conversation_cut", lambda *args: "result-2")
    args = SimpleNamespace(
        rounds=1, slot="claude-code", model="model", analyst_model="model",
        analyst_tokens=1000, timeout=10.0, runtime_bin="runtime/ash-runtime",
        parent_from=str(tmp_path / "base"), fork_full_conversation=False)
    attempts = fork_eval.run_one(
        None, args, "task-a", [1], tmp_path / "out" / "task-a", Benchmark())

    assert len(prompts) == 2
    assert "## Evidence and output rules" in prompts[0]
    assert "tests/hidden.py::test_dispatch failed" in prompts[0]
    assert "## Hint and output rules" in prompts[1]
    assert analysis["lesson"] in prompts[1]
    assert len(branch_calls) == 1
    assert branch_calls[0]["image"] == "snap-2"
    assert branch_calls[0]["resume_at"] == "result-2"
    assert branch_calls[0]["prompt"] == hint
    assert branch_calls[0]["origin"]["actor_hint"] == hint
    assert attempts[1].hint == hint
    assert branch_calls[0]["origin"]["hint_delivery"] == "reviewer-direct"
    assert not list((tmp_path / "out/task-a").glob("hint-*.json"))
    recorded = json.loads((tmp_path / "out/task-a/plan-round1.json").read_text())
    assert recorded["review"] == plan
    assert recorded["review"]["branches"][0]["hint"] == hint


def test_failed_branch_hint_is_withheld_from_analyst_but_kept_for_reviewer(
        tmp_path, monkeypatch):
    from swebench.tests.test_parent_from import write_journal

    write_journal(tmp_path / "source/task-a/parent.jsonl")
    supplied_hint = "PRIOR_DIRECTION_ONLY_FOR_ACTOR_AND_REVIEWER"
    analyst_inputs, reviewer_inputs = [], []

    def ask(model, prompt):
        if "## Every attempt so far" in prompt:
            reviewer_inputs.append(prompt)
            return json.dumps({
                "synthesis": "Boundary case.",
                "branches": [{"name": "boundary", "base": "parent", "branch_step": 2,
                              "why": "Inspect a remaining boundary.", "hint": supplied_hint}],
            })
        analyst_inputs.append(prompt)
        return json.dumps({
            "failure_reason": "Observed behavior still differs from the task.",
            "lesson": "Check the public boundary.", "salvage": "Existing code.",
            "branch_candidates": [{"step": 2, "why": "Relevant code is present."}],
        })

    def run_attempt(orch, args, instance, **kwargs):
        journal = write_journal(kwargs["out_dir"] / (kwargs["name"] + ".jsonl"))
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        records[0]["task_prompt"] = kwargs["prompt"]
        records.insert(0, {"type": "fork.origin", "actor_hint": supplied_hint})
        journal.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        return SimpleNamespace(status="completed", error=None, checkpoints=3,
                               journal_path=journal)

    class Benchmark(fork_eval.Benchmark):
        name = "analyst-input-test"

        def instance(self, raw):
            return {"instance_id": raw, "repo": "repo", "image": "image",
                    "problem": "PUBLIC_TASK", "f2p": [], "p2p": []}

        def grade(self, snapshot_id, instance, backend):
            return fork_eval.Grade(patch="PUBLIC_PATCH", detail="OBSERVED_VERIFICATION")

        def branch_prompt(self, instance, verdict, hint, **context):
            return hint

    monkeypatch.setattr(fork_eval, "ask_analyst", ask)
    monkeypatch.setattr(fork_eval, "run_attempt", run_attempt)
    monkeypatch.setattr(fork_eval, "conversation_cut", lambda *args: "cut-2")
    args = SimpleNamespace(
        rounds=2, slot="claude-code", model="model", analyst_model="model",
        analyst_tokens=1000, timeout=10.0, runtime_bin="runtime/ash-runtime",
        parent_from=str(tmp_path / "source"), fork_full_conversation=False)
    attempts = fork_eval.run_one(None, args, "task-a", [1, 1], tmp_path / "out",
                                 Benchmark())

    assert len(analyst_inputs) == len(reviewer_inputs) == 2
    assert all(supplied_hint not in prompt for prompt in analyst_inputs)
    assert all("PUBLIC_TASK" in prompt and "PUBLIC_PATCH" in prompt
               and "OBSERVED_VERIFICATION" in prompt for prompt in analyst_inputs)
    assert all("## Translate evidence into repair directions" in prompt
               and "strongest supported hypothesis" in prompt for prompt in analyst_inputs)
    assert supplied_hint not in reviewer_inputs[0]
    assert supplied_hint in reviewer_inputs[1]
    assert all(attempt.hint == supplied_hint for attempt in attempts[1:])


def test_analyst_transcript_omits_injected_note_without_rewriting_agent_speech(tmp_path):
    path = tmp_path / "branch.jsonl"
    records = [
        {"type": "fork.origin", "actor_hint": "INJECTED_NOTE"},
        {"type": "run.started", "task_prompt": "INJECTED_NOTE"},
        {"type": "agent.message", "text": "ACTUAL_RECORDED_SPEECH about a reminder"},
        {"type": "tool.started", "name": "shell", "args": {"command": "ls"}},
        {"type": "tool.finished", "output": "OBSERVED_FILES"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    text, lo, hi = fork_eval.render_transcript(path)
    assert "INJECTED_NOTE" not in text
    assert "ACTUAL_RECORDED_SPEECH about a reminder" in text
    assert "OBSERVED_FILES" in text
    assert (lo, hi) == (1, 1)
