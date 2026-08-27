"""Branch-point analysis over an unfinished attempt's checkpoints."""

import json
import types

import pytest

from swebench.branching import (BranchingError, analyze, branch_commands,
                                branchable_steps, render_trajectory)


def write_trajectory(tmp_path, *, steps=6, risky=(), events=True):
    records = []
    step_snapshots = {}
    for turn in range(steps + 1):
        snap = f"snap-{turn:03d}"
        step_snapshots[str(turn)] = snap
        records.append({"turn": turn, "snapshot_id": snap, "captured": True,
                        "reason": "captured",
                        "live_background": turn in risky})
    data = {
        "instance_id": "rust-java-lsp",
        "exit_status": "completed",
        "info": {
            "exit_status": "completed",
            "checkpoints": {"step_snapshots": step_snapshots,
                            "records": records, "disk_only": True},
            "marathon": {"reward": 0.0, "partial_score": 0.12,
                         "metrics": {"total_passed": 8, "total_tests": 68},
                         "grading_error": None},
        },
    }
    if events:
        events_list = []
        for n in range(1, steps + 1):
            events_list.append({"type": "tool_use", "step": n, "id": f"t{n}",
                                "name": "shell",
                                "input": {"command": f"cargo build step{n}"}})
            events_list.append({"type": "tool_result", "tool_use_id": f"t{n}",
                                "content": [{"type": "text",
                                             "text": f"output {n}"}],
                                "is_error": False})
        events_list.append({"type": "text", "text": "I think the parser is wrong."})
        data["trajectory"] = events_list
    else:
        data["messages"] = [{"role": "assistant", "content": "trying X",
                             "tool_calls": [{"function": {"name": "shell"}}]}]
    path = tmp_path / "traj.json"
    path.write_text(json.dumps(data))
    return path


def fake_completion(payload):
    def completion(model, messages, max_tokens):
        completion.prompt = messages[0]["content"]
        message = types.SimpleNamespace(content=json.dumps(payload))
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])
    return completion


PLAN = {"branch_step": 4, "why_here": "before the bad rewrite",
        "what_went_wrong": "parser dropped UTF-16 offsets",
        "branches": [{"name": "fix-forward", "hint": "Keep the index, fix offsets."},
                     {"name": "rewrite-parser", "hint": "Replace the parser layer."}]}


def test_render_covers_both_harness_shapes(tmp_path):
    event_shape = render_trajectory(json.loads(
        write_trajectory(tmp_path).read_text()))
    assert "[3] shell" in event_shape and "agent:" in event_shape

    message_shape = render_trajectory(json.loads(
        write_trajectory(tmp_path, events=False).read_text()))
    assert "assistant" in message_shape and "[calls: shell]" in message_shape


def test_render_says_when_it_elided(tmp_path):
    data = json.loads(write_trajectory(tmp_path, steps=60).read_text())
    rendered = render_trajectory(data, char_budget=1500)
    assert len(rendered) < 2500
    assert "elided" in rendered, "a cut transcript must not read as a short one"
    assert "[60] shell" in rendered, "the tail survives — mistakes live late"


def test_branchable_steps_carry_the_risk_flags(tmp_path):
    data = json.loads(write_trajectory(tmp_path, risky=(3,)).read_text())
    snapshots, risky = branchable_steps(data)
    assert snapshots[0] == "snap-000" and snapshots[6] == "snap-006"
    assert risky == {3}


def test_analyze_builds_a_validated_plan(tmp_path):
    completion = fake_completion(PLAN)
    plan = analyze(write_trajectory(tmp_path), "Build a language server.",
                   branches=2, completion=completion)
    assert (plan.step, plan.snapshot_id) == (4, "snap-004")
    assert [d.name for d in plan.directions] == ["fix-forward", "rewrite-parser"]
    # The analyst sees the outcome and the rules it must follow.
    assert "partial_score: 0.12" in completion.prompt
    assert "LATER IS BETTER" in completion.prompt


def test_analyst_step_without_snapshot_clamps_earlier(tmp_path):
    """Never clamp later: a later snapshot holds work the analyst decided to
    discard, and resuming from it would inherit the mistake."""
    plan = analyze(write_trajectory(tmp_path), "task",
                   completion=fake_completion({**PLAN, "branch_step": 99}))
    assert plan.step == 6, "clamped down to the last mapped step"


def test_notes_make_the_analysis_iterative(tmp_path):
    """When failures are invisible to the agent (hidden holdout), each
    finished branch is a probe: its change plus its verifier delta is
    information nothing else can produce. Round N+1 must condition on it,
    and must be told not to re-guess what round N already ruled out."""
    completion = fake_completion(PLAN)
    analyze(write_trajectory(tmp_path), "task", completion=completion,
            notes="branch fix-forward: holdout 274->274 (no change)")
    assert "holdout 274->274" in completion.prompt
    assert "Do NOT re-propose" in completion.prompt

    bare = fake_completion(PLAN)
    analyze(write_trajectory(tmp_path), "task", completion=bare)
    assert "Do NOT re-propose" not in bare.prompt, \
        "no notes, no phantom earlier round"


def test_missing_map_is_a_loud_error(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"info": {}, "trajectory": []}))
    with pytest.raises(BranchingError, match="checkpoints enabled"):
        analyze(path, "task", completion=fake_completion(PLAN))


def test_branch_commands_fan_out_with_hints(tmp_path):
    plan = analyze(write_trajectory(tmp_path), "task",
                   completion=fake_completion(PLAN))
    commands = branch_commands(plan, task_dir="/tasks/x", config="cfg.yaml",
                               output_base=str(tmp_path / "branches"))
    assert len(commands) == 2
    for (output_dir, argv), direction in zip(commands, plan.directions):
        assert argv[argv.index("--resume-from") + 1] == "snap-004"
        assert argv[argv.index("--resume-hint") + 1] == direction.hint
        assert argv[argv.index("-o") + 1] == str(output_dir)
    a, b = (c[0] for c in commands)
    assert a != b, "parallel branches must not share an output directory"


def test_resume_hint_reaches_both_harness_prompts():
    """The hint is the whole point of a branch; a harness that drops it runs
    a plain resume and the fan-out measures nothing."""
    import inspect
    from swebench.harnesses import marathon, marathon_claude_code

    for module in (marathon, marathon_claude_code):
        source = inspect.getsource(module)
        assert "resume_hint" in source, module.__name__
        assert "DIRECTION FOR THIS ATTEMPT" in source, module.__name__


def test_cli_carries_resume_hint_to_the_harness():
    import inspect
    from swebench import __main__ as cli
    source = inspect.getsource(cli)
    assert '"--resume-hint"' in source
    assert '"resume_hint"' in source
