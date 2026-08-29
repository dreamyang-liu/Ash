"""What the analyst actually gets to see."""
import json

import pytest

from swebench.fork_eval import RESULT_CHARS, _clip, render_transcript


def _journal(tmp_path, records):
    path = tmp_path / "j.jsonl"
    with open(path, "w") as fh:
        for i, r in enumerate(records, 1):
            fh.write(json.dumps(dict(r, seq=i, run_id="r", agent_id="a")) + "\n")
    return path


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
