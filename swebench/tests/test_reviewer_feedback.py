import json

import pytest

from harness.core.assistant_turn import validate_assistant_turn
from swebench.branch_plan import ReviewerPlanError, review_with_feedback
from swebench.tests.test_branch_selection import direction, exercise


def test_runner_corrects_invalid_selection_before_execution(tmp_path, monkeypatch):
    _, calls, _, prompts = exercise(
        tmp_path, monkeypatch,
        [{"branches": [direction(99)]}, {"branches": [direction(2)]},
         {"branches": []}], reviewer_max_attempts=3)
    assert len(calls) == 1
    assert calls[0]["image"] == "snap-2"
    assert "no eligible checkpoint" in prompts[1]
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert [a["status"] for a in saved["review_attempts"]] == ["validation_error", "validated"]
    assert "validation_error" not in saved


def test_nested_json_error_returned_verbatim_and_response_preserved():
    def turn(arguments):
        return {"role": "assistant", "content": "Inspect the match.",
                "tool_calls": [{"id": "review-1", "type": "function",
                                "function": {"name": "bash", "arguments": arguments}}]}
    bad = json.dumps(turn('{"command":"grep a\\|b file"}'))
    good = json.dumps(turn(json.dumps({"command": r"grep a\|b file"})))
    replies, prompts, saved = iter([bad, good]), [], {}

    def request(prompt):
        prompts.append(prompt)
        return next(replies)

    def validate(plan):
        try:
            return validate_assistant_turn(plan)
        except ValueError as error:
            raise ReviewerPlanError(str(error)) from error

    result = review_with_feedback(request, "original instructions", json.loads, validate, record=saved)
    assert result[1] == json.loads(good)
    error = saved["review_attempts"][0]["validation_error"]
    assert "Invalid \\escape" in error and "tool_calls[0]" in error
    assert error in prompts[1] and bad in prompts[1]
    assert saved["review_attempts"][0]["response"] == bad


@pytest.mark.parametrize("limit", [1, 3])
def test_exhaustion_keeps_every_reply(limit):
    saved, persisted = {}, []
    result = review_with_feedback(
        lambda _: "{bad json", "prompt", json.loads, lambda p: p,
        max_attempts=limit, record=saved,
        persist=lambda data: persisted.append(json.loads(json.dumps(data))))
    assert result is None
    assert len(saved["review_attempts"]) == limit
    assert all(a["status"] == "validation_error" for a in saved["review_attempts"])
    assert persisted[-1] == saved


def test_outer_json_corrected_and_empty_plan_accepted():
    replies = iter(["not JSON", '{"branches": []}'])
    saved = {}
    result = review_with_feedback(lambda _: next(replies), "prompt", json.loads,
                                  lambda p: p["branches"], record=saved)
    assert result == ({"branches": []}, [])
    assert len(saved["review_attempts"]) == 2


@pytest.mark.parametrize("where", ["request", "state"])
def test_non_model_failure_not_retried(where):
    saved = {}

    def fail(_):
        raise ValueError("unavailable native history" if where == "state" else "authentication")

    result = review_with_feedback(
        fail if where == "request" else lambda _: "{}",
        "prompt", json.loads, fail, record=saved)
    assert result is None
    assert len(saved["review_attempts"]) == 1
    assert saved["review_attempts"][0]["status"] == where + "_error"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_fails_before_request(limit):
    with pytest.raises(ValueError, match="positive integer"):
        review_with_feedback(lambda _: pytest.fail("unexpected request"),
                             "", json.loads, lambda p: p, max_attempts=limit)
