import json

import pytest

from swebench.branch_plan import extract_branch_plan


def fenced(body, label="branch-plan"):
    return f"```{label}\n{body}\n```"


def test_plan_ignores_commands_examples_and_prose_outside_its_block():
    plan = {"synthesis": "Check the boundary.", "branches": [
        {"name": "edge", "base": "parent", "branch_step": 2,
         "why": "Retain existing work.", "hint": 'Keep {"x": "}"} and `code` unchanged.'}]}
    text = (
        '{"command":"grep example","working_dir":"/app"}\n'
        + fenced('{"branches": [{"name": "wrong-example"}]}', "json") + "\n"
        + "The actual plan follows.\n" + fenced(json.dumps(plan)) + "\n"
        + '{"branches": [{"name": "wrong-tail"}]}'
    )
    assert extract_branch_plan(text) == plan


@pytest.mark.parametrize("text,expected_error", [
    ('{"branches":[]}', "found 0"),
    (fenced('{"branches":[]}', "json"), "found 0"),
    (fenced('{"branches":[]}', "branch-plan-example"), "found 0"),
    ("An inline ```branch-plan marker is not a block.", "found 0"),
    ('```branch-plan\n{"branches":[]}', "not closed"),
    (fenced('{"branches":['), "Invalid JSON"),
    (fenced('{"command":"grep example"}'), "branches list"),
    (fenced('[]'), "branches list"),
    (fenced('null'), "branches list"),
    (fenced('{"branches":{}}'), "branches list"),
    (fenced('{"branches":[]} {"branches":[]}'), "Invalid JSON"),
    (fenced('{"branches":[]}') + "\n" + fenced('{"branches":[]}'), "found 2"),
    (fenced('{"branches":[]}') + '\n```branch-plan\n{"branches":[]}', "found 2"),
])
def test_bad_envelopes_fail_without_falling_back_to_another_json(text, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        extract_branch_plan(text)


def test_plan_tag_inside_an_example_fence_is_not_a_second_plan():
    example = '````text\n```branch-plan\n{"branches": [{"name":"example"}]}\n```\n````\n'
    assert extract_branch_plan(example + fenced('{"branches":[]}')) == {"branches": []}
    with pytest.raises(ValueError, match="found 0"):
        extract_branch_plan(example)


def test_blockquote_is_not_the_final_plan():
    example = '> ```branch-plan\n> {"branches": []}\n> ```\n'
    with pytest.raises(ValueError, match="found 0"):
        extract_branch_plan(example)


def test_crlf_spacing_and_empty_adaptive_plan():
    text = '  ```branch-plan \r\n{"synthesis":"No useful branch.","branches":[]}\r\n  ``` \r\n'
    assert extract_branch_plan(text) == {"synthesis": "No useful branch.", "branches": []}


def test_hint_literal_newline_tolerance_matches_existing_parser():
    text = '```branch-plan\n{"branches":[{"hint":"first line\nsecond line"}]}\n```'
    assert extract_branch_plan(text)["branches"][0]["hint"] == "first line\nsecond line"
