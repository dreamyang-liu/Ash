from copy import deepcopy
import json

import pytest

from harness.core.assistant_turn import validate_assistant_turn
from swebench.assistant_branch import ASSISTANT_REVIEW_PROMPT
from swebench.branch_plan import review_with_feedback
from swebench.structured_review import extract_branch_plan, request_format, response_text


def plan(command):
    return {
        "synthesis": "Check the implementation",
        "branches": [{
            "name": "inspect", "base": "parent", "branch_step": 6, "why": "Inspect visible code",
            "assistant_turn": {
                "role": "assistant", "content": "I will inspect the implementation.",
                "tool_calls": [{
                    "id": "fresh-call", "type": "function",
                    "function": {"name": "bash", "arguments": {"command": command}},
                }],
            },
        }],
    }


def fenced(value):
    return "```branch-plan\n" + json.dumps(value, ensure_ascii=False) + "\n```"


def reply(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def prompt(reports="[]"):
    return ASSISTANT_REVIEW_PROMPT.format(
        problem="Fix the parser", reports=reports, count_rule="At most 4 branches.")


def test_nested_shell_quotes_and_multiline_commands_roundtrip_without_repair():
    command = '''python - <<'PY'\nprint("a \\"quoted\\" value", r"\\n", "$(literal)", "中文")\nPY\n'''
    wire = plan(command)
    before = deepcopy(wire)
    decoded = extract_branch_plan(response_text("reviewer", reply(fenced(wire))))
    turn = decoded["branches"][0]["assistant_turn"]
    validate_assistant_turn(turn)
    assert json.loads(turn["tool_calls"][0]["function"]["arguments"]) == {"command": command}
    assert wire == before


@pytest.mark.parametrize("mutation", [
    lambda p: p["branches"][0]["assistant_turn"]["tool_calls"][0]["function"].update(name="shell"),
    lambda p: p["branches"][0]["assistant_turn"]["tool_calls"][0]["function"]["arguments"].update(timeout=30),
    lambda p: p["branches"][0]["assistant_turn"]["tool_calls"][0]["function"].update(arguments='{"command":"pwd"}'),
    lambda p: p["branches"][0].update(hint="unexpected"),
    lambda p: p["branches"][0].update(branch_step=True),
    lambda p: p["branches"][0].pop("name"),
    lambda p: p.pop("synthesis"),
])
def test_invalid_wire_fields_are_rejected_without_filling_or_coercion(mutation):
    value = plan("pwd")
    mutation(value)
    before = deepcopy(value)
    with pytest.raises(ValueError, match="Structured branch-plan"):
        extract_branch_plan(fenced(value))
    assert value == before


def test_invalid_json_is_never_guessed_or_repaired():
    text = '```branch-plan\n{"synthesis":"x" "branches":[]}\n```'
    assert response_text("reviewer", reply(text)) == text
    with pytest.raises(ValueError, match="Invalid JSON"):
        extract_branch_plan(text)


def test_prompt_preserves_input_fence_and_complete_structure_example():
    history = json.dumps([{"native_history": {
        "messages": [{"role": "user", "content": "function.arguments is a JSON STRING containing exactly"}]
    }}])
    original = prompt(history)
    kind, adapted, fields = request_format(original)
    assert kind == "reviewer" and fields == {}
    assert adapted.split("## The response you author")[0] == original.split("## The response you author")[0]
    assert history in adapted
    assert "function.arguments is an OBJECT" in adapted
    assert "Return exactly one fenced code block labelled branch-plan" in adapted
    assert adapted.count("```branch-plan") == 1
    for key in ("synthesis", "branches", "name", "base", "branch_step", "why",
                "assistant_turn", "role", "content", "tool_calls", "id", "type", "function"):
        assert '"' + key + '"' in adapted
    assert '"arguments": {"command": "<bash command>"}' in adapted
    assert "ash_reviewer" not in adapted


def test_real_feedback_loop_keeps_response_and_aligns_argument_instructions():
    original = prompt()
    invalid = plan("pwd")
    invalid["branches"][0]["assistant_turn"]["tool_calls"][0]["function"]["arguments"]["tail"] = 20
    command = "printf '%s\\n' 'a \"quoted\" value'"
    replies = [fenced(invalid), fenced(plan(command))]
    requests, validated = [], []

    def request(text):
        kind, adapted, fields = request_format(text)
        assert fields == {}
        requests.append(adapted)
        return response_text(kind, reply(replies[len(requests) - 1]))

    def validate(parsed):
        turn = parsed["branches"][0]["assistant_turn"]
        validate_assistant_turn(turn)
        validated.append(turn)
        return parsed["branches"]

    record = {}
    result = review_with_feedback(request, original, extract_branch_plan, validate,
                                  max_attempts=3, record=record)
    assert result is not None and len(requests) == 2 and len(validated) == 1
    assert replies[0] in requests[1]
    feedback = requests[1].split("## Validation feedback\n", 1)[1]
    assert "arguments JSON string" not in feedback
    assert "return an object containing exactly one string field, command" in feedback
    assert "tail" in feedback
    assert json.loads(validated[0]["tool_calls"][0]["function"]["arguments"]) == {"command": command}
    assert [row["status"] for row in record["review_attempts"]] == ["validation_error", "validated"]


def test_prose_outside_existing_fence_is_not_wrapped_again():
    text = "A short explanation.\n\n" + fenced(plan("pwd")) + "\nEnd."
    assert response_text("reviewer", reply(text)) == text
    assert extract_branch_plan(text)["branches"][0]["name"] == "inspect"


@pytest.mark.parametrize("text", [
    json.dumps(plan("pwd")),
    "## Synthesis\nprose\n" + json.dumps(plan("pwd")),
    fenced(plan("pwd")) + "\n" + fenced(plan("ls")),
])
def test_missing_or_multiple_fences_remain_rejected(text):
    assert response_text("reviewer", reply(text)) == text
    with pytest.raises(ValueError):
        extract_branch_plan(text)


def test_existing_call_id_validation_is_not_weakened():
    parsed = extract_branch_plan(fenced(plan("pwd")))
    history = [{"tool_calls": [{"id": "fresh-call"}]}]
    with pytest.raises(ValueError, match="unique"):
        validate_assistant_turn(parsed["branches"][0]["assistant_turn"], history=history)


def test_analyst_prompt_parameters_and_response_are_unchanged():
    original = "You are analysing ONE failed coding attempt.\nPrivate evidence"
    kind, adapted, fields = request_format(original)
    assert (kind, adapted, fields) == ("analyst", original, {})
    text = '{"failure_reason":"x"}'
    assert response_text(kind, reply(text)) == text


def test_unexpected_tool_reply_is_not_executed_or_adopted():
    value = {"choices": [{"message": {"tool_calls": [{
        "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
    }]}}]}
    with pytest.raises(ValueError):
        extract_branch_plan(response_text("reviewer", value))


def test_other_prompt_modes_fail_before_request():
    with pytest.raises(ValueError, match="Only analyst"):
        request_format("Choose points only")
