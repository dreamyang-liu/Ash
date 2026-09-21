from copy import deepcopy
import json

import pytest

from harness.core.assistant_turn import validate_assistant_turn


def assistant_turn(command="printf check", identifier="reviewer-call-1"):
    return {"role": "assistant", "content": "The existing branch suggests checking this boundary.",
            "tool_calls": [{"id": identifier, "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": command})}}]}


def test_turn_is_preserved_without_sharing_mutable_reviewer_input():
    message = assistant_turn()
    result = validate_assistant_turn(message)
    assert result == message and result is not message
    result["tool_calls"][0]["function"]["arguments"] = "{}"
    assert message == assistant_turn()


@pytest.mark.parametrize("change", [
    lambda m: m.update(role="user"),
    lambda m: m.update(content=""),
    lambda m: m.update(reasoning_content="native thinking"),
    lambda m: m.update(extra={"actions": [{"command": "unvalidated"}]}),
    lambda m: m.update(tool_calls=[]),
    lambda m: m["tool_calls"].append(deepcopy(m["tool_calls"][0])),
    lambda m: m["tool_calls"][0].update(id=""),
    lambda m: m["tool_calls"][0]["function"].update(name="shell"),
    lambda m: m["tool_calls"][0]["function"].update(arguments={"command": "pwd"}),
    lambda m: m["tool_calls"][0]["function"].update(arguments="{invalid"),
    lambda m: m["tool_calls"][0]["function"].update(arguments='{"command":"pwd","cwd":"/host"}'),
    lambda m: m["tool_calls"][0]["function"].update(arguments='{"command":1}'),
])
def test_invalid_synthetic_turn_is_rejected(change):
    message = assistant_turn()
    change(message)
    with pytest.raises(ValueError, match="assistant_turn"):
        validate_assistant_turn(message)


def test_call_id_cannot_repeat_an_inherited_call():
    with pytest.raises(ValueError, match="unique across history"):
        validate_assistant_turn(assistant_turn(), history=[assistant_turn()])


@pytest.mark.parametrize("name", ["shell", "python", "apply_patch", "text_editor", "Bash", "mcp__bash"])
def test_non_bash_name_is_reported_for_reviewer_correction(name):
    message = assistant_turn()
    message["tool_calls"][0]["function"]["name"] = name
    with pytest.raises(ValueError, match="only bash is allowed") as error:
        validate_assistant_turn(message)
    assert repr(name) in str(error.value)
    assert "tool_calls[0]" in str(error.value)


def test_bash_programs_are_not_mistaken_for_tool_names():
    message = assistant_turn('python3 -c "print(1)" && git status --short')
    assert validate_assistant_turn(message) == message
