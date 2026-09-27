"""Fenced branch plans with object arguments and lossless serialization."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator, ValidationError

from swebench.branch_plan import extract_branch_plan as _extract_fenced_plan


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


STRING = {"type": "string"}
COMMAND = _object({"command": STRING})
FUNCTION = _object({"name": {"type": "string", "enum": ["bash"]},
                    "arguments": COMMAND})
TOOL_CALL = _object({"id": STRING, "type": {"type": "string", "enum": ["function"]},
                    "function": FUNCTION})
ASSISTANT_TURN = _object({
    "role": {"type": "string", "enum": ["assistant"]}, "content": STRING,
    "tool_calls": {"type": "array", "items": TOOL_CALL},
})
REVIEW_SCHEMA = _object({
    "synthesis": STRING,
    "branches": {"type": "array", "items": _object({
        "name": STRING, "base": STRING, "branch_step": {"type": "integer"},
        "why": STRING, "assistant_turn": ASSISTANT_TURN,
    })},
})
ANALYST_SCHEMA = _object({
    "failure_reason": STRING, "lesson": STRING, "salvage": STRING,
    "branch_candidates": {"type": "array", "items": _object({
        "step": {"type": "integer"}, "why": STRING,
    })},
})
_REVIEWER_START = "You are the REVIEWER for a failed coding task."
_ANALYST_START = "You are analysing ONE failed coding attempt."
_OUTPUT_MARKER = "Return exactly one fenced code block labelled branch-plan"
_FEEDBACK_MARKER = "\n\n## Previous reviewer response"
_AUTHOR_MARKER = "## The response you author\n"
_VALIDATION_MARKER = "\n\n## Validation feedback\n"
_LEGACY_ARGUMENT_FEEDBACK = (
    "For assistant_turn.function.arguments, both the outer plan and the "
    "arguments JSON string must parse independently. Escape newlines, "
    "quotes and backslashes correctly at both levels; do not rely on "
    "the controller to repair them.\n"
)
_OBJECT_ARGUMENT_FEEDBACK = (
    "For assistant_turn.function.arguments, return an object containing exactly "
    'one string field, command: {"command": "<bash command>"}. '
    "Escape strings correctly in the branch-plan JSON. Do not encode the arguments "
    "object as a JSON string; the controller serializes it for the actor without "
    "changing the command. Preserve every required field in the full plan.\n"
)


def request_format(prompt):
    """Keep the existing fence and complete plan; change only argument encoding."""
    if prompt.startswith(_ANALYST_START):
        return "analyst", prompt, {}
    if not prompt.startswith(_REVIEWER_START):
        raise ValueError("Only analyst and assistant-turn reviewer prompts are supported")
    base, feedback_marker, feedback = prompt.partition(_FEEDBACK_MARKER)
    before, author_marker, author = base.partition(_AUTHOR_MARKER)
    instructions, output_marker, output = author.partition(_OUTPUT_MARKER)
    placeholder = '"arguments": "<JSON string with the command field>"'
    if not author_marker or not output_marker or output.count(placeholder) != 1:
        raise ValueError("Unrecognized assistant-turn reviewer prompt")
    # These sections contain controller instructions, not the task or history.
    instructions = instructions.replace(
        "function.arguments is a JSON STRING containing exactly",
        "function.arguments is an OBJECT containing exactly")
    instructions = instructions.replace(
        "with JSON-string arguments containing exactly command",
        "with an arguments object containing exactly command")
    output = output.replace(placeholder, '"arguments": {"command": "<bash command>"}')
    if feedback_marker:
        previous, validation_marker, validation = feedback.rpartition(_VALIDATION_MARKER)
        if not validation_marker:
            raise ValueError("Reviewer correction is missing its validation feedback")
        validation = validation.replace(_LEGACY_ARGUMENT_FEEDBACK, _OBJECT_ARGUMENT_FEEDBACK)
        feedback = previous + validation_marker + validation
    adapted = before + author_marker + instructions + output_marker + output
    if feedback_marker:
        adapted += feedback_marker + feedback
    return "reviewer", adapted, {}


def response_text(kind, value):
    """Pass provider text through unchanged; never wrap prose or invent a plan."""
    if kind not in ("reviewer", "analyst"):
        raise ValueError("Unknown review response kind")
    choices = value.get("choices") or []
    if len(choices) != 1:
        raise ValueError("Expected exactly one review response")
    message = choices[0].get("message") or {}
    if message.get("tool_calls"):
        return ""
    text = message.get("content")
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("Review response content must be text")
    return text


def extract_branch_plan(text):
    """Validate the wire plan, then serialize command objects exactly once."""
    plan = _extract_fenced_plan(text)
    try:
        Draft202012Validator(REVIEW_SCHEMA).validate(plan)
    except ValidationError as error:
        path = ".".join(str(part) for part in error.absolute_path) or "<plan>"
        raise ValueError(f"Structured branch-plan field {path}: {error.message[:400]}") from error
    result = deepcopy(plan)
    for branch in result["branches"]:
        for call in branch["assistant_turn"]["tool_calls"]:
            arguments = call["function"]["arguments"]
            call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return result
