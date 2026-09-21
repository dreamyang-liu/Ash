"""Validate an ordinary assistant bash turn without executing or rewriting it."""

from copy import deepcopy
from collections.abc import Sequence
import json
from harness.core.mini_tools import validate_mini_tools


def validate_assistant_turn(message: dict, *, history: Sequence[dict] = (),
                            tools: list[dict] | None = None) -> dict:
    """Return an owned copy; observations and provider thinking are never supplied."""
    if tools is not None:
        tools = validate_mini_tools(tools)
    if not isinstance(message, dict) or set(message) != {"role", "content", "tool_calls"}:
        raise ValueError("assistant_turn requires exactly role, content and tool_calls")
    if message["role"] != "assistant" or not isinstance(message["content"], str) or not message["content"].strip():
        raise ValueError("assistant_turn requires assistant role and non-empty ordinary content")
    calls = message["tool_calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("assistant_turn requires at least one bash tool call")
    used = {call["id"] for entry in history for call in entry.get("tool_calls") or []}
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise ValueError("assistant_turn tool calls require id, type and function")
        identifier = call["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in used:
            raise ValueError("assistant_turn tool call ids must be non-empty and unique across history")
        used.add(identifier)
        function = call["function"]
        if (call["type"] != "function" or not isinstance(function, dict)
                or set(function) != {"name", "arguments"}
                or not isinstance(function["arguments"], str)):
            raise ValueError("assistant_turn supports only bash with JSON-string arguments")
        if function["name"] != "bash":
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) has forbidden tool name "
                f"{function['name']!r}; only bash is allowed. Return function.name='bash' "
                "with JSON-string arguments containing exactly command. "
                "No calls from this plan have executed."
            )
        try:
            arguments = json.loads(function["arguments"])
        except ValueError as error:
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) bash arguments are not valid JSON: {error}"
            ) from error
        if (not isinstance(arguments, dict) or set(arguments) != {"command"}
                or not isinstance(arguments["command"], str) or not arguments["command"].strip()):
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) bash arguments require exactly one non-empty string command"
            )
        if tools is not None:
            from jsonschema import Draft202012Validator, ValidationError

            try:
                Draft202012Validator(tools[0]["function"]["parameters"]).validate(arguments)
            except ValidationError as error:
                raise ValueError(f"assistant_turn does not match the recorded actor tool schema: {error.message}") from error
    return deepcopy(message)
