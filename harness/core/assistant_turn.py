"""Validate an ordinary assistant bash turn without executing or rewriting it."""

from copy import deepcopy
from collections.abc import Sequence
import json


def validate_assistant_turn(message: dict, *, history: Sequence[dict] = ()) -> dict:
    """Return an owned copy; observations and provider thinking are never supplied."""
    if not isinstance(message, dict) or set(message) != {"role", "content", "tool_calls"}:
        raise ValueError("assistant_turn requires exactly role, content and tool_calls")
    if message["role"] != "assistant" or not isinstance(message["content"], str) or not message["content"].strip():
        raise ValueError("assistant_turn requires assistant role and non-empty ordinary content")
    calls = message["tool_calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("assistant_turn requires at least one bash tool call")
    used = {call["id"] for entry in history for call in entry.get("tool_calls") or []}
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise ValueError("assistant_turn tool calls require id, type and function")
        identifier = call["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in used:
            raise ValueError("assistant_turn tool call ids must be non-empty and unique across history")
        used.add(identifier)
        function = call["function"]
        if (call["type"] != "function" or not isinstance(function, dict)
                or set(function) != {"name", "arguments"} or function["name"] != "bash"
                or not isinstance(function["arguments"], str)):
            raise ValueError("assistant_turn supports only bash with JSON-string arguments")
        try:
            arguments = json.loads(function["arguments"])
        except ValueError as error:
            raise ValueError("assistant_turn bash arguments are not valid JSON") from error
        if (not isinstance(arguments, dict) or set(arguments) != {"command"}
                or not isinstance(arguments["command"], str) or not arguments["command"].strip()):
            raise ValueError("assistant_turn bash arguments require exactly one non-empty string command")
    return deepcopy(message)
