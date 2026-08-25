"""Conversation state — the single source of truth for the running dialogue.

Owns both the model-facing `messages` (OpenAI format) and the saved
`trajectory`, keeping them in sync so the loop never updates two records by
hand, and tracks how many trailing assistant turns made no tool call.
"""

import json

from ..models import Trajectory


def _repaired_arguments(arguments) -> tuple[str, bool]:
    """Arguments a provider will accept, and whether they had to be repaired.

    A model that hits its output limit mid-tool-call emits arguments that stop
    partway through their own JSON (`{"command": "write", "path": "x.c"` with
    no brace, no body). Keeping that verbatim poisons the conversation
    permanently: every later request fails converting the message, so the run
    dies with its budget unspent and no way to continue -- observed on a real
    marathon attempt, 37 messages in, while writing a large C file.
    """
    text = arguments if isinstance(arguments, str) else json.dumps(arguments or {})
    try:
        json.loads(text or "{}")
        return text or "{}", False
    except (json.JSONDecodeError, TypeError):
        return "{}", True


def plain_tool_calls(tool_calls) -> list[dict]:
    """Tool calls as JSON-safe dicts, with truncated arguments repaired.

    The provider hands back model objects; the trajectory is written with
    `json.dumps`, so they have to be flattened before they can be saved. They
    used to be dropped instead, which quietly cost two things: the actions an
    agent took were absent from the record it is replayed from, and any
    accounting of what the model saw understated it badly -- a `text_editor`
    write carries a whole file in its arguments.
    """
    plain = []
    for call in tool_calls or ():
        if isinstance(call, dict):
            flat = dict(call)
        elif hasattr(call, "model_dump"):
            flat = call.model_dump()
        elif hasattr(call, "dict"):
            flat = call.dict()
        else:
            function = getattr(call, "function", None)
            flat = {
                "id": getattr(call, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(function, "name", ""),
                    "arguments": getattr(function, "arguments", ""),
                },
            }
        function = dict(flat.get("function") or {})
        arguments, repaired = _repaired_arguments(function.get("arguments"))
        if repaired:
            function["arguments"] = arguments
            flat["function"] = function
        plain.append(flat)
    return plain


class Conversation:
    def __init__(self, trajectory: Trajectory):
        self.messages: list[dict] = []
        self.trajectory = trajectory
        self.consecutive_no_tool = 0

    def add_system(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})
        self.trajectory.add_message("system", content)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.trajectory.add_message("user", content)

    def add_assistant(self, message) -> None:
        """Append the assistant turn and update the no-tool counter."""
        msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            # Repaired, and in wire format: what goes back to the provider has
            # to be convertible, and a truncated tool call is not.
            msg["tool_calls"] = plain_tool_calls(message.tool_calls)
        # Preserve thinking_blocks for Anthropic extended thinking + tool use
        if thinking := getattr(message, "thinking_blocks", None):
            msg["thinking_blocks"] = thinking
        self.messages.append(msg)
        self.trajectory.add_message(
            "assistant", message.content or "",
            **({"tool_calls": msg["tool_calls"]} if message.tool_calls else {}))
        self.consecutive_no_tool = 0 if message.tool_calls else self.consecutive_no_tool + 1

    def add_tool_result(self, tool_call_id: str, content: str, **meta) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        # The id belongs in the record too, not only in the model-facing copy.
        # A replay re-feeds this transcript, and a provider rejects a tool
        # result it cannot match to a call; without the id, results and calls
        # can only be paired by guessing at their order.
        self.trajectory.add_message("tool_result", content,
                                    tool_call_id=tool_call_id, **meta)

    def add_error(self, content: str) -> None:
        """Record an error in the trajectory only (not a real model message)."""
        self.trajectory.add_message("error", content)

    def append_to_last(self, suffix: str) -> None:
        """Append text to the last user/tool message, in both records.

        Both, because this class exists so the loop never updates two records by
        hand -- and this method used to update one. The trajectory is what the
        eval layer reads afterwards, so text appended here was invisible to it:
        anything the model was asked mid-run left no trace of having been asked.
        """
        model_msg = next(m for m in reversed(self.messages)
                         if m["role"] in ("tool", "user"))
        model_msg["content"] += suffix
        # The trajectory records tool results under a different role name, so
        # match on the saved spelling rather than the model-facing one.
        saved = next((m for m in reversed(self.trajectory.messages)
                      if m["role"] in ("tool_result", "user")), None)
        if saved is not None:
            saved["content"] = (saved.get("content") or "") + suffix
