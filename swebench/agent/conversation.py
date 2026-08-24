"""Conversation state — the single source of truth for the running dialogue.

Owns both the model-facing `messages` (OpenAI format) and the saved
`trajectory`, keeping them in sync so the loop never updates two records by
hand, and tracks how many trailing assistant turns made no tool call.
"""

from ..models import Trajectory


def plain_tool_calls(tool_calls) -> list[dict]:
    """Tool calls as JSON-safe dicts.

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
            plain.append(call)
        elif hasattr(call, "model_dump"):
            plain.append(call.model_dump())
        elif hasattr(call, "dict"):
            plain.append(call.dict())
        else:
            function = getattr(call, "function", None)
            plain.append({
                "id": getattr(call, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(function, "name", ""),
                    "arguments": getattr(function, "arguments", ""),
                },
            })
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
            msg["tool_calls"] = message.tool_calls
        # Preserve thinking_blocks for Anthropic extended thinking + tool use
        if thinking := getattr(message, "thinking_blocks", None):
            msg["thinking_blocks"] = thinking
        self.messages.append(msg)
        self.trajectory.add_message(
            "assistant", message.content or "",
            **({"tool_calls": plain_tool_calls(message.tool_calls)}
               if message.tool_calls else {}))
        self.consecutive_no_tool = 0 if message.tool_calls else self.consecutive_no_tool + 1

    def add_tool_result(self, tool_call_id: str, content: str, **meta) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        self.trajectory.add_message("tool_result", content, **meta)

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
