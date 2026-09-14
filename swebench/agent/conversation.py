"""Conversation state — the single source of truth for the running dialogue.

Owns both the model-facing `messages` (OpenAI format) and the saved
`trajectory`, keeping them in sync so the loop never updates two records by
hand, and tracks how many trailing assistant turns made no tool call.
"""

from ..models import Trajectory


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
        trajectory_extra = {}
        # Reasoning-aware OpenAI endpoints return this field separately from
        # ``content``.  It is part of the model-visible assistant message and
        # therefore must be replayed verbatim on the next turn.  Dropping it
        # makes a token/session server treat every later request as a new root
        # instead of an extension of the current trajectory.
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
            trajectory_extra["reasoning_content"] = reasoning_content
        if message.tool_calls:
            # Provider SDKs commonly return Pydantic/namespace objects.  A
            # durable trajectory key must be JSON-compatible, while retaining
            # the original OpenAI tool-call shape.
            msg["tool_calls"] = [_tool_call_dict(call) for call in message.tool_calls]
            trajectory_extra["tool_calls"] = msg["tool_calls"]
        # Preserve thinking_blocks for Anthropic extended thinking + tool use
        if thinking := getattr(message, "thinking_blocks", None):
            msg["thinking_blocks"] = thinking
            trajectory_extra["thinking_blocks"] = thinking
        self.messages.append(msg)
        self.trajectory.add_message("assistant", message.content or "", **trajectory_extra)
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


def _tool_call_dict(call) -> dict:
    """Convert an OpenAI/LiteLLM tool-call object to a plain mapping."""
    if isinstance(call, dict):
        value = dict(call)
    elif hasattr(call, "model_dump"):
        value = call.model_dump(mode="json")
    elif hasattr(call, "to_dict"):
        value = call.to_dict()
    else:
        value = {
            key: getattr(call, key)
            for key in ("id", "type", "function")
            if hasattr(call, key)
        }
    function = value.get("function")
    if function is not None and not isinstance(function, dict):
        if hasattr(function, "model_dump"):
            function = function.model_dump(mode="json")
        elif hasattr(function, "to_dict"):
            function = function.to_dict()
        else:
            function = {
                key: getattr(function, key)
                for key in ("name", "arguments")
                if hasattr(function, key)
            }
        value["function"] = function
    return value
