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
        if message.tool_calls:
            msg["tool_calls"] = message.tool_calls
        # Preserve thinking_blocks for Anthropic extended thinking + tool use
        if thinking := getattr(message, "thinking_blocks", None):
            msg["thinking_blocks"] = thinking
        self.messages.append(msg)
        self.trajectory.add_message("assistant", message.content or "")
        self.consecutive_no_tool = 0 if message.tool_calls else self.consecutive_no_tool + 1

    def add_tool_result(self, tool_call_id: str, content: str, **meta) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        self.trajectory.add_message("tool_result", content, **meta)

    def add_error(self, content: str) -> None:
        """Record an error in the trajectory only (not a real model message)."""
        self.trajectory.add_message("error", content)

    def append_to_last(self, suffix: str) -> None:
        """Append text to the last user/tool message (used by the budget hook)."""
        next(m for m in reversed(self.messages) if m["role"] in ("tool", "user"))["content"] += suffix
