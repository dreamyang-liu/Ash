"""Agent slot drivers.

Registry maps a slot name to its class. Import is lazy so that a missing
optional dependency (e.g. claude-agent-sdk) only breaks the slot that needs it,
not the whole harness.
"""

from __future__ import annotations

from typing import Dict, Type

from harness.core.slot import AgentSlot

_REGISTRY: Dict[str, str] = {
    "claude-code": "harness.slots.claude_code:ClaudeCodeSlot",
    "codex": "harness.slots.codex:CodexSlot",
    "opencode": "harness.slots.opencode:OpenCodeSlot",
}


def available() -> list:
    return sorted(_REGISTRY)


def load_slot(name: str) -> Type[AgentSlot]:
    """Resolve a slot name to its class, importing on demand."""
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            "unknown slot %r (available: %s)" % (name, ", ".join(available()))
        ) from None
    module_path, _, class_name = target.partition(":")
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)
