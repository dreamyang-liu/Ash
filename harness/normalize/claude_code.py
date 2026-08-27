"""claude-agent-sdk message objects -> journal events.

Input is the SDK's typed message stream (AssistantMessage / UserMessage /
ResultMessage / SystemMessage and their content blocks). We duck-type instead
of importing the SDK so this module stays import-light and testable with
fixtures; the SDK is only imported by harness/slots/claude_code.py.

Reference shapes (claude-agent-sdk 0.2.x):
    AssistantMessage(content=[TextBlock|ThinkingBlock|ToolUseBlock], model=...)
    UserMessage(content=[ToolResultBlock|TextBlock|str])
    ResultMessage(subtype, result, usage, total_cost_usd, session_id, is_error)
    SystemMessage(subtype="init", data={session_id, tools, capabilities, ...})
"""

from __future__ import annotations

from typing import List, Tuple

from harness.core.events import (
    AGENT_MESSAGE,
    AGENT_THINKING,
    RUN_RESULT,
    SESSION_REF,
    TOOL_FINISHED,
    TOOL_STARTED,
    TURN_COMPLETED,
    raw_event,
)

SLOT = "claude_code"

# Anthropic usage keys -> normalized Usage fields.
_USAGE_MAP = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_input_tokens": "cached_input_tokens",
    "cache_creation_input_tokens": "cache_creation_tokens",
}


def map_usage(native: object, cost_usd: float = 0.0) -> dict:
    """Anthropic ``usage`` block -> Usage dict payload."""
    payload = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
        "cost_usd": cost_usd or 0.0,
    }
    if isinstance(native, dict):
        for src, dst in _USAGE_MAP.items():
            value = native.get(src)
            if isinstance(value, (int, float)):
                payload[dst] = int(value)
    return payload


def _block_type(block: object) -> str:
    return type(block).__name__


def _text_of(block: object) -> str:
    text = getattr(block, "text", None)
    return text if isinstance(text, str) else ""


def _tool_result_content(block: object) -> str:
    content = getattr(block, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(_text_of(item) or str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def normalize(message: object) -> List[Tuple[str, dict]]:
    """Map one SDK message to journal events."""
    kind = type(message).__name__
    out: List[Tuple[str, dict]] = []

    if kind == "AssistantMessage":
        for block in getattr(message, "content", None) or []:
            btype = _block_type(block)
            if btype == "TextBlock":
                text = _text_of(block)
                if text:
                    out.append((AGENT_MESSAGE, {"text": text}))
            elif btype == "ThinkingBlock":
                text = _text_of(block) or getattr(block, "thinking", "") or ""
                if text:
                    out.append((AGENT_THINKING, {"text": text}))
            elif btype == "ToolUseBlock":
                out.append(
                    (
                        TOOL_STARTED,
                        {
                            "call_id": getattr(block, "id", None),
                            "name": getattr(block, "name", None),
                            "args": getattr(block, "input", None) or {},
                        },
                    )
                )
            else:
                out.append((raw_event(SLOT), {"block": btype, "repr": repr(block)[:2000]}))
        return out

    if kind == "UserMessage":
        # Tool results come back as a synthetic user turn.
        for block in getattr(message, "content", None) or []:
            if _block_type(block) == "ToolResultBlock":
                is_error = bool(getattr(block, "is_error", False))
                out.append(
                    (
                        TOOL_FINISHED,
                        {
                            "call_id": getattr(block, "tool_use_id", None),
                            "status": "error" if is_error else "ok",
                            "output": _tool_result_content(block),
                        },
                    )
                )
        return out

    if kind == "ResultMessage":
        usage = map_usage(
            getattr(message, "usage", None), getattr(message, "total_cost_usd", 0.0) or 0.0
        )
        out.append((TURN_COMPLETED, {"usage": usage}))
        session_id = getattr(message, "session_id", None)
        if session_id:
            out.append((SESSION_REF, {"native_session_id": session_id}))
        result_text = getattr(message, "result", None)
        out.append(
            (
                RUN_RESULT,
                {
                    "text": result_text if isinstance(result_text, str) else "",
                    "native": {
                        "subtype": getattr(message, "subtype", None),
                        "is_error": bool(getattr(message, "is_error", False)),
                        "num_turns": getattr(message, "num_turns", None),
                    },
                },
            )
        )
        return out

    if kind == "SystemMessage":
        data = getattr(message, "data", None) or {}
        if getattr(message, "subtype", None) == "init":
            out.append(
                (
                    SESSION_REF,
                    {
                        "native_session_id": data.get("session_id"),
                        "capabilities": data.get("capabilities"),
                        "tools": data.get("tools"),
                        "model": data.get("model"),
                    },
                )
            )
            return out
        out.append((raw_event(SLOT), {"system": getattr(message, "subtype", None), "data": data}))
        return out

    out.append((raw_event(SLOT), {"message": kind, "repr": repr(message)[:2000]}))
    return out
