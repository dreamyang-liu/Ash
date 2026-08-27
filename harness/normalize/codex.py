"""``codex exec --json`` JSONL events -> journal events.

Verified against codex-cli 0.145.0. The stdout event stream (NOT the on-disk
rollout under ~/.codex/sessions, which carries no version field and is
explicitly unstable) is the contract we build on.

Observed shapes::

    {"type":"thread.started","thread_id":"..."}
    {"type":"turn.started", ...}
    {"type":"item.started"|"item.updated"|"item.completed","item":{...}}
    {"type":"turn.completed","usage":{"input_tokens":N,
        "cached_input_tokens":N,"output_tokens":N,"reasoning_output_tokens":N}}
    {"type":"turn.failed","error":{...}}
    {"type":"error","message":"..."}

``item`` kinds seen: ``agent_message`` (text), ``reasoning``, ``command_execution``
(command/exit_code/aggregated_output), ``mcp_tool_call`` (server/tool/arguments/
result), ``file_change``, ``todo_list``, ``web_search``.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from harness.core.events import (
    AGENT_ERROR,
    AGENT_MESSAGE,
    AGENT_THINKING,
    RUN_RESULT,
    SESSION_REF,
    TOOL_FINISHED,
    TOOL_STARTED,
    TURN_COMPLETED,
    TURN_STARTED,
    raw_event,
)

SLOT = "codex"

_USAGE_FIELDS = {
    "input_tokens": "input_tokens",
    "cached_input_tokens": "cached_input_tokens",
    "output_tokens": "output_tokens",
    "reasoning_output_tokens": "reasoning_output_tokens",
}


def map_usage(native: object, cost_usd: float = 0.0) -> dict:
    payload = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
        "cost_usd": cost_usd or 0.0,
    }
    if isinstance(native, dict):
        for src, dst in _USAGE_FIELDS.items():
            value = native.get(src)
            if isinstance(value, (int, float)):
                payload[dst] = int(value)
    return payload


def _tool_name(item: dict) -> str:
    kind = item.get("item_type") or item.get("type") or ""
    if kind == "command_execution":
        return "shell"
    if kind == "mcp_tool_call":
        server = item.get("server") or item.get("server_name") or ""
        tool = item.get("tool") or item.get("tool_name") or ""
        return ("%s__%s" % (server, tool)).strip("_") or "mcp"
    return kind or "unknown"


def _tool_args(item: dict) -> dict:
    kind = item.get("item_type") or item.get("type") or ""
    if kind == "command_execution":
        return {"command": item.get("command"), "cwd": item.get("cwd")}
    if kind == "mcp_tool_call":
        args = item.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"raw": args}
        return args if isinstance(args, dict) else {"raw": args}
    return {k: v for k, v in item.items() if k not in ("id", "item_type", "type")}


def _output_of(item: dict) -> str:
    for key in ("aggregated_output", "output", "result", "text", "content"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        if value not in (None, "", [], {}):
            return json.dumps(value, ensure_ascii=False, default=str)
    return ""


def normalize(event: dict) -> List[Tuple[str, dict]]:
    if not isinstance(event, dict):
        return [(raw_event(SLOT), {"repr": repr(event)[:2000]})]

    etype = event.get("type") or ""
    out: List[Tuple[str, dict]] = []

    if etype == "thread.started":
        thread_id = event.get("thread_id") or event.get("id")
        return [(SESSION_REF, {"native_session_id": thread_id})]

    if etype in ("item.started", "item.completed", "item.updated"):
        item = event.get("item") or {}
        kind = item.get("item_type") or item.get("type") or ""
        call_id = item.get("id")

        if kind == "agent_message":
            if etype == "item.completed":
                text = item.get("text") or item.get("content") or ""
                if text:
                    out.append((AGENT_MESSAGE, {"text": text, "part_id": call_id}))
            return out

        if kind == "reasoning":
            if etype == "item.completed":
                text = item.get("text") or item.get("summary") or ""
                if text:
                    out.append((AGENT_THINKING, {"text": text, "part_id": call_id}))
            return out

        if kind in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
            if etype == "item.started":
                out.append(
                    (
                        TOOL_STARTED,
                        {"call_id": call_id, "name": _tool_name(item), "args": _tool_args(item)},
                    )
                )
            elif etype == "item.completed":
                exit_code = item.get("exit_code")
                status = item.get("status")
                failed = (isinstance(exit_code, int) and exit_code != 0) or status in (
                    "failed",
                    "error",
                )
                payload = {
                    "call_id": call_id,
                    "name": _tool_name(item),
                    "status": "error" if failed else "ok",
                    "output": _output_of(item),
                }
                if exit_code is not None:
                    payload["exit_code"] = exit_code
                out.append((TOOL_FINISHED, payload))
            return out

        out.append((raw_event(SLOT), {"event": etype, "item": item}))
        return out

    if etype == "turn.completed":
        return [(TURN_COMPLETED, {"usage": map_usage(event.get("usage"))})]

    if etype == "turn.failed":
        error = event.get("error") or {}
        message = error.get("message") if isinstance(error, dict) else str(error)
        return [(AGENT_ERROR, {"message": message or "turn failed", "native": error})]

    if etype == "error":
        return [(AGENT_ERROR, {"message": event.get("message") or "error", "native": event})]

    if etype == "turn.started":
        return [(TURN_STARTED, {})]

    if etype == "thread.finished":
        return [(raw_event(SLOT), {"event": etype, "data": event})]

    return [(raw_event(SLOT), {"event": etype or "unknown", "data": event})]


def final_text(events: List[dict]) -> str:
    """Last agent_message text in a captured stream (codex has no result field)."""
    for event in reversed(events):
        if event.get("type") == RUN_RESULT:
            text = event.get("text")
            if text:
                return text
        if event.get("type") == AGENT_MESSAGE:
            text = event.get("text")
            if text:
                return text
    return ""
