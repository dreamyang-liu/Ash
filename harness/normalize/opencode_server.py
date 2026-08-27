"""opencode server history/messages -> journal events.

Shapes verified against opencode 1.18.5 (``GET /session/{id}/message``)::

    [{"info": {"role": "user"|"assistant", "id": "msg_…", "sessionID": "ses_…",
               "tokens": {"total", "input", "output", "reasoning",
                          "cache": {"read", "write"}},
               "cost": 0.031, "modelID": …, "providerID": …, "finish": "stop"},
      "parts": [{"type": "step-start"|"text"|"tool"|"reasoning"|"step-finish",
                 "id": "prt_…", "messageID": "msg_…", …}]}]

Two things this format gets right that the CLI's stdout did not, and which the
mapping preserves:

- **``msg_``/``prt_`` ids are addressable.** ``POST /session/{id}/fork
  {messageID}`` branches at one of them, so a journal that records them lets
  ``fork_plan`` name a conversation-side branch point instead of only "the tip".
  Every event therefore carries ``part_id`` and ``message_id``.
- **``tokens.cache`` is split into read/write.** ``tokens.total`` counts cache
  writes, so summing it across turns double counts; the dimensions are mapped
  separately and total is ignored.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from harness.core.events import (AGENT_MESSAGE, AGENT_THINKING, TOOL_FINISHED,
                                 TOOL_STARTED, TURN_COMPLETED, Usage,
                                 raw_event)

SLOT = "opencode"


def map_usage(info: dict) -> dict:
    """``info.tokens`` + ``info.cost`` -> Usage payload."""
    tokens = (info or {}).get("tokens") or {}
    cache = tokens.get("cache") or {}
    return {
        "input_tokens": int(tokens.get("input") or 0),
        "output_tokens": int(tokens.get("output") or 0),
        "cached_input_tokens": int(cache.get("read") or 0),
        "cache_creation_tokens": int(cache.get("write") or 0),
        "reasoning_output_tokens": int(tokens.get("reasoning") or 0),
        "cost_usd": float((info or {}).get("cost") or 0.0),
    }


def _tool_events(part: dict, message_id: Optional[str]) -> List[Tuple[str, dict]]:
    """A ``tool`` part carries the whole call: state.input + state.output."""
    state = part.get("state") or {}
    call_id = part.get("callID") or part.get("id")
    name = part.get("tool") or "unknown"
    status = state.get("status")
    out: List[Tuple[str, dict]] = [
        (TOOL_STARTED, {
            "call_id": call_id,
            "name": name,
            "args": state.get("input") or {},
            "part_id": part.get("id"),
            "message_id": message_id,
        })
    ]
    if status in ("completed", "error"):
        output = state.get("output")
        if not isinstance(output, str):
            output = "" if output is None else str(output)
        out.append((TOOL_FINISHED, {
            "call_id": call_id,
            "name": name,
            "status": "error" if status == "error" else "ok",
            "output": output,
            "part_id": part.get("id"),
            "message_id": message_id,
        }))
    return out


def normalize_part(part: dict, message_id: Optional[str] = None) -> List[Tuple[str, dict]]:
    """One message part -> journal events."""
    if not isinstance(part, dict):
        return [(raw_event(SLOT), {"repr": repr(part)[:2000]})]

    ptype = part.get("type")
    common = {"part_id": part.get("id"), "message_id": message_id}

    if ptype == "text":
        text = part.get("text") or ""
        return [(AGENT_MESSAGE, dict(common, text=text))] if text else []

    if ptype == "reasoning":
        text = part.get("text") or part.get("reasoning") or ""
        return [(AGENT_THINKING, dict(common, text=text))] if text else []

    if ptype == "tool":
        return _tool_events(part, message_id)

    if ptype in ("step-start", "step-finish"):
        # Turn bookkeeping; usage rides on the message, so nothing to add here.
        return []

    return [(raw_event(SLOT), {"part": ptype, "data": part})]


def normalize_message(message: dict) -> List[Tuple[str, dict]]:
    """One ``{info, parts}`` entry -> journal events."""
    info = message.get("info") or {}
    message_id = info.get("id")
    out: List[Tuple[str, dict]] = []
    for part in message.get("parts") or []:
        out.extend(normalize_part(part, message_id))
    if info.get("role") == "assistant":
        out.append((TURN_COMPLETED, {
            "usage": map_usage(info),
            "message_id": message_id,
            "finish": info.get("finish"),
            "model": info.get("modelID"),
        }))
    return out


def emit_history(messages: List[dict], journal, since: Optional[str] = None) -> dict:
    """Journal a whole history, returning accumulated usage.

    ``since``: skip messages up to and including this id, for a driver that has
    already journalled the earlier ones (a resumed session replays its history).
    """
    total = Usage()
    skipping = since is not None
    for message in messages or []:
        info = message.get("info") or {}
        if skipping:
            if info.get("id") == since:
                skipping = False
            continue
        for event_type, payload in normalize_message(message):
            journal.emit(event_type, **payload)
            if event_type == TURN_COMPLETED:
                total.add_dict(payload.get("usage") or {})
    return total.as_dict()


def final_text(messages: List[dict]) -> str:
    """Last assistant text in a history."""
    for message in reversed(messages or []):
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        texts = [p.get("text") or "" for p in message.get("parts") or []
                 if p.get("type") == "text"]
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined
    return ""


def reply_text(reply: Optional[dict]) -> str:
    """Text of the single message a prompt call returns."""
    if not isinstance(reply, dict):
        return ""
    texts = [p.get("text") or "" for p in reply.get("parts") or []
             if isinstance(p, dict) and p.get("type") == "text"]
    return "\n".join(t for t in texts if t).strip()


def message_ids(messages: List[dict]) -> List[str]:
    """Addressable branch points, oldest first (what fork {messageID} takes)."""
    return [(m.get("info") or {}).get("id") for m in messages or []
            if (m.get("info") or {}).get("id")]
