"""``opencode run --format json`` JSONL events -> journal events.

Verified empirically against opencode 1.18.5 (see tests/fixtures/opencode_*.jsonl).

Envelope::

    {"type": <event>, "timestamp": <epoch ms>, "sessionID": "ses_...",
     "part": {"id":"prt_...","messageID":"msg_...","sessionID":...,"type":<part>, ...}}

Observed (event type, part.type) pairs:
    step_start   / step-start
    text         / text          part.text, part.time{start,end}
    tool_use     / tool          part.tool, part.callID,
                                 part.state{status,input,output,metadata,title,time}
    step_finish  / step-finish   part.reason, part.cost,
                                 part.tokens{total,input,output,reasoning,
                                             cache{write,read}}

Note that opencode reports a *single* tool event per call carrying the terminal
state (status="completed"/"error"), not separate start/finish events. We still
emit the normalized pair so downstream consumers (and ATIF export) see the same
shape as other slots; ``synthetic: true`` marks the derived start event.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from harness.core.events import (
    AGENT_ERROR,
    AGENT_MESSAGE,
    AGENT_THINKING,
    SESSION_REF,
    TOOL_FINISHED,
    TOOL_STARTED,
    TURN_COMPLETED,
    TURN_STARTED,
    raw_event,
)

SLOT = "opencode"


def map_usage(tokens: object, cost_usd: float = 0.0) -> dict:
    """opencode ``part.tokens`` -> Usage dict payload.

    ``tokens.total`` is ignored on purpose: it double counts cache writes, and
    the orchestrator sums the individual dimensions itself.
    """
    payload = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
        "cost_usd": cost_usd or 0.0,
    }
    if isinstance(tokens, dict):
        for src, dst in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("reasoning", "reasoning_output_tokens"),
        ):
            value = tokens.get(src)
            if isinstance(value, (int, float)):
                payload[dst] = int(value)
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            read = cache.get("read")
            write = cache.get("write")
            if isinstance(read, (int, float)):
                payload["cached_input_tokens"] = int(read)
            if isinstance(write, (int, float)):
                payload["cache_creation_tokens"] = int(write)
    return payload


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def normalize(event: dict) -> List[Tuple[str, dict]]:
    if not isinstance(event, dict):
        return [(raw_event(SLOT), {"repr": repr(event)[:2000]})]

    etype = event.get("type") or ""
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    out: List[Tuple[str, dict]] = []

    if etype == "text":
        text = part.get("text") or ""
        if text:
            out.append((AGENT_MESSAGE, {"text": text, "part_id": part.get("id")}))
        return out

    if etype in ("reasoning", "thinking"):
        text = part.get("text") or part.get("thinking") or ""
        if text:
            out.append((AGENT_THINKING, {"text": text, "part_id": part.get("id")}))
        return out

    if etype == "tool_use":
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        call_id = part.get("callID") or part.get("id")
        name = part.get("tool") or "unknown"
        status = state.get("status") or ""

        out.append(
            (
                TOOL_STARTED,
                {
                    "call_id": call_id,
                    "name": name,
                    "args": state.get("input") or {},
                    "synthetic": True,
                },
            )
        )
        if status in ("completed", "error", "failed"):
            output = state.get("output")
            if output in (None, "") and state.get("error") is not None:
                output = state.get("error")
            payload = {
                "call_id": call_id,
                "name": name,
                "status": "ok" if status == "completed" else "error",
                "output": _stringify(output),
            }
            time = state.get("time")
            if isinstance(time, dict) and isinstance(time.get("start"), (int, float)):
                end = time.get("end")
                if isinstance(end, (int, float)):
                    payload["duration_ms"] = int(end - time["start"])
            out.append((TOOL_FINISHED, payload))
        return out

    if etype == "step_finish":
        usage = map_usage(part.get("tokens"), part.get("cost") or 0.0)
        payload = {"usage": usage}
        reason = part.get("reason")
        if reason:
            payload["reason"] = reason
        out.append((TURN_COMPLETED, payload))
        return out

    if etype == "step_start":
        session_id = event.get("sessionID")
        if session_id:
            out.append((SESSION_REF, {"native_session_id": session_id}))
        out.append((TURN_STARTED, {}))
        return out

    if etype in ("error", "session_error"):
        message = event.get("message") or _stringify(part) or "error"
        return [(AGENT_ERROR, {"message": message, "native": event})]

    return [(raw_event(SLOT), {"event": etype or "unknown", "data": event})]
