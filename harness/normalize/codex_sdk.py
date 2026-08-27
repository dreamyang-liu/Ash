"""codex SDK notifications / TurnResult -> journal events.

Input is the typed stream from ``TurnHandle.stream()`` (openai-codex 0.147.0),
whose payloads are pydantic models generated from the app-server schema. We
duck-type rather than import them so this module stays a pure mapping table that
fixtures can drive, and so a payload the installed SDK does not model yet
(``UnknownNotification``) still lands in the journal.

Methods observed on a turn stream::

    turn/started · turn/completed
    item/started · item/completed
    item/agentMessage/delta · item/reasoning/textDelta · item/reasoning/…
    item/commandExecution/outputDelta · item/fileChange/patchUpdated
    thread/tokenUsage/updated · thread/compacted · error

Three shapes had to be read off live objects rather than the schema, and each one
silently produced wrong output before it was:

- **``payload.item`` is a RootModel.** The concrete item is ``item.root``, and its
  discriminator field is ``type`` (``agentMessage``, ``commandExecution``, …), not
  ``item_type``. Reading the wrapper made every item fall through to
  ``raw.codex`` -- a journal that looked populated but classified nothing.
- **``turn`` carries no usage.** ``TurnCompletedNotification.turn`` has
  ``status``/``items``/timings only; tokens arrive on
  ``thread/tokenUsage/updated`` as ``token_usage.{last,total}``. Taking usage from
  the turn yielded zeros for every run.
- **Cache writes are ``cache_write_input_tokens``**, not the Anthropic spelling.

Deltas are deliberately dropped: they duplicate the completed item and would
inflate a trajectory several-fold. ``item/completed`` is the record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from harness.core.events import (AGENT_ERROR, AGENT_MESSAGE, AGENT_THINKING,
                                 TOOL_FINISHED, TOOL_STARTED, TURN_COMPLETED,
                                 TURN_STARTED, USAGE_UPDATED, Usage, raw_event)

SLOT = "codex"

#: Streaming deltas -- ignored in favour of the completed item.
_DELTA_METHODS = {
    "item/agentMessage/delta",
    "item/reasoning/textDelta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded",
    "item/commandExecution/outputDelta",
    "item/fileChange/outputDelta",
    "item/fileChange/patchUpdated",
    "item/plan/delta",
    "item/mcpToolCall/progress",
    "item/commandExecution/terminalInteraction",
}

#: Item kinds that represent a tool call rather than model output.
_TOOL_ITEMS = {
    "commandExecution", "command_execution",
    "mcpToolCall", "mcp_tool_call",
    "fileChange", "file_change",
    "webSearch", "web_search",
    "dynamicToolCall", "dynamic_tool_call",
}


def _attr(obj: Any, *names, default=None):
    """First present attribute or mapping key among ``names``."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def _unwrap(item: Any) -> Any:
    """RootModel -> the concrete item. Reading the wrapper classifies nothing."""
    root = getattr(item, "root", None)
    return root if root is not None else item


def _plain(obj: Any) -> Any:
    """Best-effort JSON-safe view of a pydantic model or dataclass.

    Unwraps RootModels first: a bare ``model_dump`` of one renders as
    ``{"root": …}`` (or ``"root='/tmp/x'"`` via repr), which leaks the wrapper
    into journal payloads.
    """
    obj = _unwrap(obj)
    for method in ("model_dump", "dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                pass
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)[:2000]


def map_usage(native: Any, cost_usd: float = 0.0) -> dict:
    """Token counts -> Usage payload.

    Accepts either a ``token_usage`` block (whose ``total`` is cumulative for the
    thread) or a bare counts object. ``cached_input_tokens`` is a subset of input
    on this protocol, matching the journal's convention, so it is not subtracted.
    """
    payload = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
        "cost_usd": cost_usd or 0.0,
    }
    if native is None:
        return payload

    counts = _attr(native, "total", "last", default=native)
    mapping = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "cached_input_tokens": ("cached_input_tokens", "cachedInputTokens"),
        "cache_creation_tokens": ("cache_write_input_tokens", "cacheWriteInputTokens"),
        "reasoning_output_tokens": ("reasoning_output_tokens", "reasoningOutputTokens"),
    }
    for dst, names in mapping.items():
        value = _attr(counts, *names)
        if isinstance(value, (int, float)):
            payload[dst] = int(value)
    return payload


def _tool_name(item: Any, kind: str) -> str:
    if kind in ("commandExecution", "command_execution"):
        return "shell"
    if kind in ("mcpToolCall", "mcp_tool_call"):
        server = _attr(item, "server", "server_name", "serverName", default="") or ""
        tool = _attr(item, "tool", "tool_name", "toolName", default="") or ""
        return ("%s__%s" % (server, tool)).strip("_") or "mcp"
    if kind in ("fileChange", "file_change"):
        return "apply_patch"
    return kind or "unknown"


def _item_kind(item: Any) -> str:
    return str(_attr(_unwrap(item), "type", "item_type", "itemType", "kind", default="") or "")


def _tool_output(item: Any) -> str:
    for name in ("aggregated_output", "aggregatedOutput", "output", "result", "text"):
        value = _attr(item, name)
        if isinstance(value, str) and value:
            return value
        if value not in (None, "", [], {}):
            return str(_plain(value))
    return ""


def normalize(notification: Any) -> List[Tuple[str, dict]]:
    """One SDK notification -> journal events."""
    method = str(_attr(notification, "method", default="") or "")
    payload = _attr(notification, "payload", "params")

    if method in _DELTA_METHODS:
        return []

    if method == "turn/started":
        return [(TURN_STARTED, {"turn_id": _turn_id(payload)})]

    if method == "turn/completed":
        # No usage on the turn (see the module docstring): tokens arrive only on
        # thread/tokenUsage/updated, so this event marks the boundary and the
        # usage-bearing event is the one before it.
        turn = _attr(payload, "turn")
        return [(TURN_COMPLETED, {
            "turn_id": _attr(turn, "id") if turn is not None else _turn_id(payload),
            "status": str(_attr(turn, "status", default="") or "") or None,
            "boundary": True,
        })]

    if method == "thread/tokenUsage/updated":
        # Not a turn boundary: USAGE_UPDATED, or the checkpoint bridge would take
        # a snapshot on every token report. `total` is cumulative for the thread,
        # so a consumer replaces rather than adds.
        return [(USAGE_UPDATED, {
            "usage": map_usage(_attr(payload, "token_usage", "tokenUsage", "usage")),
            "turn_id": _attr(payload, "turn_id", "turnId"),
            "cumulative": True,
        })]

    if method in ("item/started", "item/completed"):
        item = _unwrap(_attr(payload, "item", default=payload))
        kind = _item_kind(item)
        item_id = _attr(item, "id")

        if kind in ("userMessage", "user_message"):
            return []          # our own prompt echoed back

        if kind in ("agentMessage", "agent_message"):
            if method == "item/completed":
                text = _attr(item, "text", "content", default="") or ""
                if text:
                    return [(AGENT_MESSAGE, {"text": text, "part_id": item_id})]
            return []

        if kind == "reasoning":
            if method == "item/completed":
                text = _attr(item, "text", "summary", default="") or ""
                if text:
                    return [(AGENT_THINKING, {"text": text, "part_id": item_id})]
            return []

        if kind in _TOOL_ITEMS:
            name = _tool_name(item, kind)
            if method == "item/started":
                return [(TOOL_STARTED, {
                    "call_id": item_id,
                    "name": name,
                    "args": _tool_args(item, kind),
                })]
            exit_code = _attr(item, "exit_code", "exitCode")
            status = str(_attr(item, "status", default="") or "")
            failed = (isinstance(exit_code, int) and exit_code != 0) or status in ("failed", "error")
            event = {
                "call_id": item_id,
                "name": name,
                "status": "error" if failed else "ok",
                "output": _tool_output(item),
            }
            if exit_code is not None:
                event["exit_code"] = exit_code
            return [(TOOL_FINISHED, event)]

        return [(raw_event(SLOT), {"method": method, "item": _plain(item)})]

    if method == "error":
        return [(AGENT_ERROR, {
            "message": str(_attr(payload, "message", default="error")),
            "native": _plain(payload),
        })]

    if method == "thread/compacted":
        return [(raw_event(SLOT), {"method": method, "data": _plain(payload)})]

    return [(raw_event(SLOT), {"method": method or "unknown", "data": _plain(payload)})]


def _turn_id(payload: Any) -> Optional[str]:
    turn = _attr(payload, "turn")
    if turn is not None:
        return _attr(turn, "id")
    return _attr(payload, "turn_id", "turnId", "id")


def _tool_args(item: Any, kind: str) -> Dict[str, Any]:
    if kind in ("commandExecution", "command_execution"):
        return {
            "command": _plain(_attr(item, "command")),
            "cwd": _plain(_attr(item, "cwd")),
        }
    if kind in ("mcpToolCall", "mcp_tool_call"):
        args = _attr(item, "arguments", "args")
        plain = _plain(args)
        return plain if isinstance(plain, dict) else {"raw": plain}
    plain = _plain(item)
    if isinstance(plain, dict):
        return {k: v for k, v in plain.items() if k not in ("id", "item_type", "type")}
    return {"raw": plain}


def stream_into(handle: Any, journal) -> dict:
    """Consume ``TurnHandle.stream()`` into the journal; return accumulated usage.

    The stream ends at ``turn/completed`` for this turn (the SDK breaks the loop
    itself), so this returns when the turn is done.

    Note this *exhausts* the stream. ``TurnHandle.run()`` consumes the same
    stream, so a driver must not call both -- see :func:`collect_and_journal`.
    """
    latest: Dict[str, Any] = {}
    for notification in handle.stream():
        for event_type, event in normalize(notification):
            journal.emit(event_type, **event)
            if event_type == USAGE_UPDATED and event.get("cumulative"):
                latest.update(event.get("usage") or {})
    return dict(latest)


def collect_and_journal(handle: Any, journal) -> Tuple[Any, dict]:
    """Journal the turn *and* return the SDK's own ``TurnResult``.

    A turn's notifications can only be consumed once, so this tees the iterator:
    each notification is journalled on its way to the SDK's collector, which
    assembles ``TurnResult`` (final_response, items, usage, status) exactly as
    ``TurnHandle.run()`` would. Calling ``run()`` after draining the stream
    yourself deadlocks -- it waits for notifications that were already taken.

    Falls back to a stream-only pass if the SDK's collector is not importable, so
    a version bump degrades to "journal is complete, TurnResult is None" rather
    than to a broken slot.
    """
    latest: Dict[str, Any] = {}

    def tee():
        for notification in handle.stream():
            for event_type, event in normalize(notification):
                journal.emit(event_type, **event)
                if event_type == USAGE_UPDATED and event.get("cumulative"):
                    # Thread-cumulative: keep the last one instead of summing,
                    # or a 3-turn run reports roughly 6x its real input tokens.
                    latest.update(event.get("usage") or {})
            yield notification

    try:
        from openai_codex._run import _collect_turn_result
    except ImportError:  # pragma: no cover - SDK internals moved
        for _ in tee():
            pass
        return None, dict(latest)

    result = _collect_turn_result(tee(), turn_id=getattr(handle, "id", None))
    return result, dict(latest)
