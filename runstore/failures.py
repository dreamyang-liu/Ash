"""Conservative retry classification; benchmark failure is never infrastructure."""

from __future__ import annotations

import re


def failure_kind(result: dict) -> str | None:
    if result.get("status") == "completed":
        return None
    message = str(result.get("error") or "").lower()
    if any(marker in message for marker in ("checksum changed", "revision differs", "profile changed")):
        return "configuration"
    if result.get("status") in {"timeout", "step_limit", "cost_limit", "budget_exhausted"}:
        return "actor"
    if any(marker in message for marker in ("budget exhausted", "cost limit", "step limit", "maximum turns")):
        return "actor"
    if re.search(r"\b(?:502|503|504)\b", message) or any(marker in message for marker in (
            "transportclosederror", "connection reset", "connection refused", "stream stalled",
            "execution_uncertain", "tool execution/capture did not settle", "readtimeout")):
        return "infrastructure"
    return result.get("failure_kind") or "actor"
