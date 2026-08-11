from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    output: str
    is_error: bool
    #: Raw event dicts the runtime attached to this response, if any.
    notifications: list[dict[str, Any]] = field(default_factory=list)

    @property
    def events(self) -> list:
        """The attached events, typed.

        Kept as a property over the raw list so the wire format stays visible
        while callers work with parsed values.
        """
        from .events import parse_events

        return parse_events(self.notifications)
