"""Typed view of the sandbox's event log.

The runtime reports asynchronous facts -- a background process exited, a file
changed, some agent ran a tool -- as JSON. Parsing them here means the field
names exist in one place on this side rather than in every caller's memory,
and a rename in the runtime breaks a test instead of quietly producing empty
strings.

Callers: sandbox.wait_for_events / poll_events, and typed access to the
events carried on a ToolResult.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """One asynchronous fact from inside the sandbox."""

    id: str
    kind: str
    #: The handle the fact is about: a pid, a file path, a URL, a query.
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    #: Which agent's action produced it, when it describes an action.
    origin: str = ""
    #: Set when the fact was addressed to one agent rather than anyone.
    agent_id: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Event:
        return cls(
            id=raw.get("id", ""),
            kind=raw.get("kind", ""),
            source=raw.get("source", ""),
            data=raw.get("data") or {},
            timestamp=raw.get("timestamp", ""),
            origin=raw.get("origin", ""),
            agent_id=raw.get("agent_id", ""),
        )

    @property
    def tool(self) -> str:
        """For a tool-call event, the tool that ran; otherwise empty.

        Kinds are namespaced as "tool:<name>", so this saves callers from
        slicing a convention they should not have to know.
        """
        return self.kind[5:] if self.kind.startswith("tool:") else ""


@dataclass(frozen=True)
class EventBatch:
    """What one query returned, including what it could not return.

    `missed` counts events that matched but left the log before this consumer
    saw them (TTL expiry or a budget trim). It belongs in the result rather
    than a log line: a consumer reasoning about completeness needs to know it
    lost something.
    """

    events: list[Event] = field(default_factory=list)
    timed_out: bool = False
    missed: int = 0

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __bool__(self) -> bool:
        return bool(self.events)

    def of_kind(self, *kinds: str) -> list[Event]:
        """The subset matching any of these kinds."""
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]


def parse_events(raw: list[dict] | None) -> list[Event]:
    """Convert the runtime's event dicts into Events."""
    return [Event.from_dict(item) for item in (raw or [])]


def parse_batch(tool_output: str) -> EventBatch:
    """Parse a wait_for_events payload.

    The runtime returns it as JSON text inside the tool result, since a tool
    result is text; unpacking it is this layer's job, not every caller's.
    """
    try:
        payload = json.loads(tool_output or "{}")
    except json.JSONDecodeError:
        return EventBatch()
    return EventBatch(
        events=parse_events(payload.get("events")),
        timed_out=bool(payload.get("timed_out")),
        missed=int(payload.get("missed") or 0),
    )
