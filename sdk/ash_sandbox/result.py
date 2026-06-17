from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    output: str
    is_error: bool
    notifications: list[dict[str, Any]] = field(default_factory=list)
