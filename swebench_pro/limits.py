"""Actor tool budgets adapted from the pinned official SWE-agent configuration."""

from __future__ import annotations

import threading
import time
from typing import Any

from harness.execution.interceptors import MutationTracker
from harness.execution.pipeline import CallContext, Rewrite


class OfficialToolBudget(MutationTracker):
    fail_mode = "closed"

    def __init__(self, command_timeout: int = 450, total_seconds: int = 1800,
                 consecutive_timeouts: int = 3, emit: Any = None):
        super().__init__()
        self.command_timeout = command_timeout
        self.total_seconds = total_seconds
        self.consecutive_limit = consecutive_timeouts
        self.emit = emit or (lambda **values: None)
        self.elapsed = 0.0
        self.consecutive = 0
        self.lock = threading.Lock()

    @property
    def exhausted(self) -> str | None:
        with self.lock:
            if self.elapsed > self.total_seconds:
                return "official total tool execution budget exceeded"
            if self.consecutive >= self.consecutive_limit:
                return "official consecutive tool timeout limit exceeded"
        return None

    def before(self, ctx: CallContext) -> Any:
        result = super().before(ctx)
        if ctx.tool_name != "shell":
            return result
        requested = ctx.args.get("timeout", self.command_timeout)
        if not isinstance(requested, (int, float)) or isinstance(requested, bool) or requested <= 0:
            requested = self.command_timeout
        effective = min(int(requested), self.command_timeout)
        ctx.metadata["pro_tool_started"] = time.monotonic()
        self.emit(requested_timeout=ctx.args.get("timeout"), effective_timeout=effective)
        return Rewrite({**ctx.args, "timeout": effective})

    def after(self, ctx: CallContext, result: Any) -> Any:
        started = ctx.metadata.get("pro_tool_started")
        if started is not None:
            outcome = getattr(result, "outcome", None)
            with self.lock:
                self.elapsed += time.monotonic() - started
                self.consecutive = self.consecutive + 1 if getattr(outcome, "timed_out", False) else 0
                state = {"tool_seconds": self.elapsed, "consecutive_timeouts": self.consecutive}
            self.emit(**state)
        return result
