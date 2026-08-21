"""Core types for SWE-bench benchmark."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolResult:
    """Result of one tool call.

    ``output`` and ``error`` are alternatives, not two views of one message:
    ``output`` is what the tool produced, ``error`` is the tool refusing to
    produce anything (bad arguments, no sandbox, a transport failure). A command
    that ran and exited non-zero has real output and no error -- ``success``
    already says it failed.

    Storing one message in both fields costs twice, because the agent loop
    renders a failure as ``f"Error: {error}\\n{output}"``: the model reads the
    same bytes twice and pays for both.
    """
    success: bool
    output: str
    error: Optional[str] = None
    #: What running a command produced, when the tool ran one and had something
    #: to report beyond its stdout (``ash_sandbox.CommandOutcome`` fields:
    #: ``exit_code``, ``stdout``, ``stderr``, ``timed_out``, byte counts,
    #: truncation flags). ``None`` for a plain success, a refusal, or a tool that
    #: runs no command. Interceptors read it to compose what the model sees; see
    #: ``agent/interceptors.py``.
    outcome: Optional["CommandOutcome"] = None

    @classmethod
    def from_sdk(cls, result) -> "ToolResult":
        """Convert an ``ash_sandbox.ToolResult``.

        The text slot is one string plus an ``is_error`` flag, so a failed call
        gives us no way to know whether that string is output or a refusal. Treat
        it as output: it is what the runtime chose to show, and ``success=False``
        carries the failure without duplicating the text.

        A command's structured outcome rides along unrendered — turning it into
        prose is presentation, and presentation belongs to the interceptors
        (docs/ARCHITECTURE.md, ADR-2), not to a type conversion.
        """
        return cls(success=not result.is_error, output=result.output,
                   outcome=CommandOutcome.from_sdk(result))


@dataclass
class CommandOutcome:
    """What running a command produced — the runtime's shared schema.

    Reported the same way by ``shell`` in the foreground and ``process read`` on
    a background pid, so code that reasons about a command works with either.

    ``exit_code`` is ``None`` when unknown: still running, or never run. Test it
    with ``is None``; ``0`` is a real answer.
    """
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    running: bool = False
    timed_out: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    @classmethod
    def from_sdk(cls, result) -> "Optional[CommandOutcome]":
        """Lift an ``ash_sandbox.ToolResult``'s outcome fields, or None."""
        if getattr(result, "stdout", None) is None:
            return None          # no command outcome in this response
        return cls(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            running=result.running,
            timed_out=result.timed_out,
            stdout_bytes=result.stdout_bytes or 0,
            stderr_bytes=result.stderr_bytes or 0,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )


@dataclass
class AgentConfig:
    """Configuration for the ash agent."""
    model: str = "anthropic/claude-sonnet-4-5-20250929"
    api_base: Optional[str] = None  # e.g. "http://localhost:30000/v1" for local SGLang
    api_key: Optional[str] = None   # API key / bearer token for the provider
    max_tokens: int = 16384
    step_limit: int = 250
    cost_limit: float = 3.0
    temperature: Optional[float] = None  # None = use model default
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high" | "none" (adaptive thinking)
    prompt_cache: bool = True  # Enable prompt caching for Anthropic/Bedrock models
    tools: str = "default"  # "default" (structured tools) | "bash_only" (single bash tool)
    custom_tools_dir: Optional[str] = None  # manifest dir; None = default configs/custom_tools
    workdir: str = "/testbed"
    system_template: Optional[str] = None  # Jinja2 template for system prompt (overrides AGENT.md)
    instance_template: Optional[str] = None  # Jinja2 template for instance/task message


@dataclass
class CostTracker:
    """Tracks LLM API cost using litellm's per-model pricing."""
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    api_calls: int = 0

    def update(self, response: Any):
        self.api_calls += 1
        if hasattr(response, "usage") and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0
            # Track cache usage from Anthropic/Bedrock responses
            usage = response.usage
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        try:
            from litellm import completion_cost
            self.total_cost += completion_cost(completion_response=response)
        except Exception:
            pass

    def budget_warning(self, step_limit: int, cost_limit: float) -> Optional[str]:
        """Return a one-time warning string when only a few steps remain, else None."""
        if self.api_calls < 3:
            return None
        avg = self.total_cost / self.api_calls
        remaining_steps = step_limit - self.api_calls
        est = min(remaining_steps, int((cost_limit - self.total_cost) / avg)) if avg > 0 else remaining_steps
        if est > 4 and remaining_steps > 4:
            return None
        return (
            f"\n\n[Budget Warning] ~{est} steps remaining "
            f"({self.api_calls}/{step_limit} steps, ${self.total_cost:.2f}/${cost_limit:.2f} budget). "
            f"Finalize your fix now: run tests and stop. If tests fail, revert and submit your best attempt."
        )

    def to_dict(self) -> dict:
        d = {
            "api_calls": self.api_calls,
            "instance_cost": round(self.total_cost, 4),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }
        if self.cache_read_tokens or self.cache_write_tokens:
            d["cache_read_tokens"] = self.cache_read_tokens
            d["cache_write_tokens"] = self.cache_write_tokens
        return d


@dataclass
class Trajectory:
    """One agent run, recorded: the messages, what it cost, how it ended.

    ``instance_id`` names the run rather than binding it to a benchmark -- a
    harness sets whatever it uses to tell runs apart (``django__django-10880``,
    ``iid-worker-2``). A benchmark prediction is a different thing and belongs to
    whoever defines the benchmark: each harness returns one from
    ``run_instance`` (see ``harnesses/base.py``), which is why the dead
    ``to_prediction`` that used to sit here had no callers.
    """
    instance_id: str = ""
    messages: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    cost: CostTracker = field(default_factory=CostTracker)

    def add_message(self, role: str, content: str, **extra):
        msg = {"role": role, "content": content}
        if extra:
            msg.update(extra)
        self.messages.append(msg)

    def save(self, path: Path):
        data = {
            "trajectory_format": "ash-agent-2.0",
            "instance_id": self.instance_id,
            "messages": self.messages,
            "info": {
                "model_stats": self.cost.to_dict(),
                "exit_status": self.info.get("exit_status", ""),
                "submission": self.info.get("submission", ""),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
