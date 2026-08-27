"""Core types for the SWE-bench harness.

``ToolResult`` and ``CommandOutcome`` moved to :mod:`harness.core.result` — they
are the repo-wide executor seam, not benchmark types. Re-exported here (the same
class objects, so isinstance is unaffected) because most of this package and
several external callers import them from this module.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from harness.core.result import CommandOutcome, ToolResult

__all__ = ["CommandOutcome", "ToolResult", "AgentConfig", "CostTracker", "Trajectory"]

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
    #: Input tokens of the MOST RECENT call, as the provider counted them --
    #: what that one call actually put in the context window. The totals above
    #: accumulate the transcript once per call and so say nothing about how
    #: full the window is, which is what makes this worth recording
    #: separately (window management itself counts with the model's tokenizer;
    #: see agent/context_window.py).
    last_input_tokens: int = 0

    def update(self, response: Any):
        self.api_calls += 1
        if hasattr(response, "usage") and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0
            # Track cache usage from Anthropic/Bedrock responses
            usage = response.usage
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            # Cached input still occupies the window. Providers differ on
            # whether prompt_tokens already includes it, so this sum can
            # over-count by the cached fraction -- deliberately the safe
            # direction for a guard: eliding slightly early costs a little
            # context, overflowing the window kills the run.
            self.last_input_tokens = ((usage.prompt_tokens or 0)
                                      + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                                      + (getattr(usage, "cache_creation_input_tokens", 0) or 0))
        try:
            from litellm import completion_cost
            self.total_cost += completion_cost(completion_response=response)
        except Exception:
            pass

    def steps_left(self, step_limit: int, cost_limit: float) -> int:
        """How many more model calls the budget allows, by whichever runs out first.

        Steps and money are separate ceilings and the tighter one decides. Worth
        being explicit, because they are easy to conflate: one real run stopped at
        59 of 100 steps having spent $3.07 of $3.00, so counting steps alone would
        have called it comfortable with 41 to go.
        """
        remaining_steps = step_limit - self.api_calls
        if self.api_calls < 1 or self.total_cost <= 0:
            return max(remaining_steps, 0)
        avg = self.total_cost / self.api_calls
        affordable = int((cost_limit - self.total_cost) / avg) if avg > 0 else remaining_steps
        return max(min(remaining_steps, affordable), 0)

    def budget_warning(self, step_limit: int, cost_limit: float) -> Optional[str]:
        """Return a one-time warning when the budget is nearly gone, else None.

        Says how much is left and nothing about what to do with it -- what counts
        as finishing is the caller's business, and a cost tracker that told an
        agent to "submit its best attempt" was answering a question only the eval
        layer can ask.
        """
        if self.api_calls < 3:
            return None
        est = self.steps_left(step_limit, cost_limit)
        if est > 4 and (step_limit - self.api_calls) > 4:
            return None
        return (
            f"\n\n[Budget] ~{est} step(s) left "
            f"({self.api_calls}/{step_limit} steps, "
            f"${self.total_cost:.2f}/${cost_limit:.2f})."
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
        # Whatever the harness attached to `info` is written out, with the
        # three derived fields overlaid on top. It used to be only those
        # three, which silently dropped anything else a harness recorded --
        # the per-step checkpoint map among them, so a saved trajectory could
        # not be replayed even though the run had captured every step.
        info = dict(self.info)
        info.update({
            "model_stats": self.cost.to_dict(),
            "exit_status": self.info.get("exit_status", ""),
            "submission": self.info.get("submission", ""),
        })
        data = {
            "trajectory_format": "ash-agent-2.0",
            "instance_id": self.instance_id,
            "messages": self.messages,
            "info": info,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        # `default=str` because messages now carry provider-shaped tool calls:
        # a trajectory is a record, and losing the whole run to one
        # unserializable field would be the worst possible trade.
        path.write_text(json.dumps(data, indent=2, default=str))
