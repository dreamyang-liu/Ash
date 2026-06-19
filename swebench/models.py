"""Core types for SWE-bench benchmark."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolResult:
    """Result from calling an ash tool via CLI."""
    success: bool
    output: str
    error: Optional[str] = None


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
    thinking_budget: Optional[int] = None  # Extended thinking budget tokens (e.g. 10000)
    workdir: str = "/testbed"


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
    """Agent trajectory for saving and evaluation."""
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

    def to_prediction(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "model_patch": self.info.get("submission", ""),
            "model_name_or_path": self.info.get("model", "ash-agent"),
        }
