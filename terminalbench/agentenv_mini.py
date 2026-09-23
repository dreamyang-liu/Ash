"""Host mini-swe-agent over Harbor's AgentENV task sandbox."""

from __future__ import annotations

import math
import os
import time

from harness.orchestrator.run import RunSpec
from harness.rollout import endpoint
from harness.slots.mini_swe import MiniSweSlot
from terminalbench.agentenv_agent import AgentENVClaudeCode


class AgentENVMini(AgentENVClaudeCode):
    def __init__(self, *args, inference_endpoint: str, api_key_env: str,
                 actor_timeout_s: float = 36000, max_output_tokens: int = 64000,
                 max_turns: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_endpoint = endpoint(inference_endpoint)
        if not api_key_env or not api_key_env.isidentifier():
            raise ValueError("mini requires an API key environment variable name")
        if not math.isfinite(actor_timeout_s) or actor_timeout_s <= 0:
            raise ValueError("mini actor timeout must be finite and positive")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("mini max_output_tokens must be positive")
        if type(max_turns) is not int or max_turns <= 0:
            raise ValueError("mini max_turns must be positive")
        self.api_key_env = api_key_env
        self.actor_timeout_s = float(actor_timeout_s)
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns

    @staticmethod
    def name() -> str:
        return "ash-agentenv-mini-swe"

    def version(self) -> str | None:
        return MiniSweSlot().version()

    async def setup(self, environment) -> None:
        await super().setup(environment)
        if not self.model_name:
            raise ValueError("mini requires an explicit model")
        if not os.environ.get(self.api_key_env):
            raise ValueError(f"mini model key environment variable {self.api_key_env} is unset")

    def _make_spec(self, prompt, workspace, journal, environment) -> RunSpec:
        # Harbor owns the task timeout. This finite Ash deadline also bounds
        # model traffic if Harbor loses its cancellation path.
        return RunSpec(
            prompt=prompt, slot="mini-swe-agent", model=self.model_name,
            cwd=str(workspace.resolve()), run_id=self.logs_dir.parent.name,
            journal_path=journal, timeout_s=self.actor_timeout_s,
            session=environment.session, keep_sandbox=True,
            transport="http", tools="shell_only", backend=environment.backend,
            runtime_bin=environment.runtime_bin, sandbox_image=environment.image,
            extra={
                "mini": {
                    "model": {"model_kwargs": {"max_tokens": self.max_output_tokens}},
                    "environment": {"timeout": 60},
                },
                "rollout_contract": {
                    "message_export": True,
                    "model_endpoint": self.inference_endpoint,
                    "api_key_env": self.api_key_env,
                    "model": self.model_name,
                    "deadline_at": time.time() + self.actor_timeout_s,
                    "max_turns": self.max_turns,
                    "sampling_params": {"max_new_tokens": self.max_output_tokens},
                },
            },
        )
