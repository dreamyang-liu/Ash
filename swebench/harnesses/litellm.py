"""LiteLLM harness — custom agent loop supporting any model via litellm."""

import threading
from pathlib import Path

from .base import BaseHarness
from ..dataset import resolve_image, format_task_prompt, image_registry_for_subset
from ..sandbox import AshSession
from ..models import AgentConfig
from ..agent import AshAgent
from ..agent.tools import TOOLS_SCHEMA, BASH_ONLY_SCHEMA
from .. import style as S


class LiteLLMHarness(BaseHarness):
    """Runs the custom agent loop (agent.py) with any litellm-compatible model."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._dashboard = None

    def set_dashboard(self, dashboard):
        """Set the live dashboard for step updates (parallel mode)."""
        self._dashboard = dashboard

    def run_instance(self, instance: dict, output_dir: Path) -> dict:
        c = self.config
        instance_id = instance["instance_id"]
        subset = c.get("subset", "verified")
        registry = image_registry_for_subset(subset)
        image = resolve_image(instance, template=c.get("image_template", ""), registry=registry)
        workers = int(c.get("workers", 1))
        quiet = workers > 1

        if not quiet:
            print(S.header(instance_id))
            print(S.kv("image   ", S.dim(image)))

        # Notify dashboard of spawn
        if self._dashboard:
            self._dashboard.update(instance_id, status="spawning")

        session = AshSession(runtime_bin=c.get("runtime_bin"), quiet=quiet)

        try:
            if not session.create(image):
                if self._dashboard:
                    self._dashboard.update(instance_id, status="failed", detail="session_failed")
                return self._fail(instance_id, "session_failed")

            agent_config = AgentConfig(
                model=c.get("model", "openai/Qwen/Qwen3-Coder-30B-A3B-Instruct"),
                api_base=c.get("api_base"),
                api_key=c.get("api_key"),
                max_tokens=c.get("max_tokens", 8192),
                step_limit=c.get("step_limit", 250),
                cost_limit=c.get("cost_limit", 3.0),
                temperature=c.get("temperature"),
                reasoning_effort=c.get("reasoning_effort"),
                prompt_cache=c.get("prompt_cache", True),
                tools=c.get("tools", "default"),
                system_template=c.get("system_template"),
                instance_template=c.get("instance_template"),
            )

            if not quiet:
                print(S.kv("model   ", S.dim(agent_config.model)))

            def _on_step(n: int, kind: str, text: str):
                if self._dashboard:
                    cost = agent.cost.total_cost if agent else 0
                    detail = f"{kind} {text[:36]}"
                    self._dashboard.update(instance_id, status="running", step=n, detail=detail, cost=cost)
                elif not quiet:
                    line = S.step(n, kind, text)
                    print(line, flush=True)

            agent = AshAgent(
                agent_config,
                executor=session.execute,
                on_step=_on_step,
                trace_dir=output_dir / "traces",
            )
            if quiet:
                agent.stream = False
            tools_mode = c.get("tools", "default")
            agent.set_tools_schema(BASH_ONLY_SCHEMA if tools_mode == "bash_only" else TOOLS_SCHEMA)

            if agent_config.instance_template:
                task = instance.get("problem_statement", "")
            else:
                task = format_task_prompt(instance)
            exit_status = agent.run(task, instance_id=instance_id)
            patch = session.get_patch()

            # Save trajectory
            agent.trajectory.info = {
                "exit_status": exit_status,
                "submission": patch,
                "model": agent_config.model,
            }
            agent.trajectory.cost = agent.cost
            traj_path = output_dir / "trajectories" / f"{instance_id}.json"
            agent.trajectory.save(traj_path)

            if not quiet and not self._dashboard:
                print(S.kv("exit    ", S.green(exit_status) if exit_status == "completed" else S.yellow(exit_status)))
                print(S.kv("cost    ", S.cost(agent.cost.total_cost, agent.cost.api_calls)))
                print(S.kv("patch   ", S.patch_info(patch)))

            return {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": agent_config.model,
                "exit_status": exit_status,
            }

        except Exception as e:
            if not quiet and not self._dashboard:
                print(S.kv("error   ", S.bright_red(str(e))))
            return self._fail(instance_id, f"error: {e}")

        finally:
            session.destroy()

    def _fail(self, instance_id: str, status: str) -> dict:
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": self.config.get("model", "unknown"),
            "exit_status": status,
        }
