"""LiteLLM harness — custom agent loop supporting any model via litellm."""

import threading
from pathlib import Path

from .base import BaseHarness
from ..dataset import resolve_image, format_task_prompt, image_registry_for_subset
from ..backends import backend_config
from ..prediction import failure, prediction
from ..submission import (DEFAULT_RESERVE_STEPS, SUBMISSION_KEY,
                          extract_submission, reserve_submission)
from ..sandbox import AshSession
from ..models import AgentConfig
from ..agent import AshAgent
from ..agent.checkpoints import install as install_checkpoints
from ..agent.context_window import make_context_window_guard
from ..agent.trace import new_run_id
from ..agent.tools import DEFAULT_PANEL, build_panel
from ..agent.interceptors import default_pipeline
from ..agent.pipeline import load_pipeline
from .. import style as S


def _checkpoint_tracer(agent):
    """Report checkpoints on the agent's event stream, when one is open.

    The writer is created by ``run()``, so it is resolved per call rather than
    captured at wiring time. Unknown event types are additive in the trace
    schema (docs/TRACE_SCHEMA.md), so a consumer that does not know
    ``checkpoint.recorded`` simply skips it.
    """
    def report(record):
        writer = getattr(agent, "_event_trace", None)
        if writer is None:
            return
        writer.emit(
            "checkpoint.recorded",
            turn_id=f"turn-{record.turn}",
            snapshot_id=record.snapshot_id,
            captured=record.captured,
            disk_only=record.disk_only,
            rootfs_layers=record.rootfs_layers,
            chain_size_mb=record.chain_size_mb,
            reboarded=record.reboarded,
        )
    return report


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

        session = AshSession(runtime_bin=c.get("runtime_bin"), quiet=quiet,
                             backend=backend_config(c))

        try:
            if not session.create(image):
                if self._dashboard:
                    self._dashboard.update(instance_id, status="failed", detail="session_failed")
                return self._fail(instance_id, "session_failed")

            agent_config = AgentConfig(
                custom_tools_dir=c.get("custom_tools_dir"),
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

            agent_id = "agent"
            agent = AshAgent(
                agent_config,
                executor=session.executor_for(agent_id),
                on_step=_on_step,
                trace_dir=output_dir / "traces",
                run_id=new_run_id(),
                agent_id=agent_id,
                sandbox_id=session.sandbox_id,
            )
            if quiet:
                agent.stream = False
            # Your own L2 interceptors, mounted outside the defaults. `interceptors` is
            # a path to a Python file holding `PIPELINE = [MyInterceptor()]` --
            # a file rather than a config schema because policy is code (ADR-3):
            # a YAML dialect for "reject writes under src/ unless ..." becomes a
            # crippled programming language. Absent, the agent mounts the
            # defaults exactly as before.
            # The context-window guard is a safety net rather than a policy
            # choice: it no-ops while the transcript is under the model's own
            # budget, and without it a long run dies of an API error with its
            # budget unspent. Wired unconditionally; `context_budget_fraction:
            # 0` opts out for anyone who wants the raw failure.
            budget_fraction = float(c.get("context_budget_fraction", 0.70))
            if budget_fraction > 0:
                agent.before_query_hooks.append(make_context_window_guard(
                    strategy=c.get("context_strategy", "elide"),
                    budget_fraction=budget_fraction,
                    target_fraction=float(
                        c.get("context_target_fraction", 0.45))))

            plugins = c.get("interceptors")
            if plugins:
                agent.pipeline = default_pipeline(
                    extra=load_pipeline(plugins).interceptors)
            # Per-step environment checkpoints. Mounted after the plugin
            # block so the mutation tracker composes with a configured
            # pipeline rather than replacing it. Off unless configured, and a
            # no-op on backends that cannot snapshot (Docker, k8s).
            checkpointer = None
            checkpoint_cfg = c.get("checkpoints") or {}
            if checkpoint_cfg.get("enabled") and session.supports_snapshot():
                checkpointer = install_checkpoints(
                    agent, session,
                    always=checkpoint_cfg.get("trigger", "mutation") == "every_step",
                    disk_only=checkpoint_cfg.get("mode", "disk_only") != "full",
                    reboard=checkpoint_cfg.get("reboard", True),
                    squash_lineage_at=int(checkpoint_cfg.get(
                        "squash_lineage_at", 128)),
                    name_prefix=checkpoint_cfg.get("name_prefix", ""),
                    # Checkpoints persist the run themselves, so an
                    # interrupted benchmark run is resumable too.
                    trajectory_path=(output_dir / "trajectories" /
                                     f"{instance_id}.json"),
                    on_checkpoint=_checkpoint_tracer(agent),
                )

            # One call, because schema, routing and custom tools have to agree.
            # `tools:` names a shipped panel or points at a manifest of your own.
            agent.use_panel(build_panel(c.get("tools", DEFAULT_PANEL),
                                        agent_config.custom_tools_dir,
                                        registry=session.tools))

            # The agent hands in its own diff: it knows which files it fixed,
            # which is the one thing a harness reading git state cannot know.
            # The reserve buys the turns to do it before the budget is gone.
            reserve = int(c.get("submission_reserve_steps",
                                DEFAULT_RESERVE_STEPS))
            if reserve > 0:
                before_query, before_finish = reserve_submission(
                    reserve, agent_config.workdir)
                agent.before_query_hooks.append(before_query)
                agent.before_finish_hooks.append(before_finish)

            if agent_config.instance_template:
                task = instance.get("problem_statement", "")
            else:
                task = format_task_prompt(instance)
            exit_status = agent.run(task, instance_id=instance_id)
            # No fallback to git: if the agent did not hand anything in, the
            # prediction is empty and exit_status says why. Extracting a patch
            # anyway would substitute our guess for the agent's judgement.
            patch = extract_submission(agent.trajectory) if reserve > 0 \
                else session.get_patch()

            # Save trajectory
            agent.trajectory.info = {
                "exit_status": exit_status,
                SUBMISSION_KEY: patch,
                "model": agent_config.model,
            }
            # Which environment this trajectory belongs to. Recorded whether
            # or not checkpoints are on: a replay needs it, and so does anyone
            # asking later what a saved run was actually run against.
            describe_environment = getattr(session, "environment", None)
            if callable(describe_environment):
                agent.trajectory.info["environment"] = describe_environment()
            if checkpointer is not None:
                # What a replay needs: for each step, the snapshot holding the
                # environment as it stood after that step.
                agent.trajectory.info["checkpoints"] = checkpointer.as_info()
            agent.trajectory.cost = agent.cost
            traj_path = output_dir / "trajectories" / f"{instance_id}.json"
            agent.trajectory.save(traj_path)

            if not quiet and not self._dashboard:
                print(S.kv("exit    ", S.green(exit_status) if exit_status == "completed" else S.yellow(exit_status)))
                print(S.kv("cost    ", S.cost(agent.cost.total_cost, agent.cost.api_calls)))
                print(S.kv("patch   ", S.patch_info(patch)))

            return prediction(instance_id, agent_config.model, patch, exit_status)

        except Exception as e:
            if not quiet and not self._dashboard:
                print(S.kv("error   ", S.bright_red(str(e))))
            return self._fail(instance_id, f"error: {e}")

        finally:
            session.destroy()

    def _fail(self, instance_id: str, status: str) -> dict:
        return failure(instance_id, self.config.get("model"), status)
