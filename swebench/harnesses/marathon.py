"""Harness for SWE-Marathon tasks — the same agent loop, a different task source.

`litellm.py` gets its work from a dataset row; this gets it from a task
directory (`swebench/marathon.py`), builds that task's environment locally,
runs the shared agent loop against it, and grades with the task's own
verifier. Everything the benchmark path already provides — per-step
checkpoints, the context-window guard, tool panels, interceptors, traces —
comes along, which is the entire reason for making this a harness instead of
another driver script: the alternative was a hand-rolled script per
experiment, and the last one silently ran without the context guard because
mounting it was one line nobody had added.

What this path must do differently:

- **Budgets come from the task.** Marathon allots hours (5 for zstd-decoder)
  and expects 27M tokens; a 250-step default would stop a third of the way in
  with no signal. Steps and cost still bound the run, but the config supplies
  them deliberately rather than inheriting a SWE-bench-shaped default.
- **The environment is built, not pulled.** Several tasks bake encrypted
  verification assets into their image, so no published image exists.
- **Grading is the task's script, verbatim.** Its anti-cheat (PATH
  sanitization, encrypted expected outputs, library fingerprinting) is part of
  the specification; reimplementing the checks would be reimplementing the
  benchmark.
- **A partial score is recorded next to the binary reward.** Marathon's reward
  is all-or-nothing, and on tasks this long nearly every attempt scores zero;
  `partial_score` is what distinguishes 37 of 43 tests from none.
- **Context folding summarizes instead of eliding.** See `_attempt`: on this
  horizon the facts worth keeping exist only in tool output, so dropping them
  makes the agent rediscover what it knew.
"""

from pathlib import Path
from typing import Any, Optional

from .base import BaseHarness
from .. import style as S
from ..agent import AshAgent
from ..agent.checkpoints import install as install_checkpoints
from ..agent.context_window import make_context_window_guard
from ..agent.tools import DEFAULT_PANEL, build_panel
from ..agent.trace import new_run_id
from ..backends import backend_config
from ..marathon import MarathonError, build_image, grade, load_task
from ..models import AgentConfig
from ..prediction import failure, prediction
from ..sandbox import AshSession


def _default_max_tokens(model: str | None, fallback: int = 16384) -> int:
    """The model's own output ceiling, or a generous fallback."""
    if model:
        try:
            import litellm
            info = litellm.get_model_info(model) or {}
            allowed = info.get("max_output_tokens") or info.get("max_tokens")
            if allowed:
                return int(allowed)
        except Exception:
            pass
    return fallback


class MarathonHarness(BaseHarness):
    """One agent, one marathon task, graded by the task's own verifier."""

    def run_instance(self, instance: dict, output_dir: Path,
                     quiet: bool = False) -> dict:
        c = self.config
        task_dir = instance.get("task_dir") or c.get("task_dir")
        if not task_dir:
            return self._failure(instance, "no task_dir given")

        try:
            task = load_task(task_dir)
        except MarathonError as exc:
            return self._failure(instance, f"error: {exc}")

        if not quiet:
            print(S.kv("task    ", S.cyan(task.name)))
            print(S.kv("expert  ",
                       S.dim(f"{task.metadata.get('expert_time_estimate_hours', '?')}h estimate, "
                             f"difficulty {task.metadata.get('difficulty', '?')}")))

        # Resuming starts from a checkpoint instead of the task's image: the
        # environment already holds hours of work, and rebuilding would throw
        # it away. Hours are exactly what these tasks cost, so continuing is
        # the difference between finishing one and restarting it.
        resume_from = c.get("resume_from")
        if resume_from:
            image = resume_from
            if not quiet:
                print(S.kv("resume  ", S.cyan(str(resume_from)[:13] + "…")))
        else:
            try:
                image = build_image(
                    task, registry=c.get("registry", "localhost:5000"))
            except MarathonError as exc:
                return self._failure(instance, f"error: {exc}")

        session = AshSession(runtime_bin=c.get("runtime_bin"), quiet=quiet,
                             backend=backend_config(c))
        try:
            if not session.create(image):
                return self._failure(instance,
                                     f"error: sandbox creation failed for {image}")
            return self._attempt(task, session, output_dir, quiet)
        finally:
            session.destroy()

    # --- internals ---

    def _attempt(self, task, session: AshSession, output_dir: Path,
                 quiet: bool) -> dict:
        config = self.config
        agent_config = AgentConfig(
            model=config.get("model", "bedrock/us.anthropic.claude-sonnet-4-6"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            # The output limit, not a token budget: a marathon agent writes
            # whole files in single tool calls, and a call truncated mid-JSON
            # cost one real attempt its entire remaining budget. Default to
            # what the model actually allows rather than the benchmark-shaped
            # 8192 that produced that failure.
            max_tokens=config.get("max_tokens") or _default_max_tokens(
                config.get("model")),
            # Marathon tasks are hours long; a SWE-bench-shaped default would
            # stop the run a third of the way in and report it as finished.
            step_limit=config.get("step_limit", 1000),
            cost_limit=config.get("cost_limit", 50.0),
            temperature=config.get("temperature"),
            reasoning_effort=config.get("reasoning_effort"),
            prompt_cache=config.get("prompt_cache", True),
            tools=config.get("tools", "default"),
            # The task owns its working directory; its instructions and its
            # verifier both name absolute paths under it.
            workdir=config.get("workdir", "/app"),
        )

        agent_id = "agent"
        agent = AshAgent(agent_config,
                         executor=session.executor_for(agent_id),
                         trace_dir=output_dir / "traces",
                         run_id=new_run_id(),
                         agent_id=agent_id,
                         sandbox_id=session.sandbox_id)
        if quiet:
            agent.stream = False
        agent.use_panel(build_panel(config.get("tools", DEFAULT_PANEL),
                                    agent_config.custom_tools_dir,
                                    registry=session.tools))
        # Summarize rather than elide, by default and only here. On this
        # horizon the facts worth keeping live in tool output and nowhere
        # else: measured on a real 133-step attempt, 7 of 8 sampled facts --
        # the build flags, the expected hashes, that python3 and xxd are
        # absent from the image -- were gone after elision, tool calls
        # included. Rediscovering one of those costs more steps (~$0.23 each)
        # than the summary costs to write (~$0.09), and the same run's log
        # shows the agent re-litigating conclusions it had already reached.
        # Benchmark-length runs keep eliding: at 30 steps nothing folds at all.
        agent.before_query_hooks.append(make_context_window_guard(
            strategy=config.get("context_strategy", "summarize")))

        trajectory_path = (output_dir / "trajectories" /
                           f"{task.instance_id}.json")

        checkpointer = None
        checkpoint_cfg = config.get("checkpoints") or {}
        if checkpoint_cfg.get("enabled") and session.supports_snapshot():
            checkpointer = install_checkpoints(
                agent, session,
                always=checkpoint_cfg.get("trigger", "mutation") == "every_step",
                disk_only=checkpoint_cfg.get("mode", "disk_only") != "full",
                reboard=checkpoint_cfg.get("reboard", True),
                name_prefix=f"marathon-{task.instance_id}-",
                # Every checkpoint writes the trajectory beside it: snapshots
                # that outlive an interrupted run are only resumable if
                # something records which step each one is.
                trajectory_path=trajectory_path)

        prompt = task.instruction
        if config.get("resume_from"):
            # A resumed run has the artifacts but not the transcript, so the
            # environment has to be described rather than assumed. Stating it
            # plainly beats letting the agent rediscover it: the first thing it
            # would otherwise do is read files it has already written.
            prompt += (
                "\n\n---\n\nNOTE: this environment is resumed from an earlier "
                "session of this same task. Work already exists on disk -- "
                "inspect the current state first (the source files, whether it "
                "builds, what the visible tests say) and continue from there "
                "rather than starting over. You do not have the earlier "
                "conversation, only what is on disk.")

        exit_status = agent.run(prompt, instance_id=task.instance_id)

        result = grade(session, task)
        if not quiet:
            print(S.kv("reward  ", S.cost(result.reward, 0)
                       if result.reward else S.yellow("0.0")))
            print(S.kv("partial ", S.dim(f"{result.partial_score} "
                                         f"({result.metrics.get('total_passed', '?')}"
                                         f"/{result.metrics.get('total_tests', '?')} tests)")))

        agent.trajectory.info = {
            "exit_status": exit_status,
            "submission": "",
            "environment": session.environment(),
            "marathon": {
                "task": task.name,
                "reward": result.reward,
                "partial_score": result.partial_score,
                "metrics": result.metrics,
                "grading_error": result.error,
                "expert_time_estimate_hours":
                    task.metadata.get("expert_time_estimate_hours"),
            },
        }
        if checkpointer is not None:
            agent.trajectory.info["checkpoints"] = checkpointer.as_info()
        agent.trajectory.cost = agent.cost
        agent.trajectory.save(trajectory_path)

        # The prediction shape, so this harness composes with the batch
        # runner and the resume logic like any other. Marathon has no patch to
        # submit -- the graded artifact is the environment itself -- so the
        # report is a `failure` in the builder's vocabulary (nothing to
        # submit) carrying the grade alongside it. Pretending otherwise would
        # mean inventing a tenth hand-built dict, which is what
        # `prediction.py` exists to prevent.
        report = failure(task.instance_id, agent_config.model, exit_status)
        report.update({
            "reward": result.reward,
            "partial_score": result.partial_score,
            "cost": agent.cost.total_cost,
            "turns": agent.cost.api_calls,
            "metrics": result.metrics,
            "grading_error": result.error,
        })
        return report

    def _failure(self, instance: dict, status: str) -> dict:
        report = failure(instance.get("instance_id") or "unknown",
                         self.config.get("model"), status)
        report.update({"reward": 0.0, "partial_score": 0.0, "cost": 0.0,
                       "turns": 0, "metrics": {}, "grading_error": status})
        return report
