"""Pinned Terminal-Bench jobs and explicit evaluation coverage accounting."""

from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
from typing import Any


DATASET = "terminal-bench/terminal-bench"
DATASET_VERSION = "4.0.0"
HARBOR_VERSION = "0.22.0"


def build_config(args: argparse.Namespace) -> Any:
    from harbor.models.job.config import JobConfig

    dataset = ({"path": str(args.tasks_dir.resolve())} if args.tasks_dir else
               {"name": DATASET, "version": DATASET_VERSION})
    if args.task:
        dataset["task_names"] = args.task
    agent = ({"import_path": "terminalbench.agent:AshClaudeCode", "model_name": args.model}
             if args.agent == "claude-code" else {"name": args.agent})
    environment = {"type": args.env, "delete": True}
    if args.env == "agentenv":
        environment = {"import_path": "terminalbench.agentenv:AgentENVEnvironment", "delete": True,
                       "kwargs": {"runtime_bin": str(args.runtime_bin.resolve()),
                                  "checkpoint_mode": args.checkpoint_mode,
                                  "sandbox_ttl": args.sandbox_ttl,
                                  "server_url": args.server_url,
                                  "api_key_file": str(args.api_key_file.resolve()) if args.api_key_file else None,
                                  "image_registry": args.image_registry}}
        if args.agent == "claude-code":
            agent["import_path"] = "terminalbench.agentenv_agent:AgentENVClaudeCode"
    return JobConfig.model_validate({
        "job_name": args.output.resolve().name,
        "jobs_dir": str(args.output.resolve().parent),
        "n_attempts": args.attempts,
        "n_concurrent_trials": args.workers,
        "datasets": [dataset], "agents": [agent],
        "environment": environment,
        "retry": {"max_retries": 0},
    })


def summarize(output: Path) -> dict:
    manifest = json.loads((output / "ash-eval.json").read_text())
    expected = manifest["expected_trials"]
    rows = []
    for path in sorted(output.glob("*/result.json")):
        body = json.loads(path.read_text())
        reward = ((body.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        error = body.get("exception_info")
        if not body.get("finished_at"):
            status = "incomplete"
        elif error:
            status = "error"
        elif type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
            status = "invalid_reward"
        else:
            status = "completed"
        rows.append({"task": body["task_name"], "trial": body["trial_name"],
                     "status": status, "reward": reward, "error": error, "report": str(path)})
    if len(rows) > expected or len({row["trial"] for row in rows}) != len(rows):
        raise ValueError("Trial reports do not match the expected cohort")
    completed = [row for row in rows if row["status"] == "completed"]
    total_reward = sum(row["reward"] for row in completed)
    complete = len(completed) == expected and expected > 0
    summary = {
        "dataset": manifest["dataset"], "expected_trials": expected,
        "completed_trials": len(completed),
        "resolved_trials": sum(row["reward"] == 1 for row in completed),
        "error_trials": sum(row["status"] in {"error", "invalid_reward"} for row in rows),
        "missing_or_incomplete_trials": expected - len(rows) + sum(row["status"] == "incomplete" for row in rows),
        "complete": complete,
        "mean_reward_completed": total_reward / len(completed) if completed else None,
        "mean_reward_lower_bound": total_reward / expected if expected else None,
        "final_mean_reward": total_reward / expected if complete else None,
        "task_filter": manifest["task_filter"], "attempts_per_task": manifest["attempts_per_task"],
        "trials": rows,
    }
    temporary = output / "ash-summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(output / "ash-summary.json")
    return summary


async def run_job(args: argparse.Namespace, config: Any) -> dict:
    from harbor.job import Job

    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume:
            raise FileExistsError("Output exists; choose a new --output or explicitly --resume")
        manifest = json.loads((args.output / "ash-eval.json").read_text())
        if type(config).model_validate(manifest["config"]) != config:
            raise ValueError("Resume config differs from the recorded job")
    job = await Job.create(config)
    manifest = {
        "dataset": f"{DATASET}@{DATASET_VERSION}" if not args.tasks_dir else str(args.tasks_dir.resolve()),
        "harbor_version": HARBOR_VERSION, "expected_trials": len(job),
        "task_filter": args.task or [], "attempts_per_task": args.attempts,
        "config": config.model_dump(mode="json"),
    }
    (args.output / "ash-eval.json").write_text(json.dumps(manifest, indent=2))
    try:
        if args.env == "agentenv":
            audit = audit_agentenv(job, config)
            (args.output / "agentenv-admission.json").write_text(json.dumps(audit, indent=2))
            if any(not task["supported"] for task in audit):
                raise ValueError("AgentENV admission failed; see agentenv-admission.json. No trials started.")
        if args.phase == "audit":
            return {"complete": True, "audit_only": True, "expected_trials": len(job)}
        await job.run()
    finally:
        summary = summarize(args.output)
    return summary


def audit_agentenv(job: Any, config: Any) -> list[dict]:
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config, resolve_task_verifier_mode
    from harbor.models.trial.paths import TrialPaths
    from harbor.trial.network_policy import resolve_trial_network_plan
    from terminalbench.agentenv import AgentENVEnvironment

    records = []
    for downloaded in job._task_download_results.values():
        errors = []
        task = Task(downloaded.path)
        if task.config.steps:
            errors.append("Multi-step tasks are not supported by TB4 summary accounting")
        mode = resolve_task_verifier_mode(task.config)
        plan = resolve_trial_network_plan(task.config, config.agents[0], config.environment, None, verifier_mode=mode)
        phases = [plan.agent_phase]
        if plan.verifier_env_baseline is None:
            phases.append(plan.verifier_phase)
        environments = [("agent", task.paths.environment_dir, task.config.environment,
                         plan.agent_env_baseline, phases)]
        verifier_config = resolve_effective_verifier_env_config(task.config, None)
        if verifier_config is not None:
            environments.append(("verifier", task.paths.tests_dir, verifier_config,
                                 plan.verifier_env_baseline, [plan.verifier_phase]))
        for role, directory, environment_config, baseline, phase_policies in environments:
            try:
                AgentENVEnvironment(environment_dir=directory, environment_name=task.name,
                                    session_id=f"audit-{role}", trial_paths=TrialPaths(trial_dir=job.job_dir),
                                    task_env_config=environment_config.model_copy(deep=True),
                                    network_policy=baseline, phase_network_policies=phase_policies,
                                    **config.environment.kwargs)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                errors.append(f"{role}: {exc}")
        records.append({"task": task.name, "supported": not errors, "errors": errors,
                        "path": str(downloaded.path)})
    return records


def preflight(args: argparse.Namespace) -> None:
    if args.env == "agentenv":
        if not args.runtime_bin.is_file():
            raise RuntimeError(f"Missing ash-runtime: {args.runtime_bin}")
    if args.env == "docker":
        try:
            subprocess.run(["docker", "compose", "version"], check=True,
                           capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Docker Compose v2 is required for --env docker") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["plan", "audit", "run", "summarize"], default="plan")
    parser.add_argument("--output", type=Path, default=Path("runs/terminalbench4"))
    parser.add_argument("--model")
    parser.add_argument("--agent", choices=["claude-code", "oracle", "nop"], default="claude-code")
    parser.add_argument("--env", default="agentenv")
    parser.add_argument("--runtime-bin", type=Path, default=Path(__file__).resolve().parents[1] / "runtime/ash-runtime")
    parser.add_argument("--server-url")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--image-registry", help="Registry reachable by both Docker and AgentENV for Dockerfile builds")
    parser.add_argument("--checkpoint-mode", choices=["full", "disk_only"], default="full")
    parser.add_argument("--sandbox-ttl", type=int, default=36000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--task", action="append", help="Task name or glob; repeat for a subset")
    parser.add_argument("--tasks-dir", type=Path, help="Explicit local Harbor tasks for development/gates")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.attempts < 1:
        parser.error("workers and attempts must be positive")
    if args.phase == "audit" and args.env != "agentenv":
        parser.error("--phase audit currently checks the AgentENV adapter only")
    if args.phase != "summarize" and args.agent == "claude-code" and not args.model:
        parser.error("--model is required for claude-code")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "summarize":
        summary = summarize(args.output)
    else:
        if version("harbor") != HARBOR_VERSION:
            raise RuntimeError(f"Install harbor=={HARBOR_VERSION}; this adapter is validated against that version")
        config = build_config(args)
        if args.phase == "plan":
            print(config.model_dump_json(indent=2))
            return 0
        preflight(args)
        summary = asyncio.run(run_job(args, config))
    print(json.dumps({key: value for key, value in summary.items() if key != "trials"}, indent=2))
    return 0 if summary["complete"] else 2
