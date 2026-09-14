"""Pro task adapter for the shared parent/branch rollout loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.core.guidance import render_branch_note
from swebench.fork_eval import Grade, tool_primer
from swebench_pro import grade as grading
from swebench_pro.tasks import DATASET_REVISION, Task, load_tasks


class SWEbenchPro:
    name = "swebench-pro"
    image_env = True
    stop_on_grading_error = True

    def __init__(self, args: Any):
        if not getattr(args, "pro_repo", None):
            raise SystemExit("--benchmark swebench-pro requires --pro-repo (official checkout)")
        self.repo = Path(args.pro_repo).expanduser().resolve()
        self.data = Path(args.pro_data) if getattr(args, "pro_data", None) else None
        self.revision = getattr(args, "pro_dataset_revision", None) or DATASET_REVISION
        self.no_network = bool(getattr(args, "pro_block_network", False))
        self.shape = {"cpu": getattr(args, "pro_cpus", 4), "memory_mb": getattr(args, "pro_memory_mb", 16384)}
        self.verifier_timeout = getattr(args, "pro_verifier_timeout", 3600)
        self.runtime_port = getattr(args, "pro_runtime_port", None)
        self.collector_runtime_port = getattr(args, "pro_collector_runtime_port", None)
        if any(port is not None and not 1 <= port <= 65535
               for port in (self.runtime_port, self.collector_runtime_port)):
            raise ValueError("Pro runtime ports must be between 1 and 65535")
        if min(*self.shape.values(), self.verifier_timeout) < 1:
            raise ValueError("Pro resources and verifier timeout must be positive")

    def catalogue(self, args: Any) -> dict[str, Task]:
        return load_tasks(self.repo, self.data, self.revision)

    def instance(self, task: Task) -> dict:
        return {"instance_id": task.instance_id, "repo": task.repo, "image": task.image,
                "problem": task.problem, "f2p": list(task.f2p), "p2p": list(task.p2p), "task": task}

    def prompt(self, instance: dict) -> str:
        policy = instance.get("agent_network", "deny" if self.no_network else "backend-default")
        network = "The sandbox has no internet access.\n" if policy == "deny" else ""
        return (f"You are working in {instance['repo']} at /app inside your sandbox.\n"
                f"{network}\n## Task\n{instance['problem']}\n\n"
                "Your changes will be exported as a diff against the task's base commit and tested "
                "in a fresh environment. Both committed and uncommitted source changes count, "
                "including new files. Remove scratch artifacts before finishing.\n\n"
                + tool_primer(instance.get("slot", ""), "/app"))

    def branch_prompt(self, instance: dict, verdict: str, hint: str, **context: Any) -> str:
        return render_branch_note(hint, truncated=bool(context.get("truncated")))

    def resources(self, instance: dict) -> dict:
        return dict(self.shape)

    def prepare_image(self, instance: dict, backend: dict, directory: Path) -> str:
        return grading.prepare_image(instance["task"], backend, self.resources(instance), directory)

    def grade(self, snapshot_id: str, instance: dict, backend: dict) -> Grade:
        collector_backend = None
        if self.collector_runtime_port is not None:
            collector_backend = {**backend, "microvm": {
                **backend.get("microvm", {}), "runtime_port": self.collector_runtime_port}}
        return grading.grade_snapshot(snapshot_id, instance["task"], backend,
                                      resources=self.resources(instance), timeout=self.verifier_timeout,
                                      artifacts_dir=instance.get("verifier_artifacts_dir"),
                                      collector_backend=collector_backend)

    def summary(self, results: list[dict], expected_ids: list[str] | None = None) -> dict:
        expected = set(expected_ids) if expected_ids is not None else {row["instance"] for row in results}
        pending = sorted(expected - {row["instance"] for row in results})
        errors = [row["instance"] for row in results if not row["attempts"] or any(
            attempt.get("grading_error") or attempt.get("verifier_artifact_error")
            for attempt in row["attempts"])]
        resolved = sum(row["resolved"] for row in results)
        return {"expected_tasks": len(expected), "pending_task_ids": pending,
                "grading_complete": not errors and not pending, "grading_error_ids": errors,
                "resolved_lower_bound": resolved / len(expected) if expected else None,
                "final_resolved_rate": resolved / len(expected) if expected and not errors and not pending else None}
