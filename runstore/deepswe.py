"""Run the existing DeepSWE snapshot grader against pinned task assets."""

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import subprocess

from deepswe.grade import grade_snapshot as evaluate_snapshot
from deepswe.tasks import load_task
from runstore.specs import GradeSpec


def task_manifest(task_dir: Path) -> dict[str, str]:
    """Pin all inputs read by the task loader and verifier; omit oracle files."""
    paths = [task_dir / "task.toml", task_dir / "instruction.md"]
    paths.extend(path for path in (task_dir / "tests").rglob("*") if path.is_file())
    return {str(path.relative_to(task_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def grade_snapshot(spec: GradeSpec, row: dict, backend: dict, directory: Path, *, session_factory) -> dict:
    repo = Path(spec.harness_repo).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=10,
    ).strip()
    if revision != spec.grader_revision:
        raise ValueError("DeepSWE repository revision differs from the frozen grader")
    tasks_dir = repo / "tasks"
    task_dir = (tasks_dir / spec.instance_id).resolve()
    if task_dir.parent != tasks_dir or row.get("instance_id") != spec.instance_id:
        raise ValueError("DeepSWE task identity differs from the frozen dataset")
    if task_manifest(task_dir) != row.get("task_files_sha256"):
        raise ValueError("DeepSWE task assets differ from the frozen dataset")
    task = load_task(task_dir)
    if spec.verifier_network != "deny" or not task.no_network:
        raise ValueError("DeepSWE requires the task's offline execution and verifier")
    resources = {"cpu": task.cpus, "memory_mb": task.memory_mb}
    if spec.resources != resources:
        raise ValueError("DeepSWE resources differ from task.toml")
    microvm = backend.get("microvm", {})
    backend = {**backend, "microvm": {
        **microvm, "sandbox_ttl": max(microvm.get("sandbox_ttl", 0), math.ceil(spec.timeout_s) + 600),
    }}
    verdict = evaluate_snapshot(
        spec.snapshot_id, task, backend, artifacts_dir=directory / "grading",
        session_factory=session_factory,
    )
    return {
        "status": "error" if verdict.error else "completed",
        "grade": asdict(verdict),
        "resolved": bool(verdict.resolved) if not verdict.error else None,
        "failure_kind": "infrastructure" if verdict.error else None,
        "error": verdict.error,
        "grader_revision": revision,
    }
