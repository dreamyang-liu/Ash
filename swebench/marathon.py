"""SWE-Marathon tasks: loading them, and grading what an agent did to one.

SWE-Marathon (abundant-ai/swe-marathon) is a benchmark of ultra-long-horizon
tasks -- build a zstd decoder from the RFC, port a codebase between languages
-- where logged attempts average 27M tokens and expert estimates run 12-400
hours. It is the workload the per-step checkpoint machinery exists for: a
rollout that dies at hour three should resume, not restart.

A task is a directory rather than a dataset row, which is the whole reason
this module exists alongside ``dataset.py``:

    tasks/<name>/
      task.toml        name, limits, resources, grader restore paths
      instruction.md   the prompt, verbatim
      environment/     Dockerfile -- built locally, no public image exists
      tests/           test.sh plus its fixtures, mounted at /tests to grade

Grading is the task's own ``tests/test.sh``, and it is deliberately hostile:
it sanitizes PATH and the loader environment, decrypts hidden expected
outputs that the agent never sees, and fingerprints libzstd to catch a
smuggled copy. That is why grading runs the script as-is instead of
reimplementing its checks -- the anti-cheat *is* the specification. It reports
a binary reward in ``/logs/verifier/reward.txt`` and a partial score in
``/logs/verifier/metrics.json``; both are collected, because a binary 0.0
hides the difference between 37 of 43 tests and none of them.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

#: Where the task's own files live inside the sandbox while grading.
TESTS_MOUNT = "/tests"
LOGS_MOUNT = "/logs"

#: Task fields worth carrying; the file has many more (author, review notes).
_METADATA_KEYS = ("difficulty", "category", "tags", "expert_time_estimate_hours")


class MarathonError(RuntimeError):
    """A task could not be loaded, built, or graded."""


@dataclass
class MarathonTask:
    """One task directory, parsed."""

    name: str
    directory: Path
    instruction: str
    #: Wall-clock the task itself allots the agent, in seconds.
    agent_timeout_sec: float = 18000.0
    verifier_timeout_sec: float = 1200.0
    cpus: int = 4
    memory_mb: int = 16384
    metadata: dict = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        """Identifier used for outputs; the directory name is the stable one."""
        return self.directory.name

    @property
    def dockerfile_dir(self) -> Path:
        return self.directory / "environment"

    @property
    def tests_dir(self) -> Path:
        return self.directory / "tests"


def _parse_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_task(directory: Path | str) -> MarathonTask:
    """Parse a task directory.

    Missing pieces fail here rather than mid-run: a task with no instruction
    or no environment cannot be attempted, and finding that out after building
    an image wastes minutes.
    """
    directory = Path(directory).resolve()
    config_path = directory / "task.toml"
    instruction_path = directory / "instruction.md"
    for required in (config_path, instruction_path, directory / "environment"):
        if not required.exists():
            raise MarathonError(f"not a SWE-Marathon task directory: "
                                f"{required} is missing")

    config = _parse_toml(config_path)
    task_section = config.get("task") or {}
    environment = config.get("environment") or {}
    metadata = {key: (config.get("metadata") or {}).get(key)
                for key in _METADATA_KEYS
                if (config.get("metadata") or {}).get(key) is not None}

    return MarathonTask(
        name=str(task_section.get("name") or directory.name),
        directory=directory,
        instruction=instruction_path.read_text(),
        agent_timeout_sec=float((config.get("agent") or {}).get(
            "timeout_sec", 18000.0)),
        verifier_timeout_sec=float((config.get("verifier") or {}).get(
            "timeout_sec", 1200.0)),
        cpus=int(environment.get("cpus", 4)),
        memory_mb=int(environment.get("memory_mb", 16384)),
        metadata=metadata,
    )


def discover_tasks(root: Path | str) -> list[MarathonTask]:
    """Every task under a checkout's ``tasks/`` directory, name-sorted."""
    root = Path(root)
    tasks_root = root / "tasks" if (root / "tasks").is_dir() else root
    tasks = []
    for candidate in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        if (candidate / "task.toml").exists():
            tasks.append(load_task(candidate))
    return tasks


def image_tag(task: MarathonTask, registry: str = "localhost:5000") -> str:
    """Local image reference for a task's environment."""
    return f"{registry}/swe-marathon-{task.instance_id}:latest"


def build_image(task: MarathonTask, registry: str = "localhost:5000",
                timeout: float = 2400.0, force: bool = False) -> str:
    """Build (and push) the task's environment image, returning its reference.

    Marathon environments are Dockerfiles, not published images -- several
    bake in encrypted verification assets, so there is nothing to pull. The
    push target is a registry the sandbox backend can reach; an already-pushed
    image is reused unless ``force``, because these builds run minutes.
    """
    reference = image_tag(task, registry)
    if not force and _image_in_registry(reference):
        return reference

    for command in (
        ["docker", "build", "-t", reference, str(task.dockerfile_dir)],
        ["docker", "push", reference],
    ):
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout)
        if result.returncode != 0:
            raise MarathonError(
                f"{command[1]} failed for {task.instance_id}: "
                f"{(result.stderr or result.stdout)[-400:]}")
    return reference


def _image_in_registry(reference: str) -> bool:
    result = subprocess.run(["docker", "manifest", "inspect", reference],
                            capture_output=True, text=True, timeout=120)
    return result.returncode == 0


@dataclass
class MarathonResult:
    """What grading found."""

    reward: float
    partial_score: float
    metrics: dict = field(default_factory=dict)
    #: Present when the verifier could not be run at all, as opposed to
    #: running and failing the agent.
    error: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.reward >= 1.0


def grade(session: Any, task: MarathonTask) -> MarathonResult:
    """Run the task's own verifier in the sandbox and read its verdict.

    The script expects its fixtures at ``/tests`` and a writable ``/logs``, so
    both are staged first: fixtures are uploaded through the session (the
    sandbox has no view of the host filesystem), and the script's own
    environment sanitization is left alone.

    A verifier that cannot run yields ``error`` with reward 0.0 rather than an
    exception -- an ungradeable attempt is a zero, and raising here would lose
    the trajectory that produced it.
    """
    try:
        _stage_tests(session, task)
    except Exception as exc:  # noqa: BLE001 - report, never lose the run
        return MarathonResult(0.0, 0.0, error=f"staging failed: {exc}")

    result = session.execute("shell", {
        "command": f"bash {TESTS_MOUNT}/test.sh 2>&1 | tail -40",
        "timeout": int(task.verifier_timeout_sec),
    })
    verifier_output = result.output or ""

    reward = _read_reward(session)
    metrics = _read_metrics(session)
    partial = float(metrics.get("partial_score") or 0.0)
    if reward is None:
        # No reward file: the script died before scoring (a build failure, a
        # timeout). Fall back to its printed line, then to zero.
        match = re.search(r"Reward:\s*([0-9.]+)", verifier_output)
        if match:
            reward = float(match.group(1))
        else:
            return MarathonResult(
                0.0, partial, metrics,
                error=f"verifier produced no reward: {verifier_output[-300:]}")
    return MarathonResult(reward, partial, metrics)


def _stage_tests(session: Any, task: MarathonTask) -> None:
    """Put the task's test fixtures where its verifier expects them."""
    session.execute("shell", {
        "command": f"mkdir -p {TESTS_MOUNT} {LOGS_MOUNT}/verifier"})
    for path in sorted(task.tests_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(task.tests_dir)
        destination = f"{TESTS_MOUNT}/{relative}"
        session.execute("shell", {
            "command": f"mkdir -p {shlex.quote(str(Path(destination).parent))}"})
        if not session.upload_file(path, destination):
            raise MarathonError(
                f"could not stage {relative} into the sandbox; grading needs "
                f"the task's fixtures at {TESTS_MOUNT} (binary and "
                f"multi-megabyte, so the tool surface cannot carry them)")


def _read_reward(session: Any) -> Optional[float]:
    result = session.execute("shell", {
        "command": f"cat {LOGS_MOUNT}/verifier/reward.txt 2>/dev/null"})
    text = (result.output or "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _read_metrics(session: Any) -> dict:
    result = session.execute("shell", {
        "command": f"cat {LOGS_MOUNT}/verifier/metrics.json 2>/dev/null"})
    try:
        return json.loads((result.output or "").strip())
    except Exception:
        return {}
