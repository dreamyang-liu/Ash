"""DeepSWE tasks on disk -> ``Task`` records.

A task directory (Harbor layout, as shipped in ``deep-swe/tasks/<id>/``):

    task.toml         metadata, image, cpu/memory, timeouts, network mode,
                      and the [[verifier.collect]] command that turns the
                      agent's sandbox into model.patch
    instruction.md    the agent's prompt, verbatim
    environment/      Dockerfile that produced the prebuilt image (not used:
                      task.toml names the image)
    tests/            the verifier: Dockerfile + the files it copies
    solution/         reference patch + solve.sh (the oracle for the grader gate)

Everything the grader needs is read from ``tests/Dockerfile`` rather than
assumed: which files land where, and which image they land on. If that
Dockerfile ever says something we cannot replay in a microVM (an ENV, a
WORKDIR, a second stage), loading fails loudly instead of grading against an
environment that differs from theirs.
"""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


class TaskError(ValueError):
    """A task directory is missing something, or says something we cannot honour."""


@dataclass(frozen=True)
class CollectStep:
    command: str
    timeout_s: float


@dataclass(frozen=True)
class VerifierFile:
    source: Path        # on the host, inside tests/
    destination: str    # inside the verifier VM


@dataclass(frozen=True)
class Task:
    task_id: str
    name: str
    repo_url: str
    language: str
    base_commit: str
    image: str
    instruction: str
    cpus: int
    memory_mb: int
    agent_timeout_s: float
    agent_network_mode: str
    verifier_timeout_s: float
    verifier_network_mode: str
    collect: Tuple[CollectStep, ...]
    verifier_files: Tuple[VerifierFile, ...]
    verifier_run_steps: Tuple[str, ...]
    f2p: Tuple[str, ...]
    p2p: Tuple[str, ...]
    task_dir: Path

    @property
    def repo(self) -> str:
        """``owner/name`` from the repository URL, for prompts and reports."""
        tail = re.sub(r"\.git$", "", self.repo_url.rstrip("/"))
        parts = tail.split("/")
        return "/".join(parts[-2:]) if len(parts) >= 2 else tail

    @property
    def no_network(self) -> bool:
        return self.agent_network_mode == "no-network"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"

    @property
    def solve_script(self) -> Optional[Path]:
        path = self.solution_dir / "solve.sh"
        return path if path.is_file() else None


_INSTRUCTION = re.compile(r"^\s*([A-Za-z]+)\s+(.*?)\s*$")


def parse_verifier_dockerfile(text: str) -> Tuple[str, List[Tuple[str, str]], List[str]]:
    """``(FROM image, [(COPY src, dst)], [RUN command])`` from a verifier Dockerfile.

    Only these three instructions are accepted. The verifier image is meant to
    be "the task image plus the hidden tests"; anything else would change the
    environment the tests run in, and we would not be replaying it.
    """
    from_image: Optional[str] = None
    copies: List[Tuple[str, str]] = []
    runs: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _INSTRUCTION.match(line)
        if not match:
            raise TaskError("unparseable Dockerfile line: %r" % raw)
        word, rest = match.group(1).upper(), match.group(2)
        if word == "FROM":
            if from_image is not None:
                raise TaskError("multi-stage verifier Dockerfile is not supported")
            from_image = rest.split()[0]
        elif word == "COPY":
            parts = shlex.split(rest)
            if len(parts) != 2 or any(p.startswith("--") for p in parts):
                raise TaskError("only `COPY <src> <dst>` is supported, got %r" % raw)
            copies.append((parts[0], parts[1]))
        elif word == "RUN":
            runs.append(rest)
        else:
            raise TaskError("verifier Dockerfile instruction %s is not replayed "
                            "in a microVM; refusing to grade on a different "
                            "environment (%r)" % (word, raw))
    if not from_image:
        raise TaskError("verifier Dockerfile has no FROM")
    return from_image, copies, runs


def _require(table: dict, *keys: str, where: str):
    node = table
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise TaskError("%s: missing %s" % (where, ".".join(keys)))
        node = node[key]
    return node


def load_task(task_dir: Path) -> Task:
    task_dir = Path(task_dir)
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise TaskError("%s: no task.toml" % task_dir)
    meta = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    where = str(toml_path)

    md = meta.get("metadata") or {}
    env = _require(meta, "environment", where=where)
    agent = meta.get("agent") or {}
    verifier = meta.get("verifier") or {}

    instruction_path = task_dir / "instruction.md"
    if not instruction_path.is_file():
        raise TaskError("%s: no instruction.md" % task_dir)

    tests_dir = task_dir / "tests"
    dockerfile = tests_dir / "Dockerfile"
    if not dockerfile.is_file():
        raise TaskError("%s: no tests/Dockerfile" % task_dir)
    from_image, copies, runs = parse_verifier_dockerfile(
        dockerfile.read_text(encoding="utf-8"))
    image = str(_require(env, "docker_image", where=where))
    if from_image != image:
        raise TaskError("%s: verifier FROM %s differs from environment image %s"
                        % (task_dir.name, from_image, image))
    files = []
    for src, dst in copies:
        source = tests_dir / src
        if not source.is_file():
            raise TaskError("%s: tests/Dockerfile copies %s, which is missing"
                            % (task_dir.name, src))
        files.append(VerifierFile(source=source, destination=dst))

    config_path = tests_dir / "config.json"
    f2p: Tuple[str, ...] = ()
    p2p: Tuple[str, ...] = ()
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        f2p = tuple(str(x) for x in config.get("f2p_node_ids") or ())
        p2p = tuple(str(x) for x in config.get("p2p_node_ids") or ())

    collect = tuple(
        CollectStep(command=str(step["command"]),
                    timeout_s=float(step.get("timeout_sec") or 300.0))
        for step in (verifier.get("collect") or []) if step.get("command"))
    if not collect:
        raise TaskError("%s: no [[verifier.collect]] command -- nothing would "
                        "turn the sandbox into model.patch" % task_dir.name)

    task_id = str(md.get("task_id") or task_dir.name)
    if task_id != task_dir.name:
        # The run's output directory, --instance, and --regrade all key on the
        # directory name; a metadata id that disagrees would split one task
        # into two names.
        raise TaskError("%s: metadata.task_id %r differs from the directory name"
                        % (task_dir, task_id))

    return Task(
        task_id=task_id,
        name=str((meta.get("task") or {}).get("name") or task_dir.name),
        repo_url=str(md.get("repository_url") or ""),
        language=str(md.get("language") or ""),
        base_commit=str(md.get("base_commit_hash") or ""),
        image=image,
        instruction=instruction_path.read_text(encoding="utf-8"),
        cpus=int(env.get("cpus") or 2),
        memory_mb=int(env.get("memory_mb") or 8192),
        agent_timeout_s=float(agent.get("timeout_sec") or 10800.0),
        agent_network_mode=str(agent.get("network_mode") or ""),
        verifier_timeout_s=float(verifier.get("timeout_sec") or 1800.0),
        verifier_network_mode=str(verifier.get("network_mode") or ""),
        collect=collect,
        verifier_files=tuple(files),
        verifier_run_steps=tuple(runs),
        f2p=f2p, p2p=p2p,
        task_dir=task_dir,
    )


def load_tasks(tasks_dir: Path, wanted: Optional[Iterable[str]] = None) -> List[Task]:
    """Every task under ``tasks_dir`` (or just ``wanted``), sorted by id.

    A directory without ``task.toml`` is skipped silently -- the dataset repo
    keeps non-task directories beside the tasks; a directory WITH one that fails
    to load is an error, because a task that silently drops out of a 113-task
    run changes the denominator.
    """
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        raise TaskError("tasks dir not found: %s" % tasks_dir)
    wanted_set = set(wanted) if wanted is not None else None
    tasks: List[Task] = []
    for child in sorted(tasks_dir.iterdir()):
        if not (child / "task.toml").is_file():
            continue
        if wanted_set is not None and child.name not in wanted_set:
            continue
        tasks.append(load_task(child))
    if wanted_set is not None:
        missing = wanted_set - {t.task_id for t in tasks} - {t.task_dir.name for t in tasks}
        if missing:
            raise TaskError("not in %s: %s" % (tasks_dir, ", ".join(sorted(missing))))
    return tasks
