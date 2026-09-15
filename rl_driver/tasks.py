"""Deployment-owned task bindings for rollout admission and preparation."""

from __future__ import annotations

from copy import deepcopy
import re

from rl_driver.protocol import EnvironmentRef


_COMMIT = re.compile(r"[0-9a-fA-F]{40,64}\Z")


def resolve_task(
    config: dict,
    task_id: str,
    environment_ref: EnvironmentRef,
    *,
    require_grade: bool,
) -> dict:
    """Resolve one public task ID without exposing private verifier inputs.

    The deployment catalog, not the request, binds a task to its immutable
    environment, repository baseline and grader.  The returned repository
    description contains data safe to pass to the worker; hidden tests remain
    in the frozen grader dataset.
    """
    task = config.get("tasks", {}).get(task_id)
    if task is None:
        if require_grade:
            raise ValueError(
                f"Configure Ash tasks[{task_id!r}] with environment, repository and grade"
            )
        return {}
    if not isinstance(task, dict) or set(task) - {
        "environment_ref", "repository", "grade"
    }:
        raise ValueError(
            f"Ash tasks[{task_id!r}] may contain environment_ref, repository and grade"
        )

    configured_ref = task.get("environment_ref")
    if configured_ref is None:
        if require_grade:
            raise ValueError(
                f"Ash tasks[{task_id!r}] must bind an environment_ref"
            )
    elif EnvironmentRef.from_dict(configured_ref) != environment_ref:
        raise ValueError(
            f"environment_ref does not match Ash tasks[{task_id!r}]"
        )

    repository = task.get("repository")
    if repository is None:
        if require_grade:
            raise ValueError(
                f"Ash tasks[{task_id!r}] must configure repository preflight"
            )
    else:
        _validate_repository(repository, task_id)

    grade = task.get("grade")
    if require_grade and not grade:
        raise ValueError(f"Configure Ash tasks[{task_id!r}].grade for rollout rewards")
    if grade:
        spec = grade.get("spec") if isinstance(grade, dict) else None
        if not isinstance(spec, dict) or spec.get("instance_id") != task_id:
            raise ValueError(
                f"Ash tasks[{task_id!r}].grade must target the same instance_id"
            )
    return deepcopy(task)


def _validate_repository(value: object, task_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {"workdir", "base_commit"}:
        raise ValueError(
            f"Ash tasks[{task_id!r}].repository requires workdir and base_commit"
        )
    workdir = value.get("workdir")
    commit = value.get("base_commit")
    if (
        not isinstance(workdir, str)
        or not workdir.startswith("/")
        or workdir == "/"
        or "\x00" in workdir
    ):
        raise ValueError("task repository workdir must be a non-root absolute path")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("task repository base_commit must be a full Git object ID")
