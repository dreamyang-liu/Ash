"""Pinned public data and official per-instance verifier assets."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

DATASET = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
HARNESS_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"


def string_list(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid {field} list") from exc
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Invalid {field} list")
    return tuple(value)


def validate_repo(repo: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != HARNESS_REVISION:
        raise ValueError(f"Pro harness must be pinned to {HARNESS_REVISION}; found {revision}")
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--",
                    "swe_bench_pro_eval.py", "helper_code", "run_scripts", "dockerfiles"],
                   check=True, capture_output=True)


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    base_commit: str
    image: str
    problem: str
    f2p: tuple[str, ...]
    p2p: tuple[str, ...]
    sample_json: str
    harness_repo: Path
    data_source: str

    @property
    def sample(self) -> dict:
        return json.loads(self.sample_json)

    @property
    def provenance(self) -> dict:
        paths = ["swe_bench_pro_eval.py", "helper_code/image_uri.py",
                 f"run_scripts/{self.instance_id}/run_script.sh",
                 f"run_scripts/{self.instance_id}/parser.py",
                 f"dockerfiles/base_dockerfile/{self.instance_id}/Dockerfile",
                 f"dockerfiles/instance_dockerfile/{self.instance_id}/Dockerfile"]
        return {"dataset": DATASET, "data_source": self.data_source,
                "sample_sha256": hashlib.sha256(self.sample_json.encode()).hexdigest(),
                "harness_revision": HARNESS_REVISION, "image": self.image,
                "asset_sha256": {path: hashlib.sha256((self.harness_repo / path).read_bytes()).hexdigest()
                                 for path in paths}}


def task_from_row(row: dict, repo: Path, source: str) -> Task:
    sample = dict(row)
    instance_id = sample.get("instance_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", instance_id):
        raise ValueError(f"Invalid instance_id: {instance_id!r}")
    base = sample.get("base_commit", "")
    if not re.fullmatch(r"[a-f0-9]{40}", base):
        raise ValueError(f"Invalid base_commit for {instance_id}")
    tag = sample.get("dockerhub_tag", "")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError(f"Invalid dockerhub_tag for {instance_id}")
    for field in ("repo", "problem_statement", "before_repo_set_cmd"):
        if not isinstance(sample.get(field), str) or not sample[field].strip():
            raise ValueError(f"Missing {field} for {instance_id}")
    lists = {field: string_list(sample.get(field), field)
             for field in ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run")}
    if not lists["fail_to_pass"] or not lists["selected_test_files_to_run"]:
        raise ValueError(f"No target tests for {instance_id}")
    for field, values in lists.items():
        sample[field] = repr(list(values))
    problem = sample["problem_statement"]
    for field, title in (("requirements", "Requirements"), ("interface", "Interface")):
        value = sample.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Invalid {field} for {instance_id}")
        if value and value.strip() and value.strip().lower() not in {"nan", "none", "n/a"}:
            problem += f"\n\n## {title}\n{value}"
    task = Task(instance_id, sample["repo"], base, f"jefzda/sweap-images:{tag}",
                problem, lists["fail_to_pass"], lists["pass_to_pass"],
                json.dumps(sample, sort_keys=True), repo, source)
    task.provenance
    return task


def load_tasks(repo: Path, data_path: Path | None = None,
               revision: str = DATASET_REVISION) -> dict[str, Task]:
    repo = repo.expanduser().resolve()
    validate_repo(repo)
    if data_path is None:
        from datasets import load_dataset

        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Pro dataset revision must be a full commit SHA")
        rows = load_dataset(DATASET, split="test", revision=revision)
        source = f"{DATASET}@{revision}/test"
    else:
        data_path = data_path.expanduser().resolve()
        source = f"{data_path}:sha256:{hashlib.sha256(data_path.read_bytes()).hexdigest()}"
        if data_path.suffix == ".csv":
            with data_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        elif data_path.suffix == ".jsonl":
            rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
        else:
            raise ValueError("--pro-data must be CSV or JSONL")
    tasks = {}
    for row in rows:
        task = task_from_row(dict(row), repo, source)
        if task.instance_id in tasks:
            raise ValueError(f"Duplicate Pro task: {task.instance_id}")
        tasks[task.instance_id] = task
    if not tasks:
        raise ValueError("Empty Pro dataset")
    return tasks
