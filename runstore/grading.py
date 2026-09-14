"""Queued official evaluators. Unresolved is a completed result, not a retry."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from runstore.files import read_json, write_json
from runstore.specs import GradeSpec


def dataset_rows(spec: GradeSpec) -> list[dict]:
    data = Path(spec.dataset_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.dataset_sha256:
        raise ValueError("Frozen dataset checksum changed")
    if spec.dataset_path.endswith(".jsonl"):
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    rows = json.loads(data)
    if not isinstance(rows, list):
        raise ValueError("Frozen dataset must contain a list of tasks")
    return rows


def grade(body: dict, directory: Path, *, session_factory) -> dict:
    spec = GradeSpec(**body)
    rows = dataset_rows(spec)
    selected = [row for row in rows if row.get("instance_id") == spec.instance_id]
    if len(selected) != 1:
        raise ValueError("Frozen dataset does not identify exactly one grading task")
    backend = {**spec.backend, "microvm": {**spec.backend.get("microvm", {}),
               "allow_internet": spec.verifier_network == "allow"}}
    if spec.benchmark == "deepswe":
        from runstore.deepswe import grade_snapshot

        return grade_snapshot(spec, selected[0], backend, directory, session_factory=session_factory)
    if spec.benchmark == "swe-rebench-v2":
        from runstore.swerebench import grade_snapshot

        return grade_snapshot(spec, selected[0], backend, directory, session_factory=session_factory)
    if spec.benchmark == "swebench-pro":
        from swebench_pro.grade import grade_snapshot
        from swebench_pro.tasks import HARNESS_REVISION, task_from_row, validate_repo

        if spec.grader_revision != HARNESS_REVISION or not spec.harness_repo:
            raise ValueError("Pro official grader revision differs from the pinned adapter")
        repo = Path(spec.harness_repo)
        validate_repo(repo)
        task = task_from_row(selected[0], repo, f"sha256:{spec.dataset_sha256}")
        verdict = grade_snapshot(spec.snapshot_id, task, backend, resources=spec.resources,
                                 timeout=int(spec.timeout_s), artifacts_dir=directory / "grading",
                                 session_factory=session_factory)
        return {"status": "error" if verdict.error else "completed", "grade": asdict(verdict),
                "resolved": bool(verdict.resolved) if not verdict.error else None,
                "failure_kind": "infrastructure" if verdict.error else None,
                "error": verdict.error, "grader_revision": HARNESS_REVISION}
    return grade_verified(spec, selected[0], backend, directory, session_factory=session_factory)


def grade_verified(spec: GradeSpec, row: dict, backend: dict, directory: Path, *, session_factory) -> dict:
    from swebench.patch import baseline_untracked, extract_patch
    from swebench_pro.grade import checked

    request = {"spec": {**asdict(spec), "backend": {}}, "task": row, "run_id": "rs-" + directory.name,
               "directory": str(directory.resolve())}
    request_path = directory / "verified.json"
    write_json(request_path, request)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    runner = Path(__file__).with_name("official_verified.py")
    subprocess.run([sys.executable, str(runner), str(request_path), "--describe"],
                   check=True, cwd=directory, env=env, timeout=120)
    description = read_json(directory / "verified-description.json")
    baseline_session = session_factory(quiet=True, backend=backend)
    collector = session_factory(quiet=True, backend=backend)
    try:
        if not baseline_session.create(description["image"], spec.resources):
            raise RuntimeError("Could not create pristine official Verified image")
        def baseline_shell(command: str):
            return SimpleNamespace(success=True, output=checked(baseline_session, "cd /testbed && " + command))
        baseline = baseline_untracked(baseline_shell)
        if not collector.create(spec.snapshot_id, spec.resources):
            raise RuntimeError("Could not restore grading snapshot")
        def shell(command: str):
            return SimpleNamespace(success=True, output=checked(collector, "cd /testbed && " + command))
        patch, added = extract_patch(shell, row["base_commit"], baseline)
    finally:
        collector.destroy()
        baseline_session.destroy()
    patch_path = directory / "submission.patch"
    patch_path.write_text(patch)
    if not patch.strip():
        return {"status": "completed", "resolved": False, "reason": "empty_patch",
                "patch": str(patch_path), "grader_revision": spec.grader_revision}
    with (directory / "official.log").open("w") as log:
        completed = subprocess.run([sys.executable, str(runner), str(request_path)], cwd=directory,
                                   env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=spec.timeout_s + 600)
    report_path = (directory / "logs/run_evaluation" / request["run_id"] / "runstore"
                   / spec.instance_id / "report.json")
    report = read_json(report_path)
    verdict = report.get(spec.instance_id) if report else None
    if completed.returncode or not isinstance(verdict, dict) or type(verdict.get("resolved")) is not bool:
        return {"status": "error", "resolved": None, "failure_kind": "infrastructure",
                "error": "Official evaluator did not produce a valid report", "log": str(directory / "official.log")}
    return {"status": "completed", "resolved": verdict["resolved"], "official_report": verdict,
            "report_path": str(report_path), "patch": str(patch_path), "grader_revision": spec.grader_revision}
