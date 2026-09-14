"""Resumable orchestration and coverage accounting for official SWE-bench reports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def report_path(root: Path, run_id: str, prediction: dict) -> Path:
    model = prediction["model_name_or_path"].replace("/", "__")
    return (root / "official/logs/run_evaluation" / run_id / model
            / prediction["instance_id"] / "report.json")


def read_report(path: Path, instance_id: str) -> dict | None:
    if not path.exists():
        return None
    try:
        body = json.loads(path.read_text())
        item = body.get(instance_id) if isinstance(body, dict) else None
    except (ValueError, OSError):
        item = None
    if not isinstance(item, dict) or type(item.get("resolved")) is not bool:
        raise ValueError(f"Invalid official report: {path}")
    return item


def pending_predictions(root: Path, run_id: str, path: Path) -> list[dict]:
    predictions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [prediction["instance_id"] for prediction in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate instance IDs in {path}")
    return [prediction for prediction in predictions
            if prediction.get("model_patch") and read_report(
                report_path(root, run_id, prediction), prediction["instance_id"]
            ) is None]


def cleanup_run_containers(run_id: str, timeout: int = 60) -> None:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={run_id}",
         "--format", "{{json .}}"],
        capture_output=True, text=True, check=True, timeout=timeout,
    )
    containers = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    containers = [item for item in containers if item["Names"].endswith(f".{run_id}")]
    if any("removal" in item.get("Status", "").lower() for item in containers):
        raise RuntimeError(f"Docker container removal still pending for {run_id}")
    if containers:
        subprocess.run(["docker", "rm", "-f", *[item["ID"] for item in containers]],
                       check=True, timeout=timeout, capture_output=True, text=True)


def run_batches(root: Path, prefix: str, dataset: str, split: str,
                batch_files: dict[str, str], args: Any) -> None:
    selected = set(args.batches or batch_files)
    if selected - batch_files.keys():
        raise ValueError(f"Unknown batches: {sorted(selected - batch_files.keys())}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["ASH_DOCKER_CLIENT_TIMEOUT"] = str(args.docker_client_timeout)
    env["ASH_DOCKER_MAX_POOL_SIZE"] = str(args.docker_max_pool_size)
    eval_dir = root / "official"
    eval_dir.mkdir(exist_ok=True)
    runner = Path(__file__).with_name("official_swebench_runner.py")
    incomplete_batches = []
    for batch in sorted(selected, key=lambda name: (name != "parent", name)):
        run_id = f"{prefix}-{batch}"
        predictions_path = Path(batch_files[batch])
        for retry in range(args.retry_rounds + 1):
            pending = pending_predictions(root, run_id, predictions_path)
            if not pending:
                break
            workers = args.retry_workers if retry or args.phase == "retry" else args.max_workers
            label = f"{batch}-{time.time_ns()}"
            pending_path = eval_dir / f"{label}.pending.jsonl"
            pending_path.write_text("".join(json.dumps(item) + "\n" for item in pending))
            command = [sys.executable, str(runner), "--dataset_name", dataset, "--split", split,
                       "--predictions_path", str(pending_path.resolve()), "--run_id", run_id,
                       "--namespace", args.namespace, "--max_workers", str(workers),
                       "--startup-workers", str(args.startup_workers),
                       "--timeout", str(args.timeout), "--cache_level", args.cache_level,
                       "--clean", str(args.clean).lower()]
            event = {"batch": batch, "run_id": run_id, "retry": retry, "workers": workers,
                     "pending": len(pending), "predictions_path": str(pending_path)}
            try:
                cleanup_run_containers(run_id, args.cleanup_timeout)
                prep_log = eval_dir / f"{label}.prepare.log"
                prep_report = eval_dir / f"{label}.images.jsonl"
                event.update(prepare_log=str(prep_log), prepare_report=str(prep_report))
                print(f"preparing {batch}: {len(pending)} pending, {args.prepare_workers} workers", flush=True)
                with prep_log.open("w") as log:
                    prepared = subprocess.run(
                        [*command, "--prepare-only", "--prepare-workers", str(args.prepare_workers),
                         "--prepare-report", str(prep_report.resolve())],
                        cwd=eval_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                    )
                event["prepare_exit_code"] = prepared.returncode
                if prepared.returncode == 0:
                    log_path = eval_dir / f"{label}.eval.log"
                    event["log_path"] = str(log_path)
                    print(f"evaluating {batch}: {len(pending)} pending, {workers} workers", flush=True)
                    with log_path.open("w") as log:
                        evaluated = subprocess.run(command, cwd=eval_dir, env=env,
                                                   stdout=log, stderr=subprocess.STDOUT)
                    event["exit_code"] = evaluated.returncode
                else:
                    event["error"] = "image preparation incomplete; tests not started"
            except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with (eval_dir / "retry-runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
            if event.get("exit_code", 0) != 0:
                raise RuntimeError(f"Official harness process failed; see {event['log_path']}")
        missing = pending_predictions(root, run_id, predictions_path)
        if missing:
            incomplete_batches.append(batch)
            if args.stop_on_error:
                break
    if incomplete_batches:
        raise RuntimeError(f"Evaluation incomplete after bounded retries: {incomplete_batches}")


def summarize_attempts(root: Path, prefix: str, attempts: list[dict],
                       expected_ids: set[str]) -> tuple[dict, list[dict]]:
    rows = []
    for record in attempts:
        row = {**record, "official_resolved": None, "official_report": None}
        prediction = record.get("prediction") or {}
        if not record.get("ok"):
            row["official_status"] = "export_error"
        elif not prediction.get("model_patch"):
            row.update(official_status="empty_patch", official_resolved=False)
        else:
            batch = ("parent" if record["role"] == "parent"
                     else f"branch-slot-{int(record['branch_slot']):02d}")
            path = report_path(root, f"{prefix}-{batch}", prediction)
            row["official_report"] = str(path)
            item = read_report(path, record["task"])
            if item is None:
                row["official_status"] = "missing_report"
                log = path.with_name("run_instance.log")
                row["official_error_log"] = str(log) if log.exists() else None
            else:
                row.update(official_status="completed", official_resolved=item["resolved"],
                           official_tests_status=item.get("tests_status"))
        rows.append(row)
    parents = [row for row in rows if row["role"] == "parent"]
    parent_ids = [row["task"] for row in parents]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Duplicate parent tasks; choose the parent cohort explicitly")
    if {row["task"] for row in rows} - expected_ids:
        raise ValueError("Exported tasks outside the expected cohort")
    summary: dict[str, Any] = {"expected_tasks": len(expected_ids)}
    for role, label in [("parent", "parent"), ("branch", "branch")]:
        selected = [row for row in rows if row["role"] == role]
        counts = Counter(row["official_status"] for row in selected)
        resolved = sum(row["official_resolved"] is True for row in selected)
        summary.update({
            f"{label}_attempts": len(selected),
            f"{label}_submitted": counts["completed"] + counts["missing_report"],
            f"{label}_completed": counts["completed"],
            f"{label}_resolved": resolved,
            f"{label}_unresolved": counts["completed"] - resolved,
            f"{label}_missing_reports": counts["missing_report"],
            f"{label}_export_errors": counts["export_error"],
            f"{label}_empty_patches": counts["empty_patch"],
            f"{label}_resolved_completed_rate": resolved / counts["completed"]
            if counts["completed"] else None,
        })
    parent_resolved = {row["task"] for row in parents if row["official_resolved"] is True}
    branch_resolved = {row["task"] for row in rows
                       if row["role"] == "branch" and row["official_resolved"] is True}
    missing_parents = expected_ids - set(parent_ids)
    incomplete = [row for row in rows
                  if row["official_status"] in {"missing_report", "export_error"}]
    parent_complete = not missing_parents and not any(
        row["role"] == "parent" for row in incomplete)
    summary.update(
        attempt_reports=summary["parent_completed"] + summary["branch_completed"],
        missing_attempt_reports=summary["parent_missing_reports"] + summary["branch_missing_reports"],
        total_parent_tasks_seen=len(parent_ids),
        missing_parent_ids=sorted(missing_parents),
        parent_complete=parent_complete,
        parent_resolved_total_lower_bound=len(parent_resolved) / len(expected_ids) if expected_ids else None,
        parent_final_rate=len(parent_resolved) / len(expected_ids)
        if expected_ids and parent_complete else None,
        combined_resolved_tasks=len(parent_resolved | branch_resolved),
        branch_any_resolved_tasks=len(branch_resolved),
        branch_tasks_submitted=len({row["task"] for row in rows if row["role"] == "branch"
                                   and row["official_status"] in {"completed", "missing_report"}}),
        exported_attempts_complete=not incomplete,
        resolved_parent_ids=sorted(parent_resolved),
        resolved_by_branch_ids=sorted(branch_resolved),
        pending_attempts=[{key: row[key] for key in
                           ("source", "task", "attempt", "role", "official_status", "official_report")}
                          for row in incomplete],
        rate_note="Completed-only rates are diagnostic, not benchmark scores. Missing reports "
                  "are unmeasured (infrastructure failure or unfinished); export errors are unmeasured. "
                  "An empty patch is unresolved. Final parent rate requires the full expected cohort.",
    )
    return summary, rows
