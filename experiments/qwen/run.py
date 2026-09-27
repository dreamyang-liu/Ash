"""Run fixed-four baseline and up to seven continuations from its first failure."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "sprout", "bpo", "shepherd")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def attempts(output: Path) -> list[dict]:
    body = read(output / "summary.json")
    if "instances" in body:
        if len(body["instances"]) != 1:
            raise ValueError("Expected one task per worker")
        body = body["instances"][0]
    rows = body.get("attempts", [])
    if not rows or body.get("complete") is False or body.get("branch_schedule_complete") is False:
        raise ValueError("Incomplete evaluation")
    for row in rows:
        grade = row.get("grade") or {}
        if (row.get("status") not in (None, "completed", "timeout")
                or row.get("grading_error") or row.get("verifier_artifact_error")
                or grade.get("error") or grade.get("verifier_artifact_error")):
            raise ValueError("Actor or verifier infrastructure failure")
        if type(row.get("resolved")) is not bool:
            raise ValueError("Missing binary verdict")
    return rows


def task_digest(tasks_dir: Path, names: list[str]) -> str:
    h = hashlib.sha256()
    for name in sorted(names):
        directory = tasks_dir / name
        if not (directory / "task.toml").is_file():
            raise FileNotFoundError(directory / "task.toml")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                h.update(path.relative_to(tasks_dir).as_posix().encode())
                h.update(b"\0")
                h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()


def snapshot_ids(output: Path) -> set[str]:
    ids = set()
    for path in output.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("type") == "checkpoint.captured" and row.get("snapshot_id"):
                ids.add(row["snapshot_id"])
    for path in output.rglob("snapshot.json"):
        value = read(path).get("final_snapshot_id")
        if value:
            ids.add(value)
    return ids


def retain_parent(output: Path) -> None:
    from dataclasses import asdict
    from runstore.mini_native import reference_at
    from swebench.fork_eval import available_branch_points
    from taskwise.policy_eval import parent_journal

    journal = parent_journal(output)
    points = available_branch_points(journal)
    if not points:
        raise ValueError("Initial failure has no restorable assistant-turn checkpoint")
    records = []
    for step, point in sorted(points.items()):
        native = reference_at(journal, step, point.session_ckpt)
        if native is None:
            raise ValueError(f"Missing exact native prefix at step {step}")
        records.append({"checkpoint": asdict(point), "native": native})
    write(output / "retained-checkpoints.json", {
        "schema_version": 1, "retention": "keep-for-future-policies",
        "mode": "disk_only", "journal": str(journal),
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "points": records, "snapshot_ids": sorted(snapshot_ids(output)),
        "storage": "AgentENV content-addressed shared base plus disk deltas; native transcript stored once",
        "restore_limitations": "No RAM, live processes, tmpfs or external services; use full snapshots for such tasks",
    })


def command(args, task: str, output: Path, method: str,
            parent: Path | None = None, checkpoint: bool = True) -> list[str]:
    common = ["--model", args.model, "--model-endpoint", args.model_endpoint,
              "--model-key-env", args.model_key_env, "--runtime-bin", str(args.runtime_bin),
              "--reasoning-effort", "high", "--max-turns", str(args.max_turns),
              "--max-output-tokens", str(args.max_output_tokens)]
    terminal = args.dataset != "deepswe"
    if method in ("bpo", "shepherd"):
        cmd = [sys.executable, "-u", "-m", "taskwise.policy_eval",
               "--benchmark", "terminalbench" if terminal else "deepswe",
               "--method", method, "--instance", task, "--parent-output", str(parent),
               "--output", str(output), "--max-rollouts", str(args.max_rollouts),
               "--meta-max-tokens", str(args.max_output_tokens), *common]
    else:
        widths = [min(4, args.max_rollouts - 1)]
        if args.max_rollouts > 5:
            widths.append(args.max_rollouts - 5)
        schedule = ["--review-output-tokens", str(args.max_output_tokens),
                    "--rounds", str(len(widths) if method == "sprout" else 0),
                    "--branches", *map(str, widths), "--checkpoint-mode",
                    "disk_only" if checkpoint else "none"]
        if terminal:
            cmd = [sys.executable, "-u", "-m", "terminalbench.branch_eval",
                   "--output", str(output), "--preparation-manifest", str(args.preparation),
                   *common, *schedule]
            if parent:
                cmd += ["--parent-output", str(parent)]
        else:
            cmd = [sys.executable, "-u", "-m", "swebench.mini_branch_eval",
                   "--benchmark", "deepswe", "--instance", task, "--out", str(output),
                   "--source-commit", args.source_commit, *common, *schedule]
            if parent:
                cmd += ["--parent-from", str(parent)]
    if terminal:
        cmd += ["--task-dir", str(args.tasks_dir / task), "--dataset-digest", args.digest,
                "--server-url", args.server_url]
        if args.api_key_file:
            cmd += ["--api-key-file", str(args.api_key_file)]
        if args.image_registry:
            cmd += ["--image-registry", args.image_registry]
    else:
        cmd += ["--tasks-dir", str(args.tasks_dir)]
    return cmd


def stage(args, task: str, output: Path, method: str,
          parent: Path | None = None, checkpoint: bool = True) -> list[dict]:
    receipt_path = output.parent / (output.name + ".stage.json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("complete"):
            return attempts(output)
        raise RuntimeError(f"Incomplete stage retained at {output}; choose a fresh experiment root to retry")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"task": task, "method": method, "started_at": time.time(),
               "complete": False, "output": str(output), "source_commit": args.source_commit}
    write(receipt_path, receipt)
    try:
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "sdk"),
                   AENV_SERVER_URL=args.server_url, LITELLM_LOCAL_MODEL_COST_MAP="True")
        if args.api_key_file:
            env["AENV_API_KEY"] = args.api_key_file.read_text().strip()
        with (output.parent / (output.name + ".log")).open("wb") as log:
            code = subprocess.call(command(args, task, output, method, parent, checkpoint),
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["exit_code"] = code
        # fork_eval returns 1 for a fully graded, unresolved benchmark task.
        # Its summary must still pass all infrastructure/verdict checks below.
        allowed_codes = {0, 1} if args.dataset == "deepswe" and method in ("baseline", "sprout") else {0}
        if code not in allowed_codes:
            raise RuntimeError(f"Stage exited {code}; see {output.name}.log")
        rows = attempts(output)
        if method == "baseline" and len(rows) != 1:
            raise ValueError("Baseline stage must grade exactly one independent rollout")
        if method == "sprout" and not rows[0]["resolved"] and len(rows) < 2:
            raise ValueError("SPROUT produced no graded continuation for a failed parent")
        count = len(rows) + (1 if method in ("bpo", "shepherd") else 0)
        if count > args.max_rollouts:
            raise ValueError("Method exceeded total rollout allowance")
        receipt["complete"] = True
        return rows
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
        raise
    finally:
        receipt["finished_at"] = time.time()
        receipt["seconds"] = receipt["finished_at"] - receipt["started_at"]
        write(receipt_path, receipt)


def run_task(args, task: str) -> dict:
    root = args.output / "tasks" / task
    base = (args.baseline_root or args.output) / "tasks" / task / "baseline"
    state = {"task": task, "dataset": args.dataset, "methods": {}, "complete": False}
    try:
        baseline = []
        for index in range(1, 5):
            out = base / f"attempt-{index}"
            if args.baseline_root:
                rows = attempts(out)
                if len(rows) != 1:
                    raise ValueError("Shared baseline must contain four independent single-rollout stages")
            else:
                rows = stage(args, task, out, "baseline", checkpoint=index == 1)
            baseline.extend(rows)
            if not args.baseline_root:
                if index == 1 and not rows[0]["resolved"]:
                    try:
                        retain_parent(out)
                    except ValueError as exc:
                        # Lack of a branch point must not truncate fixed-four baseline.
                        state["retention_error"] = str(exc)
        state["baseline"] = {"output": str(base), "attempts": baseline}
        parent = base / "attempt-1"
        initial = baseline[0]["resolved"]
        state["initial_resolved"] = initial
        requested = METHODS[1:] if args.method == "all" else (() if args.method == "baseline" else (args.method,))
        if requested and not initial and not (parent / "retained-checkpoints.json").is_file():
            raise ValueError("Shared failure lacks retained-checkpoints.json")
        for method in requested:
            out = root / method
            if initial:
                state["methods"][method] = {"status": "skipped_initial_success", "attempts": []}
            else:
                rows = stage(args, task, out, method, parent)
                state["methods"][method] = {"status": "complete", "output": str(out),
                                            "attempts": rows[1:] if method == "sprout" else rows}
            write(root / f"{args.method}.task.json", state)
        state["complete"] = True
    except Exception as exc:
        state.update(error_type=type(exc).__name__, error=str(exc), status="infrastructure_error")
    write(root / f"{args.method}.task.json", state)
    print(json.dumps({"task": task, "complete": state["complete"],
                      "initial_resolved": state.get("initial_resolved"),
                      "error": state.get("error")}), flush=True)
    return state


def parse_args(argv=None):
    defaults = ROOT / "experiments/qwen/run-config.json"
    config = read(defaults) if defaults.exists() else {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("deepswe", "terminalbench21", "swebenchpro-v2-hard"), default=config.get("dataset"))
    p.add_argument("--method", choices=(*METHODS, "all"), default=config.get("method", "all"))
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--task", action="append", default=[])
    p.add_argument("--selection", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--model-endpoint", default=os.environ.get("QWEN_BASE_URL"))
    p.add_argument("--model-key-env", default="QWEN_API_KEY")
    p.add_argument("--runtime-bin", type=Path, required=True)
    p.add_argument("--server-url", default=os.environ.get("AENV_SERVER_URL", "http://127.0.0.1:8000"))
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--image-registry")
    p.add_argument("--max-turns", type=int, default=300)
    p.add_argument("--max-output-tokens", type=int, default=64000)
    p.add_argument("--max-rollouts", type=int, default=8)
    p.add_argument("--plan", action="store_true", help="Validate data and print commands without API or VM calls")
    args = p.parse_args(argv)
    if not args.dataset or args.workers < 1 or not 2 <= args.max_rollouts <= 8 or args.max_turns < 1:
        p.error("Select a dataset, positive workers/turns, and 2..8 total method rollouts")
    if not args.model_endpoint:
        p.error("Set QWEN_BASE_URL or --model-endpoint")
    if not args.model_key_env.isidentifier():
        p.error("--model-key-env must be an environment variable name")
    for key in ("tasks_dir", "output", "runtime_bin", "baseline_root", "api_key_file"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    names = args.task
    if args.selection:
        selection = json.loads(args.selection.read_text())
        names += selection if isinstance(selection, list) else next(iter(selection.values()))
    if not names:
        p.error("Explicit --task or --selection is required; subsets are never silently expanded")
    if len(set(names)) != len(names) or any(Path(n).name != n or n in (".", "..") for n in names):
        p.error("Task IDs must be unique directory names")
    args.tasks = names
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.digest = task_digest(args.tasks_dir, args.tasks)
    args.source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.preparation = args.output / "preparation.json"
    manifest = {"schema_version": 1, "dataset": args.dataset, "dataset_digest": args.digest,
                "tasks": args.tasks, "model": args.model, "reasoning_effort": "high",
                "agent": "mini-swe-agent==2.4.6", "branching": "assistant-turn",
                "baseline_rollouts": 4, "max_total_method_rollouts": args.max_rollouts,
                "max_turns": args.max_turns, "max_output_tokens": args.max_output_tokens,
                "workers": args.workers, "source_commit": args.source_commit,
                "method": args.method, "baseline_root": str(args.baseline_root) if args.baseline_root else None}
    if args.plan:
        print(json.dumps({"manifest": manifest, "example_commands": {
            method: command(args, args.tasks[0], args.output / "plan" / method, method,
                            args.output / "plan/baseline" if method != "baseline" else None)
            for method in METHODS}}, indent=2))
        return 0
    if not os.environ.get(args.model_key_env):
        raise ValueError("Model key variable is unset")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit source changes before running")
    if args.baseline_root:
        prior = read(args.baseline_root / "manifest-baseline.json")
        for key in ("dataset", "dataset_digest", "tasks", "model", "reasoning_effort", "max_turns", "max_output_tokens"):
            if prior.get(key) != manifest[key]:
                raise ValueError(f"Shared baseline configuration mismatch: {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"manifest-{args.method}.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Experiment configuration changed; use a fresh output")
    lock = args.output / f".{args.method}.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        write(manifest_path, manifest)
        if args.method == "all":
            write(args.output / "manifest-baseline.json", dict(manifest, method="baseline"))
        write(args.preparation, {"dataset": args.dataset, "dataset_digest": args.digest,
              "tasks": [{"name": tomllib.loads((args.tasks_dir / n / "task.toml").read_text())["task"]["name"], "path": str(args.tasks_dir / n),
                         "task_toml_sha256": hashlib.sha256((args.tasks_dir / n / "task.toml").read_bytes()).hexdigest()}
                        for n in args.tasks]})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            states = list(pool.map(lambda task: run_task(args, task), args.tasks))
        from experiments.qwen.report import build_report
        write(args.output / f"metrics-{args.method}.json", build_report(args.output, args.method))
        return 0 if all(s["complete"] for s in states) else 2
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
