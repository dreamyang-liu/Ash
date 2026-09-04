"""The grader gate: their oracle must score 1 and doing nothing must score 0.

Not a test of *their* grader -- of *our transport around it*: snapshot restore,
the collect command, file placement, the offline verifier VM, reading
reward.json. Each check takes the exact path a real attempt takes
(``deepswe.grade.grade_snapshot``), starting from a snapshot we make here:

    oracle   fresh VM from the task image, run the task's own solution/solve.sh
             (applies solution.patch, commits on a branch), snapshot  -> must be 1
    nop      fresh VM from the task image, snapshot untouched            -> must be 0

Both outcomes are required. A grader that passes the oracle but also passes
nothing is measuring nothing; one that fails the oracle is broken plumbing.

    python -m deepswe.gate --tasks-dir ~/projects/LBP/deep-swe/tasks \\
        -o runs/deepswe-gate --workers 8            # all 113, resumable
    python -m deepswe.gate ... --instance ytt-jsonpath-query-api --mode oracle

Snapshots created here are named ``deepswe-gate-<mode>-<task>-<stamp>`` and are
never deleted by this script (nothing in this repository deletes history
automatically; see ``harness sweep``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from harness.core.journal import volatile_reason
from harness.execution.session import SandboxSession
from deepswe.grade import grade_snapshot, shell_parts
from deepswe.tasks import Task, load_tasks

SOLUTION_DIR = "/solution"
MODES = ("oracle", "nop")


def backend_for(runtime_bin: str, tasks_dir: str) -> dict:
    """Exactly the backend a real DeepSWE attempt gets -- built by the same code.

    The gate once had its own copy with `allow_internet` only; it then verified
    226 checks on templates WITHOUT the image's ENV while the real runs would
    have used templates with it. A gate that exercises a different path than
    the runs proves nothing about the runs.
    """
    from types import SimpleNamespace
    from swebench.fork_eval import backend_for as loop_backend
    from deepswe.bench import DeepSWE
    return loop_backend(SimpleNamespace(runtime_bin=runtime_bin), DeepSWE(tasks_dir))


def make_snapshot(task: Task, mode: str, backend: dict, stamp: str) -> Dict[str, object]:
    """A snapshot of the task image after the oracle ran (or after nothing did)."""
    session = SandboxSession(quiet=True, backend=dict(backend))
    if not session.create(task.image, {"cpu": task.cpus, "memory_mb": task.memory_mb}):
        return {"error": "could not start %s: %s" % (task.image, session.create_error)}
    try:
        if mode == "oracle":
            if not task.solve_script:
                return {"error": "task has no solution/solve.sh"}
            session.execute("shell", {"command": "mkdir -p %s" % SOLUTION_DIR,
                                      "timeout": 60})
            for item in sorted(task.solution_dir.iterdir()):
                if item.is_file() and not session.upload_file(
                        item, "%s/%s" % (SOLUTION_DIR, item.name)):
                    return {"error": "could not upload %s" % item.name}
            result = session.execute(
                "shell", {"command": "bash %s/solve.sh" % SOLUTION_DIR, "timeout": 900},
                timeout=960)
            out, err, rc = shell_parts(result)
            if rc != 0:
                return {"error": "solve.sh exited %d: %s" % (rc, (err or out)[-1500:])}
        snapshot = session.snapshot(name="deepswe-gate-%s-%s-%s" % (mode, task.task_id, stamp))
        if snapshot is None:
            return {"error": "snapshot failed"}
        return {"snapshot_id": snapshot.id}
    finally:
        session.destroy()


def check(task: Task, mode: str, backend: dict, stamp: str) -> dict:
    started = time.time()
    record: dict = {"task": task.task_id, "mode": mode, "language": task.language}
    made = make_snapshot(task, mode, backend, stamp)
    record.update(made)
    if "snapshot_id" in made:
        grade = grade_snapshot(str(made["snapshot_id"]), task, backend)
        expected = mode == "oracle"
        record.update({
            "resolved": grade.resolved, "expected": expected,
            "ok": grade.error is None and grade.resolved == expected,
            "grading_error": grade.error,
            "f2p_pass": grade.f2p_pass, "p2p_pass": grade.p2p_pass,
            "patch_lines": grade.patch.count("\n"),
            "detail_tail": grade.detail[-1500:],
        })
    else:
        record["ok"] = False
    record["seconds"] = round(time.time() - started, 1)
    return record


def already_ok(log_path: Path) -> set:
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("ok"):
                done.add((rec["task"], rec["mode"]))
    return done


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--instance", default="", help="comma list; default all")
    parser.add_argument("--mode", default="both", choices=["both", *MODES])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    parser.add_argument("-o", "--out", default="runs/deepswe-gate")
    parser.add_argument("--volatile-ok", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    reason = volatile_reason(out_dir)
    if reason and not args.volatile_ok:
        raise SystemExit("refusing: %s (pass --volatile-ok to override)" % reason)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "gate.jsonl"

    wanted = [x.strip() for x in args.instance.split(",") if x.strip()] or None
    tasks = load_tasks(args.tasks_dir, wanted)
    modes = list(MODES) if args.mode == "both" else [args.mode]
    done = already_ok(log_path)
    # One job per TASK, modes in sequence inside it. Two jobs on the same task
    # at once each try to build the same per-image template, and the second
    # spawns from a template whose build has not committed yet -- measured as
    # HTTP 500 "resolve committed snapshot into runnable runtime paths" on 8 of
    # the first 14 checks when oracle and nop of one task ran side by side.
    jobs = [(t, [m for m in modes if (t.task_id, m) not in done]) for t in tasks]
    jobs = [(t, ms) for t, ms in jobs if ms]
    n_checks = sum(len(ms) for _, ms in jobs)
    print("%d task(s) x %s: %d check(s) to run, %d already ok"
          % (len(tasks), "/".join(modes), n_checks, len(done)))

    backend = backend_for(args.runtime_bin, args.tasks_dir)
    print("backend:", json.dumps(backend["microvm"], sort_keys=True))
    stamp = time.strftime("%Y%m%d%H%M%S")
    failures = []

    def check_task(task: Task, task_modes: List[str]) -> List[dict]:
        records = []
        for mode in task_modes:
            try:
                records.append(check(task, mode, backend, stamp))
            except Exception as exc:  # noqa: BLE001 - one check must not sink the batch
                records.append({"task": task.task_id, "mode": mode, "ok": False,
                                "error": "%s: %s" % (type(exc).__name__, exc)})
        return records

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            log_path.open("a", encoding="utf-8") as log:
        futures = {pool.submit(check_task, t, ms): t for t, ms in jobs}
        for future in as_completed(futures):
            for record in future.result():
                n += 1
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                mark = "ok  " if record.get("ok") else "FAIL"
                print("[%d/%d] %s %-8s %-45s %s" % (
                    n, n_checks, mark, record["mode"], record["task"],
                    record.get("grading_error") or record.get("error")
                    or "resolved=%s" % record.get("resolved")), flush=True)
                if not record.get("ok"):
                    failures.append(record)

    done = already_ok(log_path)
    summary = {
        "tasks": len(tasks), "modes": modes,
        "ok": sorted("%s/%s" % k for k in done if k[0] in {t.task_id for t in tasks}),
        "failures": [{k: v for k, v in f.items() if k != "detail_tail"} for f in failures],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nGATE: %d/%d checks ok; %d failed this run -> %s"
          % (len(summary["ok"]), len(tasks) * len(modes), len(failures), out_dir))
    for f in failures:
        print("  FAIL %-8s %-45s %s" % (f["mode"], f["task"],
                                       f.get("grading_error") or f.get("error")
                                       or "resolved=%s expected=%s"
                                       % (f.get("resolved"), f.get("expected"))))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
