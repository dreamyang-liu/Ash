#!/usr/bin/env python3
"""Overlapped retry rounds with one global worker cap.

The sequential chain held 32 workers per ROUND, so one slow straggler at a
round's tail idled 31 slots. Rounds are independent samples over the same
failure list, so the right shape is a flat queue: every (round, instance) pair
is one job, 32 run at any moment, and a slot frees the moment its job ends.

Each job is its own fork_eval process with a hard subprocess timeout, which
also retires the hung-tail problem at the job level: a worker that lingers
after finishing gets killed by its own wrapper, not by a chain-level sweep.

Resumable: a job whose summary.json already exists is skipped, so re-running
this script continues where it stopped (and mops up a killed chain's partial
rounds, whatever layout they used).

    setsid nohup python3.11 -u scripts/run_adaptive_queue.py \
        > runs/adaptive/queue.log 2>&1 &
"""

from __future__ import annotations

import json
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

A = Path(__file__).resolve().parents[1]
BASE = A / "runs" / "adaptive"
ROUNDS = (1, 2, 3, 4, 5)
WORKERS = 32
#: fork_eval's own per-run timeout is 1200s; grading adds minutes. The wrapper
#: kill fires only when the job process itself is wedged.
JOB_TIMEOUT_S = 2400


def done_instances(round_no: int) -> set:
    """Instances this round already graded, in either layout (shards or queue)."""
    finished = set()
    for pattern in ("retry%d/shard-*/summary.json" % round_no,
                    "retry%d/q/*/summary.json" % round_no):
        for summary in glob.glob(str(BASE / pattern)):
            try:
                for entry in json.load(open(summary))["instances"]:
                    finished.add(entry["instance"])
            except (ValueError, KeyError):
                pass
    return finished


def run_job(round_no: int, instance: str) -> tuple:
    out = BASE / ("retry%d" % round_no) / "q" / instance
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "python3.12", "-u", "-m", "swebench.fork_eval",
        "--instance", instance, "--slot", "claude-code",
        "--model", "us.anthropic.claude-sonnet-4-6",
        "--rounds", "0", "--timeout", "1200", "-o", str(out),
    ]
    log = (out / "run.log").open("w")
    try:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       timeout=JOB_TIMEOUT_S, cwd=A,
                       env={**os.environ, "PYTHONPATH": ".:sdk"},
                       start_new_session=True, check=False)
    except subprocess.TimeoutExpired:
        return round_no, instance, "wrapper-timeout"
    finally:
        log.close()
    try:
        entry = json.load(open(out / "summary.json"))["instances"][0]
        return round_no, instance, "resolved" if entry["resolved"] else "failed"
    except (OSError, ValueError, KeyError, IndexError):
        return round_no, instance, "no-summary"


def main() -> int:
    failed = (BASE / "failed.txt").read_text().strip().split(",")
    jobs = []
    for round_no in ROUNDS:
        remaining = [i for i in failed if i not in done_instances(round_no)]
        print("round %d: %d of %d remaining" % (round_no, len(remaining),
                                                len(failed)), flush=True)
        jobs += [(round_no, instance) for instance in remaining]
    print("queue: %d jobs, %d workers" % (len(jobs), WORKERS), flush=True)

    tally = {"resolved": 0, "failed": 0, "wrapper-timeout": 0, "no-summary": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_job, r, i) for r, i in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            round_no, instance, verdict = future.result()
            tally[verdict] += 1
            print("[%d/%d] r%d %-34s %s" % (n, len(jobs), round_no,
                                            instance, verdict), flush=True)
    print("done:", tally, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
