#!/usr/bin/env python3
"""Re-grade only the instances whose verdict rested on a dead test collection.

Surgical on purpose: the docstring-id fix changes the grading commands ONLY for
instances that carry prose ids, so the other verdicts are untouched by the fix
and re-running them would spend hours re-verifying what did not change. Each
poisoned instance gets its last snapshot restored and graded by the CURRENT
grader; per-instance state means a crash resumes where it stopped.

    python scripts/regrade_poisoned.py --workers 8
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness.rollback import load_checkpoints
from swebench.dataset import (load_instances, malformed_test_ids,
                              parse_test_list, test_files_of)
from swebench.fork_eval import grade_snapshot

A = Path(__file__).resolve().parents[1]
STATE = A / "runs/v500/regrade-round2.jsonl"
BACKEND = {"backend": "microvm",
           "microvm": {"from_image": True,
                       "runtime_bin": str(A / "runtime" / "ash-runtime")}}


def poisoned_instances() -> dict:
    """instance_id -> shard, for every verdict the grader upgrade can move.

    Round 2 sweeps EVERY non-resolved instance (round 1's flips stay resolved:
    both fixes only widen what can pass). Two grader changes moved the goalposts
    -- django by output-parsing, and test-file edits reverted instead of fatal --
    so the affected set is "everything that failed", not one poisoned list.
    Instances already flipped by round 1 are re-checked too; they can only
    confirm.
    """
    flipped = set()
    round1 = A / "runs/v500/regrade-poisoned.jsonl"
    if round1.exists():
        for line in round1.read_text().splitlines():
            record = json.loads(line)
            if record.get("resolved"):
                flipped.add(record["instance"])
    out = {}
    for summary in sorted(glob.glob(str(A / "runs/v500/shard-*/summary.json"))):
        shard = summary.rsplit("/", 2)[-2]
        for entry in json.load(open(summary))["instances"]:
            if entry["resolved"] or entry["instance"] in flipped:
                continue
            out[entry["instance"]] = shard
    return out


def regrade_one(instance_id: str, shard: str, catalogue: dict) -> dict:
    raw = catalogue[instance_id]
    instance = {
        "instance_id": instance_id, "repo": raw["repo"],
        "f2p": parse_test_list(raw["FAIL_TO_PASS"]),
        "p2p": parse_test_list(raw["PASS_TO_PASS"]),
        "test_files": test_files_of(raw),
        "test_patch": raw.get("test_patch") or "",
    }
    journal = A / "runs/v500" / shard / instance_id / "parent.jsonl"
    pairs = [c for c in load_checkpoints(str(journal)) if c.snapshot_id]
    if not pairs:
        return {"instance": instance_id, "resolved": False,
                "error": "no snapshot"}
    grade = grade_snapshot(pairs[-1].snapshot_id, instance, BACKEND)
    return {"instance": instance_id, "resolved": grade.resolved,
            "f2p_pass": grade.f2p_pass, "p2p_ran": grade.p2p_ran,
            "p2p_pass": grade.p2p_pass,
            "skipped_ids": len(grade.skipped_ids),
            "broken": grade.broken[:5], "error": grade.error,
            "summary": grade.summary()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    done = set()
    if STATE.exists():
        done = {json.loads(l)["instance"] for l in STATE.read_text().splitlines()}
    targets = {k: v for k, v in poisoned_instances().items() if k not in done}
    catalogue = {i["instance_id"]: i for i in load_instances("verified")}
    print("poisoned, not yet regraded: %d" % len(targets), flush=True)

    flipped = 0
    with STATE.open("a", encoding="utf-8") as state, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(regrade_one, iid, shard, catalogue): iid
                   for iid, shard in targets.items()}
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            state.write(json.dumps(record, ensure_ascii=False) + "\n")
            state.flush()
            mark = "FLIP->PASS" if record["resolved"] else "still fail"
            if record["resolved"]:
                flipped += 1
            print("[%d/%d] %-11s %-34s %s"
                  % (n, len(targets), mark, record["instance"],
                     record.get("summary") or record.get("error")), flush=True)
    print("\nflipped to resolved: %d" % flipped, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
