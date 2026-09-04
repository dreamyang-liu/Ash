#!/usr/bin/env python3
"""Would the resolved patch still pass if the agent's edits to PRE-EXISTING test
files were discarded (the SWE-bench public-leaderboard convention)?

    python3.11 scripts/deepswe_p2p_original_check.py runs/deepswe-audit-testedits.txt \
        --single runs/deepswe-final.json --branch runs/deepswe-branch -o runs/deepswe-p2p-original.jsonl

For each task in the audit shortlist: restore the resolving snapshot, collect
model.patch exactly as the grader does, DROP the diff blocks for the listed
test files (so the repository's original tests run instead of the agent's
edited copies), then run their verifier on the filtered patch. reward.json then
answers: do the original p2p tests pass against the agent's implementation?
A drop from 1 to 0 means the edit was load-bearing -- the agent's change broke
tests the repository already had, and editing them is what made p2p green.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from harness.execution.session import SandboxSession
from harness.rollback import load_checkpoints
from swebench.fork_eval import backend_for
from deepswe.bench import DeepSWE
from deepswe.grade import collect_patch, verify_patch


def parse_shortlist(path: Path) -> dict:
    """task -> set of edited pre-existing test files, from the audit report."""
    out = {}
    task = None
    for line in path.read_text().splitlines():
        m = re.match(r"^== (\S+)", line)
        if m:
            task = m.group(1)
            out.setdefault(task, set())
            continue
        m = re.match(r"^\s+\[[^\]]*\] (\S+) \(", line)
        if m and task:
            out[task].add(m.group(1))
    return {t: sorted(f) for t, f in out.items() if f}


def resolving_journal(task: str, single: dict, branch_dir: Path):
    for rec in single["tasks"]:
        if rec["task"] == task and rec["resolved"]:
            return Path(rec["journal"]), "parent"
    for summary in branch_dir.glob("shard-*/summary.json"):
        for inst in json.loads(Path(summary).read_text()).get("instances", []):
            if inst["instance"] == task:
                for a in inst["attempts"]:
                    if a["resolved"]:
                        return Path(a["journal"]), a["name"]
    return None, None


def drop_files_from_patch(patch: str, files: list) -> tuple:
    """Remove whole `diff --git` blocks for the given paths. Returns (patch, dropped)."""
    blocks = re.split(r"(?m)^(?=diff --git )", patch)
    kept, dropped = [], []
    for block in blocks:
        if not block.strip():
            continue
        m = re.match(r"diff --git a/(\S+) b/(\S+)", block)
        path = m.group(2) if m else ""
        if path in files:
            dropped.append(path)
        else:
            kept.append(block)
    return "".join(kept), dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shortlist")
    parser.add_argument("--single", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--tasks-dir", default=str(Path.home() / "projects/LBP/deep-swe/tasks"))
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    parser.add_argument("--only", default="")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("-o", "--out", default="runs/deepswe-p2p-original.jsonl")
    args = parser.parse_args()

    bench = DeepSWE(args.tasks_dir)
    catalogue = bench.catalogue(None)
    backend = backend_for(SimpleNamespace(runtime_bin=args.runtime_bin), bench)
    single = json.loads(Path(args.single).read_text())
    shortlist = parse_shortlist(Path(args.shortlist))
    only = {x for x in args.only.split(",") if x}
    jobs = [(t, f) for t, f in shortlist.items() if not only or t in only]
    print("%d task(s) to check" % len(jobs), flush=True)

    def one(task, files):
        started = time.time()
        journal, attempt = resolving_journal(task, single, Path(args.branch))
        rec = {"task": task, "attempt": attempt, "dropped_test_files": files}
        if journal is None:
            rec["error"] = "no resolving journal"
            return rec
        pairs = [c for c in load_checkpoints(journal) if c.snapshot_id]
        session = SandboxSession(quiet=True, backend=dict(backend))
        if not session.create(pairs[-1].snapshot_id):
            rec["error"] = "restore failed: " + session.create_error
            return rec
        try:
            patch, diag = collect_patch(session, catalogue[task])
        finally:
            session.destroy()
        filtered, dropped = drop_files_from_patch(patch, files)
        rec.update({"patch_lines": patch.count("\n"), "filtered_patch_lines": filtered.count("\n"),
                    "actually_dropped": dropped})
        outcome = verify_patch(catalogue[task], filtered, backend)
        r = outcome.reward or {}
        rec.update({"reward": r, "verifier_error": outcome.error,
                    "failed_tests": outcome.failed_tests[:40],
                    "still_resolved": int(r.get("reward") or 0) == 1,
                    "seconds": round(time.time() - started, 1)})
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool, Path(args.out).open("a") as fh:
        futures = {pool.submit(one, t, f): t for t, f in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {"task": futures[fut], "error": "%s: %s" % (type(exc).__name__, exc)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            r = rec.get("reward") or {}
            print("[%d/%d] %-45s original-tests verdict: %s  f2p %s/%s p2p %s/%s  dropped=%s %s" % (
                n, len(jobs), rec["task"],
                "STILL PASS" if rec.get("still_resolved") else "FAILS",
                r.get("f2p_passed", "?"), r.get("f2p_total", "?"), r.get("p2p_passed", "?"), r.get("p2p_total", "?"),
                rec.get("actually_dropped"), rec.get("error") or rec.get("verifier_error") or ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
