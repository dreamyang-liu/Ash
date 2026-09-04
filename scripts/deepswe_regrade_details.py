#!/usr/bin/env python3
"""Re-grade DeepSWE attempts from their snapshots and keep the FULL verdict.

The loop's summary.json records resolved/f2p_pass/p2p_pass per attempt but not
their reward.json (f2p/p2p fractions, counts) nor which tests failed -- the
analysts read those from Grade.detail in memory, and with --rounds 0 nobody
did. This walks the finished journals, re-grades each last snapshot through
the same deepswe.grade path, and writes one JSON line per task with the parsed
reward.json, the non-passing test ids and the repo-state note. It is also an
independent re-verification: a verdict that flips between the run and this
pass is a grader-stability finding, reported as such.

    python3.11 scripts/deepswe_regrade_details.py runs/deepswe [runs/deepswe-rerun1 ...] \
        -o runs/deepswe-details.jsonl --workers 8 [--skip-infra]

Later batches override earlier ones for the same task (rerun over base).
Resumable: tasks already present in -o are skipped.
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

from harness.rollback import load_checkpoints
from swebench.fork_eval import backend_for
from deepswe.bench import DeepSWE
from deepswe.grade import grade_snapshot

INFRA_404 = "Client error '404"


def final_journals(batches):
    """task -> (journal, batch, prior verdict) using the LAST batch that has it."""
    out = {}
    for batch in batches:
        for summary in sorted(glob.glob(str(Path(batch) / "**" / "summary.json"), recursive=True)):
            for inst in json.loads(Path(summary).read_text()).get("instances", []):
                a = inst["attempts"][0] if inst.get("attempts") else {}
                if a.get("journal"):
                    out[inst["instance"]] = (Path(a["journal"]), batch, bool(inst["resolved"]))
    return out


def infra_404s(journal: Path) -> int:
    with journal.open(encoding="utf-8") as fh:
        return sum(1 for l in fh if '"tool.finished"' in l and INFRA_404 in l)


def parse_detail(detail: str) -> dict:
    out = {}
    m = re.search(r"^reward\.json: (\{.*\})$", detail, re.M)
    if m:
        try:
            out["reward"] = json.loads(m.group(1))
        except ValueError:
            pass
    m = re.search(r"^repo state: (.*)$", detail, re.M)
    if m:
        out["repo_state"] = m.group(1)
    m = re.search(r"^non-passing tests \((\d+) shown\): (.*)$", detail, re.M)
    if m:
        out["non_passing"] = [t.strip() for t in m.group(2).split(", ") if t.strip()]
    for flag in ("empty model.patch", "did not apply", "collect step failed"):
        if flag in detail:
            out.setdefault("notes", []).append(flag)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batches", nargs="+")
    parser.add_argument("-o", "--out", default="runs/deepswe-details.jsonl")
    parser.add_argument("--tasks-dir", default=str(Path.home() / "projects/LBP/deep-swe/tasks"))
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-infra", action="store_true",
                        help="skip journals with proxy-404 tool results (their verdict is not the agent's)")
    parser.add_argument("--only", default="", help="comma list of task ids")
    args = parser.parse_args()

    bench = DeepSWE(args.tasks_dir)
    catalogue = bench.catalogue(None)
    backend = backend_for(SimpleNamespace(runtime_bin=args.runtime_bin), bench)
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["task"])
            except (ValueError, KeyError):
                pass

    journals = final_journals(args.batches)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    jobs = []
    for task, (journal, batch, prior) in sorted(journals.items()):
        if task in done or (only and task not in only) or task not in catalogue:
            continue
        n404 = infra_404s(journal)
        if args.skip_infra and n404:
            continue
        jobs.append((task, journal, batch, prior, n404))
    print("%d task(s) to re-grade (%d already done)" % (len(jobs), len(done)), flush=True)

    def one(task, journal, batch, prior, n404):
        started = time.time()
        pairs = [c for c in load_checkpoints(journal) if c.snapshot_id]
        rec = {"task": task, "batch": batch, "journal": str(journal), "infra_404s": n404,
               "prior_resolved": prior, "language": catalogue[task].language}
        if not pairs:
            rec.update({"error": "no snapshot"})
            return rec
        g = grade_snapshot(pairs[-1].snapshot_id, catalogue[task], backend)
        rec.update({"snapshot_id": pairs[-1].snapshot_id, "resolved": g.resolved,
                    "f2p_pass": g.f2p_pass, "p2p_pass": g.p2p_pass, "p2p_ran": g.p2p_ran,
                    "grading_error": g.error, "patch_lines": g.patch.count("\n"),
                    "broken": g.broken, **parse_detail(g.detail),
                    "verdict_changed": (g.error is None and g.resolved != prior),
                    "seconds": round(time.time() - started, 1)})
        return rec

    flips = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a") as fh:
        futures = {pool.submit(one, *job): job[0] for job in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            task = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {"task": task, "error": "%s: %s" % (type(exc).__name__, exc)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            r = rec.get("reward") or {}
            flips += bool(rec.get("verdict_changed"))
            print("[%d/%d] %-4s %-48s f2p %s/%s p2p %s/%s %s%s" % (
                n, len(jobs), "PASS" if rec.get("resolved") else "fail", task,
                r.get("f2p_passed", "?"), r.get("f2p_total", "?"),
                r.get("p2p_passed", "?"), r.get("p2p_total", "?"),
                "FLIPPED " if rec.get("verdict_changed") else "",
                rec.get("grading_error") or rec.get("error") or ""), flush=True)
    print("done; verdict flips vs the run: %d -> %s" % (flips, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
