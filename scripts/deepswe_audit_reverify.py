#!/usr/bin/env python3
"""Is a resolved patch still resolved once its edits to PRE-EXISTING test files
are stripped? The decisive check for "fixed the regression by editing the test".

    python3.11 scripts/deepswe_audit_reverify.py runs/deepswe-audit [--only a,b] --workers 8

For every task in <audit>/audit.jsonl with ``persisting_test_file_edits``, take
<audit>/patches/<task>.patch, drop the per-file diffs for those test files
(hunks for hidden test.patch files are neutralised by the grader anyway), run
their verifier on a pristine offline VM with the filtered patch, and compare
reward with the original. Writes <audit>/reverify.jsonl:

  still_resolved=True   the test edits were not load-bearing (cosmetic, or a
                        legitimate update the hidden suite agrees with)
  still_resolved=False  the verdict DEPENDED on the agent's test edits -> flag
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from swebench.fork_eval import backend_for
from deepswe.bench import DeepSWE
from deepswe.grade import verify_patch, grade_from_verifier


def split_patch(text: str):
    """[(path_b, chunk)] per `diff --git` block, in order."""
    out = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)\n", text, re.M):
        out.append((m.group(2), m.start()))
    blocks = []
    for i, (path, start) in enumerate(out):
        end = out[i + 1][1] if i + 1 < len(out) else len(text)
        blocks.append((path, text[start:end]))
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_dir")
    ap.add_argument("--tasks-dir", default=str(Path.home() / "projects/LBP/deep-swe/tasks"))
    ap.add_argument("--runtime-bin", default="runtime/ash-runtime")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    adir = Path(args.audit_dir)
    bench = DeepSWE(args.tasks_dir); cat = bench.catalogue(None)
    backend = backend_for(SimpleNamespace(runtime_bin=args.runtime_bin), bench)
    only = {x for x in args.only.split(",") if x}
    jobs = []
    for l in (adir / "audit.jsonl").read_text().splitlines():
        r = json.loads(l); c = r.get("classification") or {}
        if c.get("persisting_test_file_edits") and (not only or r["task"] in only):
            jobs.append((r["task"], c["persisting_test_file_edits"]))
    print("%d task(s) with persisting test-file edits to re-verify" % len(jobs), flush=True)

    def one(task_id, test_files):
        task = cat[task_id]
        patch = (adir / "patches" / ("%s.patch" % task_id)).read_text(errors="replace")
        kept = [chunk for path, chunk in split_patch(patch) if path not in set(test_files)]
        filtered = "".join(kept)
        outcome = verify_patch(task, filtered, backend)
        g = grade_from_verifier(filtered, outcome)
        rw = outcome.reward or {}
        return {"task": task_id, "stripped_files": test_files, "still_resolved": g.resolved,
                "grading_error": g.error, "reward_without_test_edits": rw,
                "f2p": "%s/%s" % (rw.get("f2p_passed"), rw.get("f2p_total")),
                "p2p": "%s/%s" % (rw.get("p2p_passed"), rw.get("p2p_total")),
                "p2p_failing_now": [t for t in outcome.failed_tests if t.startswith("p2p:")][:15]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, (adir / "reverify.jsonl").open("w") as fh:
        futs = {pool.submit(one, t, f): t for t, f in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            try: rec = f.result()
            except Exception as exc:
                rec = {"task": futs[f], "error": "%s: %s" % (type(exc).__name__, exc)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            print("[%d/%d] %-45s still_resolved=%s f2p=%s p2p=%s %s" % (
                n, len(jobs), rec["task"], rec.get("still_resolved"), rec.get("f2p"), rec.get("p2p"),
                rec.get("grading_error") or rec.get("error") or ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
