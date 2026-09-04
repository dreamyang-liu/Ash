#!/usr/bin/env python3
"""Progress of a sharded DeepSWE batch from its summary.json files.

    python3.11 scripts/deepswe_progress.py runs/deepswe

Counts come from summaries, not from process states: workers are known to
linger after writing their final summary.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/deepswe")
    shards = sorted(glob.glob(str(out / "shard-*.txt")))
    planned = sum(len(Path(s).read_text().split(",")) for s in shards if Path(s).read_text().strip())
    done = resolved = errors = 0
    by_status: Counter = Counter()
    rows = []
    for summary in sorted(out.glob("shard-*/summary.json")):
        data = json.loads(Path(summary).read_text())
        for inst in data.get("instances", []):
            done += 1
            resolved += bool(inst["resolved"])
            for attempt in inst.get("attempts", []):
                by_status[attempt.get("status")] += 1
                errors += bool(attempt.get("grading_error"))
            rows.append((inst["instance"], inst["resolved"],
                         inst["attempts"][0].get("status") if inst["attempts"] else "?",
                         inst["attempts"][0].get("grading_error") if inst["attempts"] else None))
    print("%d / %d tasks graded, %d resolved (%.1f%% of graded), %d grading errors"
          % (done, planned, resolved, 100.0 * resolved / done if done else 0.0, errors))
    print("attempt statuses:", dict(by_status))
    for name, ok, status, err in rows:
        print("  %-4s %-48s %-10s %s" % ("PASS" if ok else "fail", name, status, err or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
