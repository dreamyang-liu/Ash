#!/usr/bin/env python3
"""Spot-check the two failure categories a grader is most likely to be lying about.

For each picked instance: restore its last snapshot, and put side by side the
facts a verdict rests on --

- collision class ("test_patch did not apply"): WHICH files the agent touched vs
  which files the graded test_patch modifies. Overlap = the agent really edited
  graded tests; no overlap = the grader's patch application is too fragile and
  the verdict is suspect.
- regression class (target passes, regressions fail): re-run the reported broken
  tests alone. A named test that fails alone is a real regression; a batch-only
  failure or an empty broken list means the verdict came from an exit code, not
  from a test.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

from harness.execution.session import SandboxSession
from harness.rollback import load_checkpoints
from swebench.dataset import (build_batch_test_command, load_instances,
                              malformed_test_ids, parse_test_list)

A = Path(__file__).resolve().parents[1]
BACKEND = {"backend": "microvm",
           "microvm": {"from_image": True,
                       "runtime_bin": str(A / "runtime" / "ash-runtime")}}


def patched_files(patch_text: str) -> set:
    return set(re.findall(r"^\+\+\+ b/(\S+)", patch_text, re.M))


def sh(session, command, timeout=600):
    result = session.execute("shell", {"command": command, "timeout": timeout})
    try:
        body = json.loads(result.output)
        return body.get("exit_code"), (body.get("stdout") or "") + (body.get("stderr") or "")
    except Exception:
        return None, result.output or result.error or ""


def main():
    picks = json.load(open(A / "runs/v500/spotcheck-picks.json"))
    catalogue = {i["instance_id"]: i for i in load_instances("verified")}
    report = []

    for category, entries in picks.items():
        for instance_id, shard in entries:
            raw = catalogue[instance_id]
            journal = glob.glob(str(A / "runs/v500" / shard / instance_id / "parent.jsonl"))[0]
            pairs = [c for c in load_checkpoints(journal) if c.snapshot_id]
            finding = {"instance": instance_id, "category": category}
            if not pairs:
                finding["verdict"] = "no snapshot"
                report.append(finding)
                continue
            session = SandboxSession(quiet=True, backend=BACKEND)
            try:
                if not session.create(pairs[-1].snapshot_id):
                    finding["verdict"] = "restore failed: %s" % session.create_error
                    report.append(finding)
                    continue
                _, diff = sh(session, "cd /testbed && git add -A >/dev/null 2>&1; git diff --cached")
                agent_files = patched_files(diff)
                finding["agent_files"] = sorted(agent_files)

                if category == "collision":
                    graded_files = patched_files(raw.get("test_patch") or "")
                    overlap = agent_files & graded_files
                    finding["graded_test_files"] = sorted(graded_files)
                    finding["overlap"] = sorted(overlap)
                    finding["verdict"] = ("REAL: agent edited graded tests"
                                          if overlap else
                                          "SUSPECT: no overlap -- grader fragility")
                else:
                    # Re-derive which regressions broke, then run them ALONE.
                    p2p = [t for t in parse_test_list(raw["PASS_TO_PASS"])
                           if t not in set(malformed_test_ids(parse_test_list(raw["PASS_TO_PASS"])))]
                    code, out = sh(session,
                                   "cd /testbed && " + build_batch_test_command(raw["repo"], p2p),
                                   timeout=900)
                    failing = sorted(set(re.findall(r"^(?:FAILED|ERROR)[: ]+(\S+)", out, re.M)))[:5]
                    finding["batch_exit"] = code
                    finding["named_failures"] = failing
                    if code == 0:
                        finding["verdict"] = "SUSPECT: regressions PASS on re-run"
                    elif failing:
                        finding["verdict"] = "REAL: named tests fail (%d shown)" % len(failing)
                    else:
                        finding["verdict"] = "SUSPECT: non-zero exit but NO named failure"
            finally:
                session.destroy()
            report.append(finding)
            print(json.dumps(finding, ensure_ascii=False), flush=True)

    out = A / "runs/v500/spotcheck-report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("\nwrote", out)
    for f in report:
        print("%-12s %-32s %s" % (f["category"], f["instance"], f.get("verdict")))


if __name__ == "__main__":
    sys.exit(main())
