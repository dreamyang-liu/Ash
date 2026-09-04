#!/usr/bin/env python3
"""Aggregate a DeepSWE batch (plus optional rerun batches) into one verdict table.

    python3.11 scripts/deepswe_aggregate.py runs/deepswe [runs/deepswe-rerun ...] \
        [--json out.json]

Rules, stated so the number can be audited:

- A task's verdict comes from the LAST batch listed that graded it; earlier
  batches are the fallback. Reruns therefore override the base batch.
- A run is **infra-affected** when its journal holds a tool result that is the
  MCP proxy's ``404`` -- the agent was talking to a sandbox Ash had already
  destroyed (Failure log #7). Such a verdict is not the agent's; it is reported
  separately and, when no clean rerun exists, counted as *unmeasured*, not as
  failed.
- pass@1 is reported twice: over every task (unmeasured = fail, the
  conservative number) and over measured tasks only.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

INFRA_404 = "Client error '404"


def infra_affected(journal: Path) -> int:
    """Count of tool results that were the proxy's 404 (0 = clean run)."""
    n = 0
    try:
        with journal.open(encoding="utf-8") as fh:
            for line in fh:
                if '"tool.finished"' in line and INFRA_404 in line:
                    n += 1
    except OSError:
        return 0
    return n


OUR_PROMPT_PREFIXES = ("You are working in", "You are continuing work", "You are fixing a bug",
                       "<system-reminder>\nHidden-test grading")
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def foreign_queued_messages(journal: Path) -> list:
    """User-turn messages in the run's native transcript that this harness did
    not send: another run's CronCreate prompt, a stray task notification. A run
    that received one was steered by something outside the experiment (seen
    2026-09-04: one task's `* * * * *` cron delivered into 2 other tasks'
    branches). Returns the first 80 chars of each."""
    session = None
    try:
        with journal.open(encoding="utf-8") as fh:
            for line in fh:
                if '"session.ref"' in line:
                    session = json.loads(line).get("native_session_id")
                    break
    except OSError:
        return []
    if not session:
        return []
    out = []
    for transcript in CLAUDE_PROJECTS.glob("*/%s.jsonl" % session):
        with transcript.open(encoding="utf-8") as fh:
            for line in fh:
                if '"queue-operation"' not in line or '"enqueue"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                content = rec.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                if content and not content.startswith(OUR_PROMPT_PREFIXES):
                    out.append(content[:80].replace("\n", " "))
    return out


def load_batch(out_dir: Path) -> dict:
    """task -> record for one batch directory (sharded or flat)."""
    records = {}
    for summary in sorted(glob.glob(str(out_dir / "**" / "summary.json"), recursive=True)):
        data = json.loads(Path(summary).read_text())
        for inst in data.get("instances", []):
            attempt = inst["attempts"][0] if inst.get("attempts") else {}
            journal = Path(attempt.get("journal", "")) if attempt.get("journal") else None
            if journal and not journal.is_absolute():
                journal = Path.cwd() / journal
            records[inst["instance"]] = {
                "task": inst["instance"],
                "resolved": bool(inst["resolved"]),
                "status": attempt.get("status"),
                "patch_lines": attempt.get("patch_lines"),
                "grading_error": attempt.get("grading_error"),
                "infra_404s": infra_affected(journal) if journal else 0,
                "foreign_messages": foreign_queued_messages(journal) if journal else [],
                "batch": str(out_dir),
                "journal": str(journal) if journal else None,
            }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batches", nargs="+", help="base batch first, reruns after")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    final: dict = {}
    for batch in args.batches:
        for task, rec in load_batch(Path(batch)).items():
            final[task] = rec          # later batches override

    rows = sorted(final.values(), key=lambda r: r["task"])
    def tainted(r):
        return r["infra_404s"] or r.get("foreign_messages")
    measured = [r for r in rows if not tainted(r)]
    unmeasured = [r for r in rows if tainted(r)]
    resolved_all = sum(r["resolved"] for r in rows)
    resolved_measured = sum(r["resolved"] for r in measured)
    errors = [r for r in rows if r["status"] != "completed" or r["grading_error"]]

    print("tasks graded: %d   infra-affected (unmeasured): %d   agent/grading errors: %d"
          % (len(rows), len(unmeasured), len(errors)))
    print("pass@1 conservative (unmeasured=fail): %d/%d = %.1f%%"
          % (resolved_all, len(rows), 100.0 * resolved_all / len(rows) if rows else 0))
    print("pass@1 over measured tasks:            %d/%d = %.1f%%"
          % (resolved_measured, len(measured),
             100.0 * resolved_measured / len(measured) if measured else 0))
    if unmeasured:
        print("\ninfra-affected or externally steered (need clean rerun):")
        for r in unmeasured:
            print("  %-50s 404s=%-4d foreign_msgs=%d batch=%s" % (
                r["task"], r["infra_404s"], len(r.get("foreign_messages") or []), r["batch"]))
            for m in (r.get("foreign_messages") or [])[:2]:
                print("        foreign: %s" % m)
    if errors:
        print("\nnon-completed / grading errors:")
        for r in errors:
            print("  %-50s status=%s error=%s" % (r["task"], r["status"], r["grading_error"]))
    print("\nper task:")
    for r in rows:
        flag = "INFRA" if r["infra_404s"] else ("PASS" if r["resolved"] else "fail")
        print("  %-5s %-50s patch=%5s %s" % (flag, r["task"], r["patch_lines"],
                                            "" if r["status"] == "completed" else r["status"]))
    if args.json:
        Path(args.json).write_text(json.dumps({
            "tasks": rows,
            "pass_at_1_conservative": [resolved_all, len(rows)],
            "pass_at_1_measured": [resolved_measured, len(measured)],
        }, indent=2))
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
