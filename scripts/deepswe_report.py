#!/usr/bin/env python3
"""One markdown report for a DeepSWE evaluation: verdicts, cost, failure shape.

    python3.11 scripts/deepswe_report.py runs/deepswe-final.json runs/deepswe-details.jsonl \
        --tasks-dir ~/projects/LBP/deep-swe/tasks [--extra-cost runs/deepswe] > report.md

Inputs:
  final.json    scripts/deepswe_aggregate.py output: the verdict per task and the
                journal that produced it (reruns already layered over the base).
  details.jsonl scripts/deepswe_regrade_details.py output: their reward.json per
                task (f2p/p2p counts), non-passing ids, repo-state note.
  --extra-cost  batch dirs whose journals were NOT the final verdict (infra-
                affected base runs) -- their spend is reported as overhead so
                the total is honest, but they carry no verdict.

Every number here is recomputable from those files; the report says which.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from deepswe.tasks import load_tasks


def journal_stats(path: str) -> dict:
    first = last = None
    usage = {}
    tool_calls = 0
    infra_404 = 0
    status = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ts = e.get("ts")
            if ts:
                first = first or ts
                last = ts
            t = e.get("type")
            if t == "tool.finished":
                tool_calls += 1
                if "Client error '404" in str(e.get("output", ""))[:60]:
                    infra_404 += 1
            elif t == "run.finished":
                usage = e.get("usage") or {}
                status = e.get("status")
    wall = None
    if first and last:
        a = dt.datetime.fromisoformat(first.replace("Z", "+00:00"))
        b = dt.datetime.fromisoformat(last.replace("Z", "+00:00"))
        wall = (b - a).total_seconds()
    return {"wall_s": wall, "cost_usd": usage.get("cost_usd"), "output_tokens": usage.get("output_tokens"),
            "tool_calls": tool_calls, "infra_404": infra_404, "status": status}


def bucket(rec: dict) -> str:
    """Failure shape from the verifier's own numbers."""
    if rec.get("resolved"):
        return "resolved"
    if rec.get("status") not in (None, "completed"):
        return "agent error (run did not complete)"
    notes = rec.get("notes") or []
    if "did not apply" in notes:
        return "patch did not apply to pristine base"
    if "empty model.patch" in notes or (rec.get("patch_lines") or 0) == 0:
        return "nothing committed (empty patch)"
    r = rec.get("reward") or {}
    f2p_total, f2p_passed = r.get("f2p_total") or 0, r.get("f2p_passed") or 0
    p2p_total, p2p_passed = r.get("p2p_total") or 0, r.get("p2p_passed") or 0
    if f2p_total and f2p_passed == f2p_total and p2p_passed < p2p_total:
        return "target tests all pass, regression(s) broke"
    frac = f2p_passed / f2p_total if f2p_total else 0.0
    if frac >= 0.9:
        return "near miss: >=90% of target tests pass"
    if frac >= 0.5:
        return "partial: 50-90% of target tests pass"
    if frac > 0:
        return "weak: <50% of target tests pass"
    return "no target test passes"


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("final")
    parser.add_argument("details")
    parser.add_argument("--tasks-dir", default=str(Path.home() / "projects/LBP/deep-swe/tasks"))
    parser.add_argument("--extra-cost", nargs="*", default=[])
    args = parser.parse_args()

    final = json.loads(Path(args.final).read_text())
    details = {}
    for line in Path(args.details).read_text().splitlines():
        try:
            rec = json.loads(line)
            details[rec["task"]] = rec
        except (ValueError, KeyError):
            continue
    tasks = {t.task_id: t for t in load_tasks(args.tasks_dir)}

    rows = []
    for rec in final["tasks"]:
        task = rec["task"]
        stats = journal_stats(rec["journal"]) if rec.get("journal") else {}
        d = details.get(task, {})
        rows.append({**rec, **stats, "language": tasks[task].language if task in tasks else "?",
                     "reward": d.get("reward"), "notes": d.get("notes"), "non_passing": d.get("non_passing"),
                     "repo_state": d.get("repo_state"),
                     "verdict_changed": d.get("verdict_changed"),
                     "details_resolved": d.get("resolved")})

    n = len(rows)
    resolved = [r for r in rows if r["resolved"]]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    walls = [r["wall_s"] for r in rows if r.get("wall_s")]
    calls = [r["tool_calls"] for r in rows if r.get("tool_calls") is not None]

    extra_cost = 0.0
    extra_runs = 0
    final_journals = {str(Path(r["journal"]).resolve()) for r in rows if r.get("journal")}
    for batch in args.extra_cost:
        for j in glob.glob(str(Path(batch) / "**" / "parent.jsonl"), recursive=True):
            if str(Path(j).resolve()) in final_journals:
                continue
            s = journal_stats(j)
            if s.get("cost_usd"):
                extra_cost += s["cost_usd"]
                extra_runs += 1

    out = []
    out.append("## DeepSWE (datacurve-ai/deep-swe, 113 tasks) — single attempt, no branching\n")
    out.append("**pass@1 = %d/%d = %s**  (claude-code slot, `us.anthropic.claude-sonnet-4-6`, sandbox offline, "
               "2 CPU / 8 GB, agent cap 10800 s, their verifier verbatim in a pristine offline microVM)\n"
               % (len(resolved), n, pct(len(resolved), n)))
    out.append("Not leaderboard-comparable: the leaderboard runs mini-swe-agent; this is Claude Code over "
               "Ash's two MCP tools.\n")

    out.append("### Cost and time (final-verdict runs only)\n")
    if costs:
        out.append("- agent cost: **$%.0f total**, mean $%.2f / task, median $%.2f, max $%.2f"
                   % (sum(costs), statistics.mean(costs), statistics.median(costs), max(costs)))
    if extra_runs:
        out.append("- plus $%.0f spent on %d runs whose verdict was discarded (infra-affected base runs, rerun cleanly)"
                   % (extra_cost, extra_runs))
    if walls:
        out.append("- wall per task: mean %.0f s, median %.0f s, max %.0f s (cap 10800 s; %d hit the cap)"
                   % (statistics.mean(walls), statistics.median(walls), max(walls),
                      sum(1 for w in walls if w >= 10800 * 0.98)))
    if calls:
        out.append("- tool calls per task: mean %.0f, median %.0f, max %d"
                   % (statistics.mean(calls), statistics.median(calls), max(calls)))
    out.append("")

    out.append("### By language\n")
    out.append("| language | tasks | resolved | pass@1 | mean cost |")
    out.append("|---|---|---|---|---|")
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    for lang, rs in sorted(by_lang.items()):
        c = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        out.append("| %s | %d | %d | %s | $%.2f |" % (lang, len(rs), sum(r["resolved"] for r in rs),
                                                  pct(sum(r["resolved"] for r in rs), len(rs)),
                                                  statistics.mean(c) if c else 0))
    out.append("")

    out.append("### Failure shape (from their reward.json, re-graded from snapshots)\n")
    shapes = Counter(bucket(r) for r in rows)
    order = ["resolved", "near miss: >=90% of target tests pass", "target tests all pass, regression(s) broke",
             "partial: 50-90% of target tests pass", "weak: <50% of target tests pass",
             "no target test passes", "nothing committed (empty patch)",
             "patch did not apply to pristine base", "agent error (run did not complete)"]
    out.append("| shape | tasks |")
    out.append("|---|---|")
    for k in order:
        if shapes.get(k):
            out.append("| %s | %d |" % (k, shapes[k]))
    for k, v in shapes.items():
        if k not in order:
            out.append("| %s | %d |" % (k, v))
    out.append("")
    fr = [r for r in rows if not r["resolved"] and r.get("reward") and (r["reward"].get("f2p_total") or 0)]
    if fr:
        fracs = sorted(r["reward"]["f2p_passed"] / r["reward"]["f2p_total"] for r in fr)
        out.append("Among failures with a verifier score, the target-test pass fraction has median %.2f "
                   "(quartiles %.2f / %.2f); %d of %d failures pass every regression test."
                   % (statistics.median(fracs), fracs[len(fracs) // 4], fracs[(3 * len(fracs)) // 4],
                      sum(1 for r in fr if (r["reward"].get("p2p_passed") or 0) == (r["reward"].get("p2p_total") or 0)),
                      len(fr)))
    out.append("")

    flips = [r for r in rows if r.get("verdict_changed")]
    infra = [r for r in rows if r.get("infra_404")]
    out.append("### Integrity checks\n")
    out.append("- independent re-grade from snapshots: %d verdict flips out of %d re-graded"
               % (len(flips), sum(1 for r in rows if r.get("details_resolved") is not None)))
    out.append("- runs with proxy-404 tool results in the FINAL verdict set: %d (must be 0)" % len(infra))
    out.append("- agent runs that did not complete: %s"
               % (", ".join("%s (%s)" % (r["task"], r["status"]) for r in rows if r.get("status") != "completed") or "none"))
    out.append("")

    out.append("### Per task\n")
    out.append("| task | lang | verdict | f2p | p2p | cost | wall s | shape |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (not r["resolved"], r["task"])):
        rw = r.get("reward") or {}
        out.append("| %s | %s | %s | %s/%s | %s/%s | $%.2f | %.0f | %s |" % (
            r["task"], r["language"], "PASS" if r["resolved"] else "fail",
            rw.get("f2p_passed", "?"), rw.get("f2p_total", "?"), rw.get("p2p_passed", "?"), rw.get("p2p_total", "?"),
            r.get("cost_usd") or 0, r.get("wall_s") or 0, bucket(r)))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
