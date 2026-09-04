#!/usr/bin/env python3
"""Branching round report for DeepSWE, in the shape of the SWE-bench branch134 table.

    python3.11 scripts/deepswe_branch_report.py runs/deepswe-branch \
        --single runs/deepswe-final.json [--details runs/deepswe-details.jsonl]

Reads every shard's summary.json (attempt names: parent, r1b<k>-*, r2b<k>-*),
the per-round plan files (fork step, base) and the branch journals (cost,
wall). ``--single`` is the single-pass aggregate: its resolved count plus the
rescues here is the combined 113 number. ``--details`` adds the single-pass
failure shape per task, so rescue rate can be split by shape.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def journal_cost_wall(path: str):
    first = last = None
    cost = None
    steps = 0
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                e = json.loads(line)
                ts = e.get("ts")
                if ts:
                    first = first or ts
                    last = ts
                t = e.get("type")
                if t == "tool.finished":
                    steps += 1
                elif t == "run.finished":
                    cost = (e.get("usage") or {}).get("cost_usd")
    except OSError:
        return None, None, 0
    wall = None
    if first and last:
        wall = (dt.datetime.fromisoformat(last.replace("Z", "+00:00"))
                - dt.datetime.fromisoformat(first.replace("Z", "+00:00"))).total_seconds()
    return cost, wall, steps


def shape_of(rec: dict) -> str:
    r = rec.get("reward") or {}
    if not r.get("f2p_total"):
        return "no verifier score / error"
    if r["f2p_passed"] == r["f2p_total"] and (r.get("p2p_passed") or 0) < (r.get("p2p_total") or 0):
        return "target all pass, regression broke"
    frac = r["f2p_passed"] / r["f2p_total"]
    if frac >= 0.9:
        return "near miss (>=90% f2p)"
    if frac >= 0.5:
        return "partial (50-90%)"
    return "weak (<50%)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("branch_dir")
    parser.add_argument("--single", default=None)
    parser.add_argument("--details", default=None)
    args = parser.parse_args()

    root = Path(args.branch_dir)
    tasks = {}
    for summary in sorted(root.glob("shard-*/summary.json")):
        for inst in json.loads(Path(summary).read_text()).get("instances", []):
            tasks[inst["instance"]] = (inst, Path(summary).parent / inst["instance"])
    planned = set()
    for shard in root.glob("shard-*.txt"):
        planned |= {t for t in shard.read_text().strip().split(",") if t}

    stage = Counter()
    per_branch = Counter()      # round -> (graded, resolved)
    branch_ok = Counter()
    branch_cost = []
    branch_wall = []
    branch_steps = []
    fork_pos = []               # (fraction of base trajectory, success)
    rescued = []
    unrescued = []
    for task, (inst, tdir) in sorted(tasks.items()):
        attempts = inst["attempts"]
        by_round = defaultdict(list)
        for a in attempts:
            name = a["name"]
            rnd = 0 if name == "parent" else int(name[1])
            by_round[rnd].append(a)
            if rnd:
                per_branch[rnd] += 1
                branch_ok[rnd] += bool(a["resolved"])
                cost, wall, steps = journal_cost_wall(a["journal"])
                if cost is not None:
                    branch_cost.append(cost)
                if wall:
                    branch_wall.append(wall)
                branch_steps.append(steps)
        winner = next((a["name"] for a in attempts if a["resolved"]), None)
        if winner is None:
            unrescued.append(task)
            stage["unrescued"] += 1
        else:
            rescued.append(task)
            stage["parent (recorded single pass)" if winner == "parent" else "round %s" % winner[1]] += 1
        # fork positions from plan files
        for plan_path in sorted(tdir.glob("plan-round*.json")):
            plan = json.loads(plan_path.read_text())
            review = plan.get("review") or {}
            base = str(review.get("base") or "parent")
            step = review.get("branch_step")
            steps_total = (plan.get("reports") or {}).get(base, {}).get("steps")
            rnd = int(plan_path.stem[-1])
            if step and steps_total:
                ok = any(a["resolved"] for a in by_round.get(rnd, []))
                fork_pos.append((int(step) / int(steps_total), ok))

    n = len(tasks)
    out = []
    out.append("## DeepSWE branching (`%s`): the %d tasks the single pass failed\n" % (root, len(planned) or n))
    out.append("Recipe: recorded single-pass parent as base (no re-run), verifier-guided branching, "
               "2 rounds, width 4 then 3; analyst = agent model. %d/%d tasks finished.\n" % (n, len(planned) or n))
    out.append("| | |")
    out.append("|---|---:|")
    out.append("| rescued | **%d/%d = %.1f%%** |" % (len(rescued), n, 100.0 * len(rescued) / n if n else 0))
    for k in ("parent (recorded single pass)", "round 1", "round 2"):
        if stage.get(k):
            out.append("| — by %s | %d |" % (k, stage[k]))
    out.append("| unrescued | %d |" % stage.get("unrescued", 0))
    total_branches = sum(per_branch.values())
    total_ok = sum(branch_ok.values())
    out.append("| per-branch success rate (%d graded branches) | %.0f%% |" % (total_branches, 100.0 * total_ok / total_branches if total_branches else 0))
    for rnd in sorted(per_branch):
        out.append("| — round %d branches | %d/%d = %.0f%% |" % (rnd, branch_ok[rnd], per_branch[rnd], 100.0 * branch_ok[rnd] / per_branch[rnd]))
    out.append("")
    if branch_cost:
        out.append("Branch cost: **$%.0f total**, mean $%.2f / branch (median $%.2f); mean wall %.0f s, mean %.0f tool calls."
                   % (sum(branch_cost), statistics.mean(branch_cost), statistics.median(branch_cost),
                      statistics.mean(branch_wall) if branch_wall else 0,
                      statistics.mean(branch_steps) if branch_steps else 0))
    if args.single:
        single = json.loads(Path(args.single).read_text())
        s_res, s_n = single["pass_at_1_conservative"]
        out.append("\n**Combined: single pass %d + rescued %d = %d/%d = %.1f%%** (single pass alone %d/%d = %.1f%%)."
                   % (s_res, len(rescued), s_res + len(rescued), s_n, 100.0 * (s_res + len(rescued)) / s_n,
                      s_res, s_n, 100.0 * s_res / s_n))
        s_cost = sum((t.get("cost_usd") or 0) for t in single["tasks"]) if single["tasks"] and "cost_usd" in single["tasks"][0] else None
    out.append("")
    if fork_pos:
        out.append("### Where forks were placed (reviewer's chosen step / base trajectory length)\n")
        out.append("| fork position | rounds | round rescued |")
        out.append("|---|---:|---:|")
        for lo, hi, label in ((0, 1 / 3, "0–33%"), (1 / 3, 2 / 3, "33–66%"), (2 / 3, 1.01, "66–100%")):
            rows = [ok for pos, ok in fork_pos if lo <= pos < hi]
            if rows:
                out.append("| %s | %d | %.0f%% |" % (label, len(rows), 100.0 * sum(rows) / len(rows)))
        out.append("\nmedian chosen position: %.0f%%" % (100 * statistics.median(p for p, _ in fork_pos)))
        out.append("")
    if args.details:
        shapes = {}
        for line in Path(args.details).read_text().splitlines():
            try:
                rec = json.loads(line)
                shapes[rec["task"]] = shape_of(rec)
            except (ValueError, KeyError):
                continue
        out.append("### Rescue rate by single-pass failure shape\n")
        out.append("| prior failure shape | n | rescued |")
        out.append("|---|---:|---:|")
        by_shape = defaultdict(list)
        for task in tasks:
            by_shape[shapes.get(task, "unknown")].append(task in rescued)
        for shape, oks in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
            out.append("| %s | %d | %.0f%% |" % (shape, len(oks), 100.0 * sum(oks) / len(oks)))
        out.append("")
    out.append("### Unrescued\n")
    for t in unrescued:
        out.append("- %s" % t)
    out.append("")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
