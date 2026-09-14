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
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swebench.branching import branch_run_name, review_branches


def fork_positions(plan: dict, attempts: list, round_no: int):
    if plan.get("validation_error"):
        return ("branches" if plan.get("branch_policy") in ("adaptive-per-branch", "per-branch") else "rounds"), []
    review = plan.get("review") or {}
    reports = plan.get("reports") or {}
    if plan.get("branch_policy") in ("adaptive-per-branch", "per-branch") or (
            "base" not in review and any("base" in b for b in review.get("branches") or [])):
        outcomes = {a["name"]: a for a in attempts}
        positions = []
        for index, branch in enumerate(review_branches(review), 1):
            attempt = outcomes.get(branch_run_name(round_no, index, branch.get("name")))
            total = reports.get(branch.get("base"), {}).get("steps")
            step = branch.get("branch_step")
            if attempt is not None and total and step:
                positions.append((int(step) / int(total), bool(attempt["resolved"])))
        return "branches", positions
    base = str(review.get("base") or "parent")
    total, step = reports.get(base, {}).get("steps"), review.get("branch_step")
    return "rounds", ([(int(step) / int(total), any(a["resolved"] for a in attempts))]
                      if step and total else [])

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
    parser.add_argument("branch_dir", nargs="+",
                        help="branching batch dir(s); later ones override earlier for the same task "
                             "(e.g. a rerun of tasks the first batch could not fork)")
    parser.add_argument("--single", default=None)
    parser.add_argument("--details", default=None)
    args = parser.parse_args()

    roots = [Path(d) for d in args.branch_dir]
    root = roots[0]
    tasks = {}
    for r in roots:
        for summary in sorted(r.glob("shard-*/summary.json")):
            for inst in json.loads(Path(summary).read_text()).get("instances", []):
                tasks[inst["instance"]] = (inst, Path(summary).parent / inst["instance"])
    planned = set()
    for shard in root.glob("shard-*.txt"):
        planned |= {t for t in shard.read_text().strip().split(",") if t}
    # fallbacks: branches that ran with the full conversation because the parent
    # was compacted before the fork step (origin.cut_note) -- reported, not hidden
    fallbacks = 0
    for _inst, tdir in tasks.values():
        for j in tdir.glob("r*.jsonl"):
            try:
                head = j.open().readline()
                if any('"cut_note": "%s"' % reason in head for reason in (
                        "compacted-before-fork", "cut-refused-by-cli", "explicit-full-conversation")):
                    fallbacks += 1
            except OSError:
                pass

    stage = Counter()
    per_branch = Counter()      # round -> (graded, resolved)
    branch_ok = Counter()
    branch_cost = []
    branch_wall = []
    branch_steps = []
    fork_pos = {"rounds": [], "branches": []}
    rescued = []
    unrescued = []
    for task, (inst, tdir) in sorted(tasks.items()):
        attempts = inst["attempts"]
        by_round = defaultdict(list)
        for a in attempts:
            name = a["name"]
            rnd = 0 if name == "parent" else int(name[1:].split("b", 1)[0])
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
            stage["parent (recorded single pass)" if winner == "parent" else
                  "round %s" % winner[1:].split("b", 1)[0]] += 1
        # fork positions from plan files
        for plan_path in sorted(tdir.glob("plan-round*.json")):
            plan = json.loads(plan_path.read_text())
            rnd = int(plan_path.stem.removeprefix("plan-round"))
            unit, positions = fork_positions(plan, by_round.get(rnd, []), rnd)
            fork_pos[unit].extend(positions)

    n = len(tasks)
    out = []
    out.append("## DeepSWE branching (`%s`): the %d tasks the single pass failed\n" % (", ".join(map(str, roots)), len(planned) or n))
    out.append("Recipe: recorded single-pass parents (no re-run), verifier-guided branching; "
               "per-run limits and policy are recorded in summary.json. %d/%d tasks finished.\n" %
               (n, len(planned) or n))
    out.append("| | |")
    out.append("|---|---:|")
    out.append("| rescued | **%d/%d = %.1f%%** |" % (len(rescued), n, 100.0 * len(rescued) / n if n else 0))
    for k in ("parent (recorded single pass)", "round 1", "round 2"):
        if stage.get(k):
            out.append("| — by %s | %d |" % (k, stage[k]))
    out.append("| unrescued | %d |" % stage.get("unrescued", 0))
    if fallbacks:
        out.append("| branches recorded with FULL-conversation context | %d |" % fallbacks)
    total_branches = sum(per_branch.values())
    total_ok = sum(branch_ok.values())
    out.append("| per-branch success rate (%d recorded branches) | %.0f%% |" % (total_branches, 100.0 * total_ok / total_branches if total_branches else 0))
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
    for unit, positions in fork_pos.items():
        if not positions:
            continue
        out.append("### Where forks were placed (%s; chosen step / base trajectory length)\n" % unit)
        out.append("| fork position | %s | %s |" %
                   (unit, "round rescued" if unit == "rounds" else "branch resolved"))
        out.append("|---|---:|---:|")
        for lo, hi, label in ((0, 1 / 3, "0–33%"), (1 / 3, 2 / 3, "33–66%"), (2 / 3, 1.01, "66–100%")):
            rows = [ok for pos, ok in positions if lo <= pos < hi]
            if rows:
                out.append("| %s | %d | %.0f%% |" % (label, len(rows), 100.0 * sum(rows) / len(rows)))
        out.append("\nmedian chosen position: %.0f%%" % (100 * statistics.median(p for p, _ in positions)))
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
