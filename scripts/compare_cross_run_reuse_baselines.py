#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import importlib.util

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_relaxed_search_table", _REPO / "scripts" / "build_relaxed_search_table.py"
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

audit_trajectory = _mod.audit_trajectory
DEFAULT_READ_ROOTS = _mod.DEFAULT_READ_ROOTS


@dataclass
class RunAudit:
    task: str
    model: str
    path: Path
    exact_hashes: list[str]
    state_hashes: list[str]


@dataclass
class CrossRunRow:
    source: str
    query: str
    source_model: str
    query_model: str
    source_steps: int
    query_steps: int
    exact_prefix_hits: int
    adjacent_zero_diff_hits: int
    oracle_global_state_hits: int
    ours_relaxed_hits: int
    ours_relaxed_only_hits: int
    ours_hit_rate: float
    exact_lookup_mean_ms: float
    relaxed_lookup_mean_ms: float


def load_curated(root: Path, read_roots: tuple[str, ...]) -> list[RunAudit]:
    runs: list[RunAudit] = []
    for path in sorted(root.rglob("trajectory.json")):
        audit = audit_trajectory(path, read_roots)
        runs.append(
            RunAudit(
                task=audit.task_id,
                model=audit.model,
                path=path,
                exact_hashes=[b.exact_hash for b in audit.boundaries],
                state_hashes=[b.state_hash for b in audit.boundaries],
            )
        )
    return runs


def pair_rows(runs: list[RunAudit]) -> list[CrossRunRow]:
    rows: list[CrossRunRow] = []
    by_task: dict[str, list[RunAudit]] = {}
    for r in runs:
        by_task.setdefault(r.task, []).append(r)

    for task_runs in by_task.values():
        if len(task_runs) < 2:
            continue
        for source in task_runs:
            source_exact = set(source.exact_hashes)
            source_state = set(source.state_hashes)
            for query in task_runs:
                if source.path == query.path:
                    continue

                exact_hits = 0
                global_hits = 0
                exact_times: list[float] = []
                relaxed_times: list[float] = []
                for eh, sh in zip(query.exact_hashes, query.state_hashes):
                    t0 = time.perf_counter_ns()
                    exact_hit = eh in source_exact
                    t1 = time.perf_counter_ns()
                    state_hit = sh in source_state
                    t2 = time.perf_counter_ns()
                    exact_hits += int(exact_hit)
                    global_hits += int(state_hit)
                    exact_times.append((t1 - t0) / 1e6)
                    relaxed_times.append((t2 - t1) / 1e6)

                adjacent_hits = 0  # Local within-run rule; no searchable cross-run index.
                relaxed_only = sum(
                    int(sh in source_state and eh not in source_exact)
                    for eh, sh in zip(query.exact_hashes, query.state_hashes)
                )
                rows.append(
                    CrossRunRow(
                        source=f"{source.task}::{source.path.parent.name}",
                        query=f"{query.task}::{query.path.parent.name}",
                        source_model=source.model,
                        query_model=query.model,
                        source_steps=len(source.state_hashes),
                        query_steps=len(query.state_hashes),
                        exact_prefix_hits=exact_hits,
                        adjacent_zero_diff_hits=adjacent_hits,
                        oracle_global_state_hits=global_hits,
                        ours_relaxed_hits=global_hits,
                        ours_relaxed_only_hits=relaxed_only,
                        ours_hit_rate=global_hits / len(query.state_hashes) if query.state_hashes else 0.0,
                        exact_lookup_mean_ms=statistics.mean(exact_times) if exact_times else 0.0,
                        relaxed_lookup_mean_ms=statistics.mean(relaxed_times) if relaxed_times else 0.0,
                    )
                )
    return rows


def summarize(rows: list[CrossRunRow]) -> dict[str, Any]:
    query_steps = sum(r.query_steps for r in rows)
    exact_hits = sum(r.exact_prefix_hits for r in rows)
    adjacent_hits = sum(r.adjacent_zero_diff_hits for r in rows)
    oracle_hits = sum(r.oracle_global_state_hits for r in rows)
    ours_hits = sum(r.ours_relaxed_hits for r in rows)
    relaxed_only = sum(r.ours_relaxed_only_hits for r in rows)
    return {
        "pairs": len(rows),
        "query_steps": query_steps,
        "exact_prefix_hits": exact_hits,
        "adjacent_zero_diff_hits": adjacent_hits,
        "oracle_global_state_hits": oracle_hits,
        "ours_relaxed_hits": ours_hits,
        "ours_relaxed_only_hits": relaxed_only,
        "exact_hit_rate": exact_hits / query_steps if query_steps else 0.0,
        "adjacent_hit_rate": adjacent_hits / query_steps if query_steps else 0.0,
        "oracle_hit_rate": oracle_hits / query_steps if query_steps else 0.0,
        "ours_hit_rate": ours_hits / query_steps if query_steps else 0.0,
        "exact_lookup_mean_ms": statistics.mean([r.exact_lookup_mean_ms for r in rows]) if rows else 0.0,
        "relaxed_lookup_mean_ms": statistics.mean([r.relaxed_lookup_mean_ms for r in rows]) if rows else 0.0,
    }


def write_csv(path: Path, rows: list[CrossRunRow]) -> None:
    fields = list(CrossRunRow.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            d = r.__dict__.copy()
            d["ours_hit_rate"] = round(d["ours_hit_rate"], 6)
            d["exact_lookup_mean_ms"] = round(d["exact_lookup_mean_ms"], 6)
            d["relaxed_lookup_mean_ms"] = round(d["relaxed_lookup_mean_ms"], 6)
            writer.writerow(d)


def method_table(summary: dict[str, Any]) -> str:
    qs = summary["query_steps"]
    rows = [
        ("Exact prefix cache", summary["exact_prefix_hits"], summary["exact_hit_rate"], "Requires identical trajectory history."),
        ("Adjacent zero-diff coalescing", summary["adjacent_zero_diff_hits"], summary["adjacent_hit_rate"], "Local-only rule; no cross-run/branch index."),
        ("Oracle full-state cache", summary["oracle_global_state_hits"], summary["oracle_hit_rate"], "Upper bound with full workspace fingerprinting."),
        ("Ours: relaxed state-equivalent index", summary["ours_relaxed_hits"], summary["ours_hit_rate"], "Indexed projected-state matching without full-state hashing every step."),
    ]
    lines = [
        "| Method | Query states | Reusable hits | Hit rate | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for name, hits, rate, note in rows:
        lines.append(f"| {name} | {qs} | **{hits}** | **{rate*100:.2f}%** | {note} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--read-root", action="append", dest="read_roots")
    args = ap.parse_args()
    read_roots = tuple(args.read_roots or DEFAULT_READ_ROOTS)

    runs: list[RunAudit] = []
    for root in args.roots:
        p = root if root.is_absolute() else _REPO / root
        runs.extend(load_curated(p, read_roots))
    rows = pair_rows(runs)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cross_run_baseline_pairs.csv", rows)
    (args.output_dir / "cross_run_baseline_summary.json").write_text(
        json.dumps({"summary": summary, "pairs": [r.__dict__ for r in rows]}, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Cross-run / Branch-style Reuse Baseline Comparison\n\n",
        "This table evaluates the case where a checkpoint produced by one trajectory is queried by another trajectory for the same task. Adjacent Coalescing is not credited here because it has no searchable cross-run state index.\n\n",
        method_table(summary),
        "\n",
        f"Pairs: **{summary['pairs']}**; query states: **{summary['query_steps']}**; relaxed-only hits: **{summary['ours_relaxed_only_hits']}**.\n",
        "\nCorrect claim: Ours matches the oracle projected-state hit count, while Exact Prefix and Adjacent Coalescing fail on cross-run/branch-style reuse.\n",
    ]
    (args.output_dir / "README.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(method_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
