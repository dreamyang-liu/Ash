#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.benchmark_search_storage_baselines import (
    DEFAULT_READ_ROOTS,
    audit_trajectory,
    sequence_keys,
    tool_messages,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def audit_map(audit_json: Path) -> tuple[dict[str, Path], tuple[str, ...]]:
    data = load_json(audit_json)
    mapping = {str(row["task"]): Path(row["trajectory"]) for row in data.get("trajectories", [])}
    roots = tuple(data.get("protocol", {}).get("read_roots") or DEFAULT_READ_ROOTS)
    return mapping, roots


def exact_keys(path: Path, roots: tuple[str, ...]) -> list[str]:
    return [row.exact_hash for row in audit_trajectory(path, roots).boundaries]


def method_keys(path: Path, roots: tuple[str, ...], mode: str) -> list[str]:
    if mode == "exact_history":
        return exact_keys(path, roots)
    return sequence_keys(tool_messages(path), mode=mode, read_roots=roots)


def compare_pair(
    source: Path,
    query: Path,
    *,
    roots: tuple[str, ...],
    method: str,
) -> dict[str, Any]:
    src_keys = method_keys(source, roots, method)
    qry_keys = method_keys(query, roots, method)
    known = set(src_keys)
    hits = sum(key in known for key in qry_keys)
    return {
        "source_steps": len(src_keys),
        "query_steps": len(qry_keys),
        "hits": hits,
        "hit_rate": hits / len(qry_keys) if qry_keys else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-audit", type=Path, default=Path("results/swemarathon-9x2-lightstage/audit-qwen-curated-dense/real_trajectory_search.json"))
    ap.add_argument("--luna-audit", type=Path, default=Path("results/swemarathon-9x2-lightstage/audit-luna-curated-dense/real_trajectory_search.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/cross-model-cache-baselines"))
    args = ap.parse_args()

    qwen, roots_q = audit_map(args.qwen_audit)
    luna, roots_l = audit_map(args.luna_audit)
    roots = roots_q if roots_q == roots_l else tuple(sorted(set(roots_q) | set(roots_l)))
    tasks = sorted(set(qwen) & set(luna))
    methods = [
        ("Exact history", "exact_history"),
        ("TVCACHE LPM", "tvcache_exact"),
        ("TVCACHE Stateful (static)", "tvcache_stateful"),
        ("TVCACHE Stateful (oracle)", "tvcache_stateful_oracle"),
        ("Stateless tool cache", "stateless_args"),
        ("Naive read heuristic", "naive_read_heuristic"),
        ("Ours: proof-based relaxed", "ours"),
    ]

    rows: list[dict[str, Any]] = []
    for task in tasks:
        for direction, source, query in (
            ("Qwen->Luna", qwen[task], luna[task]),
            ("Luna->Qwen", luna[task], qwen[task]),
        ):
            for label, mode in methods:
                stats = compare_pair(source, query, roots=roots, method=mode)
                rows.append({"task": task, "direction": direction, "method": label, **stats})

    aggregates: list[dict[str, Any]] = []
    for direction in ("Qwen->Luna", "Luna->Qwen"):
        for label, _mode in methods:
            selected = [r for r in rows if r["direction"] == direction and r["method"] == label]
            qsteps = sum(int(r["query_steps"]) for r in selected)
            hits = sum(int(r["hits"]) for r in selected)
            aggregates.append({
                "direction": direction,
                "method": label,
                "tasks": len(selected),
                "query_steps": qsteps,
                "hits": hits,
                "hit_rate": hits / qsteps if qsteps else 0.0,
            })

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"tasks": tasks, "aggregate": aggregates, "rows": rows}, indent=2), encoding="utf-8")
    with (out / "aggregate.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(aggregates[0]))
        w.writeheader(); w.writerows(aggregates)
    with (out / "per_task.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    lines = [
        "# Cross-model / cross-policy cache reuse",
        "",
        f"Same **{len(tasks)} SWE-Marathon tasks**, controlled Qwen3.8-27B and Luna Max trajectories.",
        "",
        "| Direction | Method | Query steps | Hits | Hit rate |",
        "|---|---|---:|---:|---:|",
    ]
    for r in aggregates:
        lines.append(f"| {r['direction']} | {r['method']} | {r['query_steps']} | {r['hits']} | {100*r['hit_rate']:.2f}% |")
    lines += [
        "",
        "This is a policy-shift stress test: source and query use different models on the same task/environment. It is not a benchmark solve-score comparison.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": tasks, "aggregate": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
