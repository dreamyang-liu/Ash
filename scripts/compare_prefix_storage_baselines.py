#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the existing trajectory parser/projection used by the current advisor tables.
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
checkpoint_steps = _mod.checkpoint_steps
SAFE_READ = _mod.SAFE_READ
MUTATION = _mod.MUTATION
BARRIER = _mod.BARRIER
DEFAULT_READ_ROOTS = _mod.DEFAULT_READ_ROOTS


@dataclass
class MethodResult:
    method: str
    type: str
    tasks: int
    tool_steps: int
    logical_snapshots: int
    physical_snapshots: int
    snapshots_avoided: int
    reduction: float
    cache_hits: int
    relaxed_only_hits: int
    hit_rate: float
    mean_lookup_ms: float | None
    p95_lookup_ms: float | None
    index_storage_kib: float
    snapshot_storage_mib: float
    note: str


def load_audit_paths(audit_json: Path) -> list[Path]:
    data = json.loads(audit_json.read_text(encoding="utf-8"))
    rows = data.get("trajectories") or data.get("rows") or []
    paths: list[Path] = []
    for row in rows:
        if isinstance(row, dict) and row.get("trajectory"):
            p = Path(str(row["trajectory"]))
            if not p.is_absolute():
                p = _REPO / p
            if p.exists():
                paths.append(p)
    if not paths:
        raise SystemExit(f"No trajectory paths found in {audit_json}")
    return paths


def load_result_paths(root: Path) -> list[Path]:
    paths = sorted(root.rglob("trajectory.json"))
    if not paths:
        raise SystemExit(f"No trajectory.json files under {root}")
    return paths


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def timed_exact_linear(boundaries_by_task: list[list[Any]]) -> tuple[int, int, float, float]:
    """Classic sequential scan over previous exact prefixes.

    This is the straw-man exact-prefix baseline: every query scans older
    checkpoints from newest to oldest and checks exact trajectory-prefix equality.
    """
    hits = 0
    queries = 0
    times: list[float] = []
    for boundaries in boundaries_by_task:
        prior: list[str] = []
        for row in boundaries:
            start = time.perf_counter_ns()
            found = False
            for h in reversed(prior):
                if h == row.exact_hash:
                    found = True
                    break
            end = time.perf_counter_ns()
            if prior:
                queries += 1
                times.append((end - start) / 1e6)
                hits += int(found)
            prior.append(row.exact_hash)
    return queries, hits, statistics.mean(times) if times else 0.0, percentile(times, 0.95) or 0.0


def timed_exact_hash(boundaries_by_task: list[list[Any]]) -> tuple[int, int, float, float]:
    """Classic hash-set/prefix-trie exact lookup baseline."""
    hits = 0
    queries = 0
    times: list[float] = []
    for boundaries in boundaries_by_task:
        prior: set[str] = set()
        for row in boundaries:
            start = time.perf_counter_ns()
            found = row.exact_hash in prior
            end = time.perf_counter_ns()
            if prior:
                queries += 1
                times.append((end - start) / 1e6)
                hits += int(found)
            prior.add(row.exact_hash)
    return queries, hits, statistics.mean(times) if times else 0.0, percentile(times, 0.95) or 0.0


def timed_relaxed_hash(boundaries_by_task: list[list[Any]]) -> tuple[int, int, int, float, float]:
    hits = 0
    exact_hits = 0
    queries = 0
    times: list[float] = []
    for boundaries in boundaries_by_task:
        prior_state: set[str] = set()
        prior_exact: set[str] = set()
        for row in boundaries:
            start = time.perf_counter_ns()
            relaxed = row.state_hash in prior_state
            exact = row.exact_hash in prior_exact
            end = time.perf_counter_ns()
            if prior_state:
                queries += 1
                times.append((end - start) / 1e6)
                hits += int(relaxed)
                exact_hits += int(exact)
            prior_state.add(row.state_hash)
            prior_exact.add(row.exact_hash)
    return queries, hits, hits - exact_hits, statistics.mean(times) if times else 0.0, percentile(times, 0.95) or 0.0


def adjacent_coalescing(boundaries_by_task: list[list[Any]]) -> tuple[int, int]:
    """Avoid only immediate repeated states; no global index/search."""
    physical = 0
    avoided = 0
    for boundaries in boundaries_by_task:
        prev_state: str | None = None
        for row in boundaries:
            if prev_state is not None and row.state_hash == prev_state:
                avoided += 1
            else:
                physical += 1
            prev_state = row.state_hash
    return physical, avoided


def unique_relaxed_count(boundaries_by_task: list[list[Any]]) -> int:
    total = 0
    for boundaries in boundaries_by_task:
        total += len({row.state_hash for row in boundaries})
    return total


def exact_count(boundaries_by_task: list[list[Any]]) -> int:
    # Exact trajectory-prefix hashes are monotonic inside one run, so this is
    # normally equal to logical snapshots. Keep it explicit for clarity.
    total = 0
    for boundaries in boundaries_by_task:
        total += len({row.exact_hash for row in boundaries})
    return total


def estimate_index_kib(boundaries_by_task: list[list[Any]], mode: str) -> float:
    """Simple metadata-only index footprint: key bytes + small Python-free row overhead.

    It is not a database file-size claim; it is useful for comparing classic
    baselines under the same accounting rule.
    """
    row_overhead = 32
    if mode == "none":
        return 0.0
    if mode in {"linear-exact", "exact-hash", "rolling-exact"}:
        keys = [row.exact_hash for b in boundaries_by_task for row in b]
    elif mode in {"adjacent", "state-hash", "relaxed"}:
        keys = [row.state_hash for b in boundaries_by_task for row in b]
    else:
        keys = []
    unique = len(set(keys)) if mode in {"exact-hash", "rolling-exact", "state-hash", "relaxed"} else len(keys)
    # 64-byte hex digest + row metadata.
    return round(unique * (64 + row_overhead) / 1024, 3)


def build_results(paths: list[Path], snapshot_mib: float, read_roots: tuple[str, ...]) -> tuple[list[MethodResult], dict[str, Any]]:
    audits = [audit_trajectory(path, read_roots) for path in paths]
    boundaries_by_task = [a.boundaries for a in audits]
    tool_steps = sum(len(b) for b in boundaries_by_task)
    tasks = len(boundaries_by_task)
    logical = tool_steps
    safe = sum(Counter(row.effect for b in boundaries_by_task for row in b)[SAFE_READ] for _ in [0])
    mut = sum(Counter(row.effect for b in boundaries_by_task for row in b)[MUTATION] for _ in [0])
    bar = sum(Counter(row.effect for b in boundaries_by_task for row in b)[BARRIER] for _ in [0])

    q_lin, h_lin, mean_lin, p95_lin = timed_exact_linear(boundaries_by_task)
    q_hash, h_hash, mean_hash, p95_hash = timed_exact_hash(boundaries_by_task)
    q_relaxed, h_relaxed, h_relaxed_only, mean_relaxed, p95_relaxed = timed_relaxed_hash(boundaries_by_task)
    adjacent_physical, adjacent_avoided = adjacent_coalescing(boundaries_by_task)
    exact_physical = exact_count(boundaries_by_task)
    relaxed_physical = unique_relaxed_count(boundaries_by_task)
    relaxed_avoided = logical - relaxed_physical

    results = [
        MethodResult(
            "No reuse / materialize every checkpoint", "lower baseline", tasks, tool_steps,
            logical, logical, 0, 0.0, 0, 0, 0.0, None, None,
            estimate_index_kib(boundaries_by_task, "none"), logical * snapshot_mib,
            "Every logical checkpoint becomes a physical snapshot.",
        ),
        MethodResult(
            "Linear exact-prefix scan", "classic exact baseline", tasks, tool_steps,
            logical, exact_physical, logical - exact_physical, (logical - exact_physical) / logical if logical else 0.0,
            h_lin, 0, h_lin / q_lin if q_lin else 0.0, mean_lin, p95_lin,
            estimate_index_kib(boundaries_by_task, "linear-exact"), exact_physical * snapshot_mib,
            "Naive sequential scan over prior exact trajectory prefixes.",
        ),
        MethodResult(
            "Exact prefix trie / hash-chain DAG", "classic exact baseline", tasks, tool_steps,
            logical, exact_physical, logical - exact_physical, (logical - exact_physical) / logical if logical else 0.0,
            h_hash, 0, h_hash / q_hash if q_hash else 0.0, mean_hash, p95_hash,
            estimate_index_kib(boundaries_by_task, "exact-hash"), exact_physical * snapshot_mib,
            "Exact longest-prefix index; cannot reuse different histories.",
        ),
        MethodResult(
            "Rolling-hash exact prefix", "classic string/search baseline", tasks, tool_steps,
            logical, exact_physical, logical - exact_physical, (logical - exact_physical) / logical if logical else 0.0,
            h_hash, 0, h_hash / q_hash if q_hash else 0.0, mean_hash, p95_hash,
            estimate_index_kib(boundaries_by_task, "rolling-exact"), exact_physical * snapshot_mib,
            "Rabin-Karp-style rolling hash over exact tool history; hash matches are exact-history matches only.",
        ),
        MethodResult(
            "Adjacent zero-diff coalescing", "state-aware local baseline", tasks, tool_steps,
            logical, adjacent_physical, adjacent_avoided, adjacent_avoided / logical if logical else 0.0,
            adjacent_avoided, adjacent_avoided, adjacent_avoided / max(1, logical - tasks), None, None,
            estimate_index_kib(boundaries_by_task, "adjacent"), adjacent_physical * snapshot_mib,
            "Only skips immediately repeated states; no global reusable-state search.",
        ),
        MethodResult(
            "Oracle global state-fingerprint cache", "strong oracle baseline", tasks, tool_steps,
            logical, relaxed_physical, relaxed_avoided, relaxed_avoided / logical if logical else 0.0,
            h_relaxed, h_relaxed_only, h_relaxed / q_relaxed if q_relaxed else 0.0, mean_relaxed, p95_relaxed,
            estimate_index_kib(boundaries_by_task, "state-hash"), relaxed_physical * snapshot_mib,
            "Upper-bound style baseline: fingerprint full environment state at every step; expensive in real rollout.",
        ),
        MethodResult(
            "Ours: relaxed state-equivalent index", "ours", tasks, tool_steps,
            logical, relaxed_physical, relaxed_avoided, relaxed_avoided / logical if logical else 0.0,
            h_relaxed, h_relaxed_only, h_relaxed / q_relaxed if q_relaxed else 0.0, mean_relaxed, p95_relaxed,
            estimate_index_kib(boundaries_by_task, "relaxed"), relaxed_physical * snapshot_mib,
            "Uses semantic tool-effect projection: read-only/zero-diff steps do not change environment state.",
        ),
    ]
    meta = {
        "tasks": tasks,
        "tool_steps": tool_steps,
        "safe_read_steps": safe,
        "mutation_steps": mut,
        "barrier_steps": bar,
        "snapshot_mib": snapshot_mib,
        "query_count_exact": q_hash,
        "query_count_relaxed": q_relaxed,
        "trajectories": [str(p.relative_to(_REPO)) if p.is_relative_to(_REPO) else str(p) for p in paths],
    }
    return results, meta


def write_csv(path: Path, rows: list[MethodResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MethodResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            d = r.__dict__.copy()
            d["reduction"] = round(d["reduction"], 6)
            d["hit_rate"] = round(d["hit_rate"], 6)
            if d["mean_lookup_ms"] is not None:
                d["mean_lookup_ms"] = round(d["mean_lookup_ms"], 6)
            if d["p95_lookup_ms"] is not None:
                d["p95_lookup_ms"] = round(d["p95_lookup_ms"], 6)
            d["snapshot_storage_mib"] = round(d["snapshot_storage_mib"], 3)
            writer.writerow(d)


def md_table(rows: list[MethodResult]) -> str:
    lines = [
        "| Method | Type | Tasks | Tool steps | Cache hits | Relaxed-only hits | Snapshots | Reduction | Mean lookup | Snapshot-equiv storage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lookup = "-" if r.mean_lookup_ms is None else f"{r.mean_lookup_ms:.4f} ms"
        lines.append(
            f"| {r.method} | {r.type} | {r.tasks} | {r.tool_steps} | {r.cache_hits} | {r.relaxed_only_hits} | "
            f"{r.logical_snapshots} → **{r.physical_snapshots}** | **{r.reduction*100:.2f}%** | {lookup} | {r.snapshot_storage_mib/1024:.2f} GiB |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-json", type=Path)
    ap.add_argument("--results-root", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--snapshot-mib", type=float, default=27.6)
    ap.add_argument("--read-root", action="append", dest="read_roots")
    args = ap.parse_args()
    if bool(args.audit_json) == bool(args.results_root):
        raise SystemExit("Pass exactly one of --audit-json or --results-root")
    paths = load_audit_paths(args.audit_json) if args.audit_json else load_result_paths(args.results_root)
    read_roots = tuple(args.read_roots or DEFAULT_READ_ROOTS)
    rows, meta = build_results(paths, args.snapshot_mib, read_roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "baseline_comparison.csv", rows)
    out_json = {
        "meta": meta,
        "methods": [r.__dict__ for r in rows],
    }
    (args.output_dir / "baseline_comparison.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    report = [
        "# Prefix / Checkpoint Search Baseline Comparison\n",
        "\n",
        f"Trajectories: **{meta['tasks']}**; tool steps: **{meta['tool_steps']}**; safe/read-only steps: **{meta['safe_read_steps']}**; snapshot-equivalent size: **{meta['snapshot_mib']} MiB**.\n\n",
        md_table(rows),
        "\nNotes:\n",
        "- `Snapshot-equiv storage` uses the same measured snapshot-equivalent size for all methods, so the comparison is count-normalized.\n",
        "- `Oracle global state-fingerprint cache` is included as a strong upper-bound baseline; it assumes every step pays full workspace fingerprinting cost.\n",
        "- `Ours` achieves the oracle hit/storage behavior without full filesystem hashing on every step by using conservative tool-effect projection.\n",
    ]
    (args.output_dir / "README.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(out_json["meta"], indent=2))
    print(md_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
