#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from ash_sandbox import ExactPrefixIndex, RelaxedChangeIndex, TrajectoryCache


DEFAULT_READ_ROOTS = ("/app", "/dev_tests", "/usr/local/cargo", "/root/.cargo")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def tool_boundaries(messages: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if str(message.get("role") or "") in {"tool_result", "tool"}
    ]


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return ordered[index]


def stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.mean(values), percentile95(values)


def benchmark_one(
    *,
    trajectory_path: Path,
    task_id: str,
    stride: int,
    read_roots: tuple[str, ...],
) -> dict[str, Any]:
    trajectory = load_json(trajectory_path)
    messages = [row for row in (trajectory.get("messages") or []) if isinstance(row, dict)]
    boundaries = tool_boundaries(messages)
    env = f"lookup-bench:{task_id}"

    exact_ms: list[float] = []
    relaxed_ms: list[float] = []
    registration_ms: list[float] = []
    exact_hits = 0
    relaxed_hits = 0
    relaxed_only_hits = 0
    first_checkpoint_seen = False

    with tempfile.TemporaryDirectory(prefix="ash-lookup-table-") as td:
        root = Path(td)
        with (
            ExactPrefixIndex(root / "exact.sqlite3") as exact,
            RelaxedChangeIndex(root / "relaxed.sqlite3") as relaxed,
        ):
            cache = TrajectoryCache(exact, relaxed)
            for tool_step, message_index in enumerate(boundaries, start=1):
                prefix = messages[: message_index + 1]

                if first_checkpoint_seen:
                    started = time.perf_counter()
                    exact_hit = cache.lookup_materialized_state(
                        mode="exact",
                        task_id=task_id,
                        env_fingerprint=env,
                        trajectory_prefix=prefix,
                    )
                    exact_ms.append((time.perf_counter() - started) * 1000.0)

                    started = time.perf_counter()
                    relaxed_hit = cache.lookup_materialized_state(
                        mode="relaxed",
                        task_id=task_id,
                        env_fingerprint=env,
                        trajectory_prefix=prefix,
                        messages=prefix,
                        allow_safe_shell=True,
                        workspace_roots=read_roots,
                    )
                    relaxed_ms.append((time.perf_counter() - started) * 1000.0)

                    is_exact = exact_hit is not None
                    is_relaxed = relaxed_hit is not None
                    exact_hits += int(is_exact)
                    relaxed_hits += int(is_relaxed)
                    relaxed_only_hits += int(is_relaxed and not is_exact)

                is_checkpoint = tool_step % stride == 0 or tool_step == len(boundaries)
                if is_checkpoint:
                    started = time.perf_counter()
                    cache.register(
                        task_id=task_id,
                        env_fingerprint=env,
                        trajectory_prefix=prefix,
                        messages=prefix,
                        reference=f"snapshot:{tool_step}",
                        step_id=tool_step,
                        allow_safe_shell=True,
                        workspace_roots=read_roots,
                    )
                    registration_ms.append((time.perf_counter() - started) * 1000.0)
                    first_checkpoint_seen = True

    exact_mean, exact_p95 = stats(exact_ms)
    relaxed_mean, relaxed_p95 = stats(relaxed_ms)
    reg_mean, reg_p95 = stats(registration_ms)
    return {
        "task": task_id,
        "trajectory": str(trajectory_path),
        "tool_steps": len(boundaries),
        "checkpoint_stride": stride,
        "query_count": len(exact_ms),
        "checkpoint_count": len(registration_ms),
        "exact_current_state_hits": exact_hits,
        "relaxed_current_state_hits": relaxed_hits,
        "relaxed_only_hits": relaxed_only_hits,
        "exact_lookup_mean_ms": round(exact_mean, 6),
        "exact_lookup_p95_ms": round(exact_p95, 6),
        "relaxed_lookup_mean_ms": round(relaxed_mean, 6),
        "relaxed_lookup_p95_ms": round(relaxed_p95, 6),
        "registration_mean_ms": round(reg_mean, 6),
        "registration_p95_ms": round(reg_p95, 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure actual SQLite Exact-vs-Relaxed lookup overhead on recorded trajectories.")
    ap.add_argument("--audit-json", type=Path, default=Path("results/relaxed-search-final/real_trajectory_search.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/relaxed-search-final"))
    ap.add_argument("--checkpoint-stride", type=int, default=10)
    ap.add_argument("--read-root", action="append", dest="read_roots")
    args = ap.parse_args()
    if args.checkpoint_stride < 1:
        raise ValueError("checkpoint stride must be >= 1")
    read_roots = tuple(args.read_roots or DEFAULT_READ_ROOTS)

    audit = load_json(args.audit_json)
    selected = audit.get("trajectories") or []
    rows: list[dict[str, Any]] = []
    for selected_row in selected:
        if not isinstance(selected_row, dict):
            continue
        path = Path(str(selected_row.get("trajectory") or ""))
        task_id = str(selected_row.get("task") or "")
        if not path.exists() or not task_id:
            continue
        rows.append(
            benchmark_one(
                trajectory_path=path,
                task_id=task_id,
                stride=args.checkpoint_stride,
                read_roots=read_roots,
            )
        )

    exact_times = [float(row["exact_lookup_mean_ms"]) for row in rows if row["query_count"]]
    relaxed_times = [float(row["relaxed_lookup_mean_ms"]) for row in rows if row["query_count"]]
    aggregate = {
        "trajectory_count": len(rows),
        "query_count": sum(int(row["query_count"]) for row in rows),
        "checkpoint_count": sum(int(row["checkpoint_count"]) for row in rows),
        "exact_current_state_hits": sum(int(row["exact_current_state_hits"]) for row in rows),
        "relaxed_current_state_hits": sum(int(row["relaxed_current_state_hits"]) for row in rows),
        "relaxed_only_hits": sum(int(row["relaxed_only_hits"]) for row in rows),
        "mean_of_trajectory_exact_lookup_mean_ms": round(statistics.mean(exact_times), 6) if exact_times else 0.0,
        "mean_of_trajectory_relaxed_lookup_mean_ms": round(statistics.mean(relaxed_times), 6) if relaxed_times else 0.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "lookup_latency.csv", rows)
    (args.output_dir / "lookup_latency.json").write_text(
        json.dumps({"aggregate": aggregate, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate, "rows": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
