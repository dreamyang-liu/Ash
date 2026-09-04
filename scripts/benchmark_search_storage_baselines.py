#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ash_sandbox.checkpoints import (
    canonical_prefix,
    extend_trajectory_prefix_chain,
    trajectory_prefix_chain_seed_hash,
)
from ash_sandbox.prefix_index import ExactPrefixIndex
from ash_sandbox.relaxed_prefix import (
    SAFE_READ,
    canonical_tool_event,
    normalize_tool_result_messages,
)

# Reuse the audited, conservative exact/relaxed boundary construction.
from scripts.build_relaxed_search_table import DEFAULT_READ_ROOTS, audit_trajectory


STATIC_STATE_PRESERVING_TOOLS = {"grep_files"}
NAIVE_READ_PROGRAMS = {
    "pwd", "ls", "cat", "head", "tail", "wc", "grep", "rg", "stat", "file",
    "sort", "uniq", "tr", "echo", "sed", "find", "which", "cd",
}


@dataclass(frozen=True)
class MethodResult:
    method: str
    logical: int
    physical: int
    avoided: int
    hit_rate: float
    lookup_mean_us: float
    lookup_p95_us: float
    key_payload_bytes: int
    notes: str


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return xs[idx]


def tool_messages(path: Path) -> list[dict[str, Any]]:
    traj = load_json(path)
    messages = [m for m in (traj.get("messages") or []) if isinstance(m, dict)]
    return list(normalize_tool_result_messages(messages))


def tool_descriptor(message: dict[str, Any]) -> dict[str, Any]:
    args = message.get("tool_args") or {}
    if not isinstance(args, dict):
        args = {"__raw__": canonical_prefix(args)}
    return json.loads(canonical_prefix({
        "tool_name": str(message.get("tool_name") or ""),
        "tool_args": args,
    }))


def is_tvcache_state_preserving(message: dict[str, Any]) -> bool:
    """TVCACHE-style explicit tool annotation baseline.

    TVCACHE supports a will_mutate_state annotation.  For Ash's generic tools we
    use only tool-level annotations that are safe without parsing shell syntax:
    grep_files and text_editor view/read.  Generic shell remains stateful/unknown.
    """
    name = str(message.get("tool_name") or "")
    args = message.get("tool_args") or {}
    if name in STATIC_STATE_PRESERVING_TOOLS:
        return True
    if name == "text_editor" and isinstance(args, dict):
        return str(args.get("command") or "").strip().lower() in {"view", "read"}
    return False


def _first_shell_program(command: str) -> str:
    try:
        tokens = shlex.split(str(command or ""), posix=True)
    except ValueError:
        return ""
    if not tokens:
        return ""
    return PurePosixPath(tokens[0]).name


def is_naive_read_heuristic(message: dict[str, Any]) -> bool:
    """Intentionally unsafe classic heuristic: classify by command name only."""
    name = str(message.get("tool_name") or "")
    args = message.get("tool_args") or {}
    if name == "grep_files":
        return True
    if name == "text_editor" and isinstance(args, dict):
        return str(args.get("command") or "").strip().lower() in {"view", "read"}
    if name == "shell" and isinstance(args, dict):
        return _first_shell_program(str(args.get("command") or "")) in NAIVE_READ_PROGRAMS
    return False


def sequence_keys(
    messages: list[dict[str, Any]],
    *,
    mode: str,
    read_roots: tuple[str, ...],
) -> list[str]:
    parent = trajectory_prefix_chain_seed_hash()
    keys: list[str] = []
    for message in messages:
        if mode == "tvcache_exact":
            parent = extend_trajectory_prefix_chain(parent, tool_descriptor(message))
        elif mode == "tvcache_stateful":
            if not is_tvcache_state_preserving(message):
                parent = extend_trajectory_prefix_chain(parent, tool_descriptor(message))
        elif mode == "tvcache_stateful_oracle":
            # Upper-bound TVCACHE instantiation: assume a developer supplies a
            # perfect per-call state-preserving label.  We use the same proven
            # SAFE_READ ground truth as the automatic classifier, but TVCACHE's
            # key remains the sequence of state-modifying tool descriptors.
            event = canonical_tool_event(message, allow_safe_shell=True, workspace_roots=read_roots)
            if str(event.get("effect")) != SAFE_READ:
                parent = extend_trajectory_prefix_chain(parent, tool_descriptor(message))
        elif mode == "stateless_args":
            # Classic/unsafe tool-result cache: key only the current tool name
            # and arguments, ignoring sandbox state and trajectory history.
            parent = hashlib.sha256(canonical_prefix(tool_descriptor(message)).encode("utf-8")).hexdigest()
        elif mode == "naive_read_heuristic":
            if not is_naive_read_heuristic(message):
                parent = extend_trajectory_prefix_chain(parent, tool_descriptor(message))
        elif mode == "ours":
            event = canonical_tool_event(message, allow_safe_shell=True, workspace_roots=read_roots)
            if str(event.get("effect")) != SAFE_READ:
                parent = extend_trajectory_prefix_chain(parent, event)
        else:
            raise ValueError(mode)
        keys.append(parent)
    return keys


def key_membership_stats(keys_by_trajectory: list[list[str]], repeats: int) -> tuple[int, int, list[float]]:
    logical = 0
    physical = 0
    timings_us: list[float] = []
    # Run the real chronological insert/query once for counts.
    for keys in keys_by_trajectory:
        seen: set[str] = set()
        for key in keys:
            logical += 1
            if key not in seen:
                physical += 1
                seen.add(key)
    # Re-run membership-only timing several times; reset per trajectory because
    # task environments are independent and cannot share snapshots blindly.
    for _ in range(max(1, repeats)):
        for keys in keys_by_trajectory:
            seen: set[str] = set()
            for key in keys:
                t0 = time.perf_counter_ns()
                _ = key in seen
                timings_us.append((time.perf_counter_ns() - t0) / 1000.0)
                seen.add(key)
    return logical, physical, timings_us


def evaluate_semantic_methods(
    trajectories: list[Path],
    *,
    read_roots: tuple[str, ...],
    repeats: int,
) -> tuple[list[MethodResult], dict[str, Any]]:
    audits = [audit_trajectory(path, read_roots) for path in trajectories]
    normalized = [tool_messages(path) for path in trajectories]
    exact_keys = [[row.exact_hash for row in audit.boundaries] for audit in audits]
    methods: list[tuple[str, list[list[str]], str]] = [
        (
            "Exact history hash",
            exact_keys,
            "Full model-facing prefix; conservative Ash exact baseline.",
        ),
        (
            "TVCACHE LPM (tool sequence)",
            [sequence_keys(m, mode="tvcache_exact", read_roots=read_roots) for m in normalized],
            "Recent literature baseline: full tool-call sequence longest-prefix matching.",
        ),
        (
            "TVCACHE Stateful (static annotations)",
            [sequence_keys(m, mode="tvcache_stateful", read_roots=read_roots) for m in normalized],
            "State-preserving tools removed using explicit tool-level annotations; generic shell stays stateful.",
        ),
        (
            "TVCACHE Stateful (oracle per-call annotations)",
            [sequence_keys(m, mode="tvcache_stateful_oracle", read_roots=read_roots) for m in normalized],
            "Oracle upper bound: every generic-shell invocation receives a perfect developer-provided mutation label.",
        ),
        (
            "Stateless tool cache (name+args)",
            [sequence_keys(m, mode="stateless_args", read_roots=read_roots) for m in normalized],
            "Classic unsafe tool-result cache keyed only by current tool descriptor; ignores sandbox state.",
        ),
        (
            "Naive read-command heuristic",
            [sequence_keys(m, mode="naive_read_heuristic", read_roots=read_roots) for m in normalized],
            "Command-name-only heuristic; included to expose safety/coverage trade-off.",
        ),
        (
            "Ours: proof-based relaxed state",
            [sequence_keys(m, mode="ours", read_roots=read_roots) for m in normalized],
            "Conservative shell/read proof + observed-result digest for mutation/unknown barriers.",
        ),
    ]

    logical_total = sum(len(a.boundaries) for a in audits)
    no_cache = MethodResult(
        method="No cache / snapshot every boundary",
        logical=logical_total,
        physical=logical_total,
        avoided=0,
        hit_rate=0.0,
        lookup_mean_us=0.0,
        lookup_p95_us=0.0,
        key_payload_bytes=0,
        notes="Lower-bound baseline: no search or reuse.",
    )
    results = [no_cache]
    detail: dict[str, Any] = {"trajectory_count": len(trajectories), "tool_steps": logical_total}
    for name, keys_by_traj, note in methods:
        logical, physical, times = key_membership_stats(keys_by_traj, repeats)
        avoided = logical - physical
        payload_bytes = sum(len(k.encode("ascii")) for keys in keys_by_traj for k in set(keys))
        results.append(MethodResult(
            method=name,
            logical=logical,
            physical=physical,
            avoided=avoided,
            hit_rate=(avoided / logical) if logical else 0.0,
            lookup_mean_us=statistics.fmean(times) if times else 0.0,
            lookup_p95_us=percentile(times, 0.95),
            key_payload_bytes=payload_bytes,
            notes=note,
        ))
    return results, detail


def evaluate_safety() -> list[dict[str, Any]]:
    cases = [
        ("ls -la /app", True),
        ("grep -n TODO /app/a.py", True),
        ("cat /app/a.py | head -20", True),
        ("cd /app && rg TODO .", True),
        ("find /app -name '*.py'", True),
        ("sed -n '1,20p' /app/a.py", True),
        ("echo x > /app/x", False),
        ("sed -i 's/a/b/' /app/a.py", False),
        ("find /app -delete", False),
        ("tail -f /app/a.py", False),
        ("rg --pre 'rm -f /app/a.py' TODO /app", False),
        ("python -c 'open(\"/app/x\",\"w\").write(\"x\")'", False),
        ("make test", False),
    ]
    rows: list[dict[str, Any]] = []
    for command, safe in cases:
        message = {
            "role": "tool_result",
            "tool_name": "shell",
            "tool_args": {"command": command},
            "content": "",
            "success": True,
        }
        tv_preserving = is_tvcache_state_preserving(message)
        naive = is_naive_read_heuristic(message)
        ours_event = canonical_tool_event(message, allow_safe_shell=True, workspace_roots=("/app",))
        ours = ours_event.get("effect") == SAFE_READ
        rows.append({
            "command": command,
            "ground_truth_safe": safe,
            "tvcache_static_preserving": tv_preserving,
            "naive_read_heuristic": naive,
            "ours_proven_safe": ours,
        })
    return rows


class TrieNode:
    __slots__ = ("children", "target")
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.target = False


def classic_index_microbench(depth: int, query_repeats: int) -> list[dict[str, Any]]:
    if depth < 10:
        raise ValueError("depth must be >= 10")
    items = [f"step-{i:06d}" for i in range(depth + 1)]

    # Prefix hashes are the compact content-addressed representation used by the
    # hash-map and DAG variants.
    parent = trajectory_prefix_chain_seed_hash()
    hashes: list[str] = []
    for item in items[:-1]:
        parent = extend_trajectory_prefix_chain(parent, item)
        hashes.append(parent)
    query_parent = trajectory_prefix_chain_seed_hash()
    query_hashes: list[str] = []
    for item in items:
        query_parent = extend_trajectory_prefix_chain(query_parent, item)
        query_hashes.append(query_parent)

    rows: list[dict[str, Any]] = []

    # 1) Classic linear scan over stored full prefixes.
    flat_prefixes = [tuple(items[:i]) for i in range(1, depth + 1)]
    flat_serialized_bytes = sum(len(json.dumps(p, separators=(",", ":")).encode()) for p in flat_prefixes)
    times = []
    for _ in range(query_repeats):
        t0 = time.perf_counter_ns()
        matched = 0
        q = tuple(items)
        for i in range(min(depth, len(q)), 0, -1):
            if q[:i] == flat_prefixes[i - 1]:
                matched = i
                break
        times.append((time.perf_counter_ns() - t0) / 1e6)
    rows.append({
        "index": "Linear scan / full-prefix list",
        "depth": depth,
        "matched_depth": matched,
        "lookup_mean_ms": statistics.fmean(times),
        "lookup_p95_ms": percentile(times, 0.95),
        "serialized_key_bytes": flat_serialized_bytes,
        "complexity": "O(depth) comparisons; O(depth^2) prefix payload",
    })

    # 2) Classic hash table over content-addressed prefix hashes.
    hash_targets = set(hashes)
    times = []
    for _ in range(query_repeats):
        t0 = time.perf_counter_ns()
        matched = 0
        for i in range(min(depth, len(query_hashes)), 0, -1):
            if query_hashes[i - 1] in hash_targets:
                matched = i
                break
        times.append((time.perf_counter_ns() - t0) / 1e6)
    rows.append({
        "index": "HashMap / SHA-256 prefixes",
        "depth": depth,
        "matched_depth": matched,
        "lookup_mean_ms": statistics.fmean(times),
        "lookup_p95_ms": percentile(times, 0.95),
        "serialized_key_bytes": len(hash_targets) * 32,
        "complexity": "O(depth) worst-case probes, O(1) hash membership",
    })

    # 3) Classic trie / TVCACHE TCG shape.
    root = TrieNode()
    node = root
    for item in items[:-1]:
        node = node.children.setdefault(item, TrieNode())
        node.target = True
    times = []
    for _ in range(query_repeats):
        t0 = time.perf_counter_ns()
        node = root
        matched = 0
        for i, item in enumerate(items, start=1):
            nxt = node.children.get(item)
            if nxt is None:
                break
            node = nxt
            if node.target:
                matched = i
        times.append((time.perf_counter_ns() - t0) / 1e6)
    trie_payload = sum(len(item.encode()) for item in items[:-1])
    rows.append({
        "index": "Trie / Tool-Call Graph",
        "depth": depth,
        "matched_depth": matched,
        "lookup_mean_ms": statistics.fmean(times),
        "lookup_p95_ms": percentile(times, 0.95),
        "serialized_key_bytes": trie_payload,
        "complexity": "O(depth) traversal; shared-prefix storage",
    })

    # 4) Actual SQLite hash-chain DAG implementation.
    with tempfile.TemporaryDirectory(prefix="ash-index-baseline-") as td:
        db_path = Path(td) / "prefix.sqlite3"
        with ExactPrefixIndex(db_path) as index:
            cursor = index.root()
            for i, item in enumerate(items[:-1], start=1):
                cursor = index.append(cursor, item)
                index.register(task_id="bench", cursor=cursor, reference=f"snap-{i}", step_id=i)
            times = []
            for _ in range(query_repeats):
                t0 = time.perf_counter_ns()
                hit = index.longest_match(task_id="bench", trajectory_prefix=items)
                times.append((time.perf_counter_ns() - t0) / 1e6)
            matched = 0 if hit is None else hit.cursor.depth
        sqlite_bytes = db_path.stat().st_size
        wal = db_path.with_name(db_path.name + "-wal")
        shm = db_path.with_name(db_path.name + "-shm")
        if wal.exists():
            sqlite_bytes += wal.stat().st_size
        if shm.exists():
            sqlite_bytes += shm.stat().st_size
    rows.append({
        "index": "Ours exact index / SQLite hash-chain DAG",
        "depth": depth,
        "matched_depth": matched,
        "lookup_mean_ms": statistics.fmean(times),
        "lookup_p95_ms": percentile(times, 0.95),
        "serialized_key_bytes": sqlite_bytes,
        "complexity": "shared content-addressed DAG + persistent SQLite metadata",
    })
    return rows


def parse_size_bytes(value: str) -> int:
    value = value.strip()
    units = {"B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2, "GB": 1000**3}
    for unit in ("GB", "MB", "kB", "KB", "B"):
        if value.endswith(unit):
            return int(float(value[:-len(unit)].strip()) * units[unit])
    return 0


def docker_snapshot_storage(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {"available": False, "reason": f"missing {status_path}"}
    status = load_json(status_path)
    rows = []
    for model in ("qwen", "luna"):
        for task, rec in (status.get(model) or {}).items():
            summary_path = Path(str(rec.get("summary") or ""))
            if not summary_path.is_absolute():
                summary_path = Path.cwd() / summary_path
            if not summary_path.exists():
                continue
            summary = load_json(summary_path)
            image = str(summary.get("final_snapshot_id") or "")
            if not image:
                continue
            proc = subprocess.run(
                ["docker", "history", "--no-trunc", "--format", "{{.Size}}", image],
                text=True, capture_output=True, check=False,
            )
            if proc.returncode != 0:
                continue
            sizes = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            top = parse_size_bytes(sizes[0]) if sizes else 0
            rows.append({"model": model, "task": task, "snapshot": image, "top_layer_bytes": top})
    values = [r["top_layer_bytes"] for r in rows]
    return {
        "available": bool(rows),
        "count": len(rows),
        "rows": rows,
        "top_layer_bytes_total": sum(values),
        "top_layer_bytes_mean": statistics.fmean(values) if values else 0.0,
        "top_layer_bytes_median": statistics.median(values) if values else 0.0,
        "note": "docker history top layer measures the writable diff added by docker commit; shared base layers are excluded from this per-snapshot delta statistic.",
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    semantic: list[MethodResult],
    safety: list[dict[str, Any]],
    indices: list[dict[str, Any]],
    storage: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# Prefix / Checkpoint Search Baseline Benchmark",
        "",
        f"Workload: **{metadata['trajectory_count']} real SWE-Marathon trajectories**, {metadata['tool_steps']} tool boundaries.",
        "",
        "## Main comparison",
        "",
        "| Method | Logical | Physical snapshots | Avoided | Hit / dedup rate | Lookup mean | Key payload |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in semantic:
        lines.append(
            f"| {r.method} | {r.logical} | {r.physical} | {r.avoided} | {100*r.hit_rate:.2f}% | "
            f"{r.lookup_mean_us:.3f} us | {r.key_payload_bytes/1024:.2f} KiB |"
        )
    lines += [
        "",
        "**Interpretation.** Exact/history and full tool-sequence matching cannot collapse read-only suffixes inside a single long rollout. "
        "TVCACHE Stateful can do so when tools are explicitly annotated state-preserving. The Ash relaxed method additionally proves a conservative subset of shell commands read-only, while unknown/mutating events retain observed-result digests.",
        "",
        "## Safety / coverage check",
        "",
        "| Shell example | Ground truth safe | TVCACHE static | Naive command-name | Ours proof |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in safety:
        command_display = str(row["command"]).replace("|", "\\|")
        lines.append(
            f"| `{command_display}` | {row['ground_truth_safe']} | "
            f"{row['tvcache_static_preserving']} | {row['naive_read_heuristic']} | {row['ours_proven_safe']} |"
        )
    lines += [
        "",
        "## Classic prefix-index microbenchmark",
        "",
        "| Index | Depth | Lookup mean | p95 | Serialized / on-disk key payload |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in indices:
        lines.append(
            f"| {row['index']} | {row['depth']} | {row['lookup_mean_ms']:.4f} ms | "
            f"{row['lookup_p95_ms']:.4f} ms | {row['serialized_key_bytes']/1024/1024:.3f} MiB |"
        )
    lines += ["", "## Measured Docker snapshot delta storage", ""]
    if storage.get("available"):
        lines += [
            f"Measured retained snapshots: **{storage['count']}**.",
            f"Mean writable top-layer delta: **{storage['top_layer_bytes_mean']/1024/1024:.3f} MiB**; "
            f"median: **{storage['top_layer_bytes_median']/1024/1024:.3f} MiB**; "
            f"sum: **{storage['top_layer_bytes_total']/1024/1024:.3f} MiB**.",
            "",
            storage.get("note", ""),
        ]
    else:
        lines.append(f"Unavailable: {storage.get('reason', 'no retained snapshots found')}")
    lines += [
        "",
        "## Baseline definitions",
        "",
        "- **No cache**: snapshot every logical boundary.",
        "- **Exact history hash**: full model-facing trajectory prefix; safest but strictest.",
        "- **TVCACHE LPM**: tree / TCG over the full sequence of tool calls.",
        "- **TVCACHE Stateful**: removes calls explicitly annotated state-preserving; generic shell is not assumed safe.",
        "- **Naive read heuristic**: skips shell commands solely from their executable name; fast but unsafe on redirection / mutating flags.",
        "- **Ours**: conservative command proof over shell composition and trusted workspace roots; only proven reads are projected out.",
        "",
        "The naive heuristic is an ablation, not a correctness-preserving production baseline.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-json", type=Path, default=Path("results/relaxed-search-dense-final/real_trajectory_search.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/prefix-baseline-comparison"))
    ap.add_argument("--lookup-repeats", type=int, default=100)
    ap.add_argument("--index-depth", type=int, default=3000)
    ap.add_argument("--index-query-repeats", type=int, default=50)
    ap.add_argument("--lightstage-status", type=Path, default=Path("results/swemarathon-9x2-lightstage/status.json"))
    args = ap.parse_args()

    audit = load_json(args.audit_json)
    trajectories = [Path(row["trajectory"]) for row in audit.get("trajectories", [])]
    if not trajectories:
        raise SystemExit("no trajectories in audit json")
    missing = [str(p) for p in trajectories if not p.exists()]
    if missing:
        raise SystemExit(f"missing trajectories: {missing}")
    read_roots = tuple(audit.get("protocol", {}).get("read_roots") or DEFAULT_READ_ROOTS)

    semantic, metadata = evaluate_semantic_methods(trajectories, read_roots=read_roots, repeats=args.lookup_repeats)
    safety = evaluate_safety()
    indices = classic_index_microbench(args.index_depth, args.index_query_repeats)
    storage = docker_snapshot_storage(args.lightstage_status)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    semantic_rows = [r.__dict__ for r in semantic]
    result = {
        "protocol": {
            "audit_json": str(args.audit_json),
            "trajectory_count": metadata["trajectory_count"],
            "tool_steps": metadata["tool_steps"],
            "read_roots": list(read_roots),
            "lookup_repeats": args.lookup_repeats,
            "index_depth": args.index_depth,
            "index_query_repeats": args.index_query_repeats,
            "semantic_note": "Within-task dense checkpoint dedup; no cross-task snapshot reuse.",
        },
        "semantic_methods": semantic_rows,
        "safety_cases": safety,
        "classic_index_microbenchmark": indices,
        "docker_snapshot_storage": storage,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "semantic_methods.csv", semantic_rows)
    write_csv(out / "safety_cases.csv", safety)
    write_csv(out / "classic_index_microbenchmark.csv", indices)
    if storage.get("rows"):
        write_csv(out / "snapshot_storage.csv", storage["rows"])
    (out / "REPORT.md").write_text(render_report(semantic, safety, indices, storage, metadata), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
