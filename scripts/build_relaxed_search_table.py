#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ash_sandbox import CheckpointStore
from ash_sandbox.checkpoints import extend_trajectory_prefix_chain, trajectory_prefix_chain_seed_hash
from ash_sandbox.relaxed_prefix import (
    BARRIER,
    MUTATION,
    SAFE_READ,
    canonical_tool_event,
    normalize_tool_result_messages,
)


DEFAULT_READ_ROOTS = ("/app", "/dev_tests", "/usr/local/cargo", "/root/.cargo")


@dataclass
class ToolBoundary:
    step: int
    exact_hash: str
    state_hash: str
    effect: str
    tool_name: str


@dataclass
class TrajectoryAudit:
    path: Path
    run_dir: Path
    task_id: str
    model: str
    env_key: str
    summary: dict[str, Any]
    boundaries: list[ToolBoundary]
    api_calls: int
    input_tokens: int
    output_tokens: int
    rollout_ms: float
    partial_score: float | None

    @property
    def tool_steps(self) -> int:
        return len(self.boundaries)

    @property
    def effects(self) -> Counter[str]:
        return Counter(row.effect for row in self.boundaries)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object in {path}")
    return value


def _tool_boundary_exact_hashes(messages: list[dict[str, Any]]) -> list[str]:
    parent = ""
    hashes: list[str] = []
    for message in messages:
        parent = extend_trajectory_prefix_chain(parent, message)
        if str(message.get("role") or "") in {"tool_result", "tool"}:
            hashes.append(parent)
    return hashes


def _project_boundaries(messages: list[dict[str, Any]], read_roots: tuple[str, ...]) -> list[ToolBoundary]:
    exact_hashes = _tool_boundary_exact_hashes(messages)
    normalized = list(normalize_tool_result_messages(messages))
    if len(exact_hashes) != len(normalized):
        # Saved ash-agent trajectories should align 1:1. Skip malformed/incomplete
        # histories rather than inventing exact-prefix boundaries.
        raise RuntimeError(
            f"tool boundary mismatch: exact={len(exact_hashes)} normalized={len(normalized)}"
        )

    state_hash = trajectory_prefix_chain_seed_hash()
    rows: list[ToolBoundary] = []
    for step, (message, exact_hash) in enumerate(zip(normalized, exact_hashes), start=1):
        event = canonical_tool_event(
            message,
            allow_safe_shell=True,
            workspace_roots=read_roots,
        )
        effect = str(event["effect"])
        if effect != SAFE_READ:
            state_hash = extend_trajectory_prefix_chain(state_hash, event)
        rows.append(
            ToolBoundary(
                step=step,
                exact_hash=exact_hash,
                state_hash=state_hash,
                effect=effect,
                tool_name=str(event.get("tool_name") or ""),
            )
        )
    return rows


def audit_trajectory(path: Path, read_roots: tuple[str, ...]) -> TrajectoryAudit:
    trajectory = load_json(path)
    messages = trajectory.get("messages") or []
    if not isinstance(messages, list):
        raise RuntimeError("trajectory messages is not a list")
    messages = [row for row in messages if isinstance(row, dict)]
    boundaries = _project_boundaries(messages, read_roots)

    summary_path = path.with_name("summary.json")
    summary = load_json(summary_path) if summary_path.exists() else {}
    task_id = str(
        trajectory.get("instance_id")
        or summary.get("task_id")
        or path.parent.name.split("-2026", 1)[0]
    )
    model = str(summary.get("model") or trajectory.get("model") or "")
    env_key = str(
        summary.get("task_fingerprint")
        or summary.get("environment_image")
        or trajectory.get("environment_image")
        or ""
    )
    model_stats = summary.get("model_stats") or {}
    verification = summary.get("verification") or {}
    partial = verification.get("partial_score")
    return TrajectoryAudit(
        path=path,
        run_dir=path.parent,
        task_id=task_id,
        model=model,
        env_key=env_key,
        summary=summary,
        boundaries=boundaries,
        api_calls=int(model_stats.get("api_calls") or 0),
        input_tokens=int(model_stats.get("input_tokens") or 0),
        output_tokens=int(model_stats.get("output_tokens") or 0),
        rollout_ms=float(summary.get("rollout_ms") or 0.0),
        partial_score=float(partial) if isinstance(partial, (int, float)) else None,
    )


def checkpoint_steps(tool_steps: int, stride: int) -> list[int]:
    if tool_steps <= 0:
        return []
    steps = list(range(stride, tool_steps + 1, stride))
    if not steps or steps[-1] != tool_steps:
        steps.append(tool_steps)
    return steps


def per_trajectory_row(audit: TrajectoryAudit, *, stride: int, snapshot_mib: float) -> dict[str, Any]:
    effects = audit.effects
    rows = audit.boundaries
    state_first_step: dict[str, int] = {}
    relaxed_duplicate_steps = 0
    read_only_zero_diff_steps = 0
    longest_read_only_run = 0
    current_read_run = 0
    for row in rows:
        if row.effect == SAFE_READ:
            read_only_zero_diff_steps += 1
            current_read_run += 1
            longest_read_only_run = max(longest_read_only_run, current_read_run)
        else:
            current_read_run = 0
        if row.state_hash in state_first_step:
            relaxed_duplicate_steps += 1
        else:
            state_first_step[row.state_hash] = row.step

    cp_steps = checkpoint_steps(audit.tool_steps, stride)
    cp_rows = [rows[step - 1] for step in cp_steps]
    unique_exact = len({row.exact_hash for row in cp_rows})
    unique_relaxed = len({row.state_hash for row in cp_rows})
    exact_avoided = len(cp_rows) - unique_exact
    relaxed_avoided = len(cp_rows) - unique_relaxed

    # Search each non-checkpoint tool boundary against snapshots that existed before
    # the query. A strict full-depth exact prefix cannot equal a shorter earlier
    # prefix; Relaxed can still be a current-state hit after proven safe reads.
    cp_by_step = {row.step: row for row in cp_rows}
    prior_exact: set[str] = set()
    prior_state: set[str] = set()
    exact_current_hits = 0
    relaxed_current_hits = 0
    relaxed_only_hits = 0
    replay_steps_saved = 0
    last_cp_step = 0
    for row in rows:
        if row.step in cp_by_step:
            # Register only after evaluating the current boundary so self-hits do not
            # inflate either method.
            pass
        if prior_exact and row.exact_hash in prior_exact:
            exact_current_hits += 1
        relaxed_hit = bool(prior_state and row.state_hash in prior_state)
        if relaxed_hit:
            relaxed_current_hits += 1
            if row.exact_hash not in prior_exact:
                relaxed_only_hits += 1
                replay_steps_saved += max(0, row.step - last_cp_step)
        if row.step in cp_by_step:
            prior_exact.add(row.exact_hash)
            prior_state.add(row.state_hash)
            last_cp_step = row.step

    checkpoint_overheads = audit.summary.get("checkpoint_overheads") or {}
    measured_logical = int(checkpoint_overheads.get("logical_checkpoint_count") or checkpoint_overheads.get("checkpoint_count") or 0)
    measured_physical = int(checkpoint_overheads.get("physical_snapshot_count") or 0)
    measured_reused = int(checkpoint_overheads.get("snapshot_reused_count") or 0)
    measured_snapshot_ms = float(checkpoint_overheads.get("snapshot_ms") or 0.0)
    measured_lookup_ms = float(checkpoint_overheads.get("trajectory_cache_lookup_ms") or 0.0)
    measured_registration_ms = float(checkpoint_overheads.get("trajectory_cache_registration_ms") or 0.0)

    return {
        "task": audit.task_id,
        "run": audit.run_dir.name,
        "model": audit.model,
        "tool_steps": audit.tool_steps,
        "safe_read_steps": effects[SAFE_READ],
        "mutation_steps": effects[MUTATION],
        "barrier_steps": effects[BARRIER],
        "safe_read_fraction": round(effects[SAFE_READ] / audit.tool_steps, 6) if audit.tool_steps else 0.0,
        "zero_diff_projection_steps": read_only_zero_diff_steps,
        "longest_zero_diff_run": longest_read_only_run,
        "unique_relaxed_states_all_steps": len(state_first_step),
        "relaxed_duplicate_state_steps_all": relaxed_duplicate_steps,
        "sim_checkpoint_stride": stride,
        "sim_logical_checkpoints": len(cp_rows),
        "sim_exact_physical_snapshots": unique_exact,
        "sim_relaxed_physical_snapshots": unique_relaxed,
        "sim_exact_snapshots_avoided": exact_avoided,
        "sim_relaxed_snapshots_avoided": relaxed_avoided,
        "sim_snapshot_reduction_fraction": round(relaxed_avoided / len(cp_rows), 6) if cp_rows else 0.0,
        "sim_disk_saved_est_mib": round(relaxed_avoided * snapshot_mib, 3),
        "exact_current_state_hits": exact_current_hits,
        "relaxed_current_state_hits": relaxed_current_hits,
        "relaxed_only_current_state_hits": relaxed_only_hits,
        "relaxed_replay_tool_steps_saved": replay_steps_saved,
        "api_calls": audit.api_calls,
        "input_tokens": audit.input_tokens,
        "output_tokens": audit.output_tokens,
        "rollout_ms": round(audit.rollout_ms, 3),
        "partial_score": audit.partial_score,
        "measured_logical_checkpoints": measured_logical,
        "measured_physical_snapshots": measured_physical,
        "measured_snapshot_reused": measured_reused,
        "measured_snapshot_ms": round(measured_snapshot_ms, 3),
        "measured_cache_lookup_ms": round(measured_lookup_ms, 3),
        "measured_cache_registration_ms": round(measured_registration_ms, 3),
        "trajectory": str(audit.path),
    }


def compatible_env(source: TrajectoryAudit, query: TrajectoryAudit) -> bool:
    if source.task_id != query.task_id:
        return False
    if source.env_key and query.env_key and source.env_key != query.env_key:
        return False
    return True


def cross_run_pair_row(source: TrajectoryAudit, query: TrajectoryAudit, *, stride: int) -> dict[str, Any]:
    src_steps = checkpoint_steps(source.tool_steps, stride)
    src = [source.boundaries[step - 1] for step in src_steps]
    exact = {row.exact_hash for row in src}
    relaxed = {row.state_hash for row in src}

    exact_hits = 0
    relaxed_hits = 0
    relaxed_only = 0
    for row in query.boundaries:
        is_exact = row.exact_hash in exact
        is_relaxed = row.state_hash in relaxed
        exact_hits += int(is_exact)
        relaxed_hits += int(is_relaxed)
        relaxed_only += int(is_relaxed and not is_exact)

    return {
        "task": source.task_id,
        "source_run": source.run_dir.name,
        "query_run": query.run_dir.name,
        "source_model": source.model,
        "query_model": query.model,
        "source_tool_steps": source.tool_steps,
        "query_tool_steps": query.tool_steps,
        "source_checkpoints": len(src),
        "exact_full_depth_hits": exact_hits,
        "relaxed_current_state_hits": relaxed_hits,
        "relaxed_only_hits": relaxed_only,
        "relaxed_hit_gain": relaxed_hits - exact_hits,
        "query_relaxed_hit_rate": round(relaxed_hits / query.tool_steps, 6) if query.tool_steps else 0.0,
        "env_key_verified_equal": bool(source.env_key and query.env_key and source.env_key == query.env_key),
    }


def measured_checkpoint_db_row(audit: TrajectoryAudit, read_roots: tuple[str, ...]) -> dict[str, Any] | None:
    db = audit.run_dir / "checkpoints.sqlite3"
    if not db.exists():
        return None
    task_candidates = [audit.task_id, audit.task_id.removeprefix("swemarathon/")]
    records = []
    with CheckpointStore(db) as store:
        for task_id in task_candidates:
            records = store.list_for_task(task_id)
            if records:
                break
    if not records:
        return None
    records = sorted(records, key=lambda row: row.step_id)
    state_hashes: list[str] = []
    from ash_sandbox.relaxed_prefix import project_environment_prefix

    snapshot_ms = 0.0
    lookup_ms = 0.0
    registration_ms = 0.0
    for record in records:
        projection = project_environment_prefix(
            record.trajectory_prefix,
            allow_safe_shell=True,
            workspace_roots=read_roots,
        )
        state_hashes.append(projection.state_hash)
        meta = record.metadata or {}
        snapshot_ms += float(meta.get("snapshot_ms") or 0.0)
        lookup_ms += float(meta.get("trajectory_cache_lookup_ms") or 0.0)
        registration_ms += float(meta.get("trajectory_cache_registration_ms") or 0.0)
    unique = len(set(state_hashes))
    return {
        "task": audit.task_id,
        "run": audit.run_dir.name,
        "logical_checkpoints": len(records),
        "unique_relaxed_states": unique,
        "counterfactual_physical_snapshots": unique,
        "counterfactual_snapshots_avoided": len(records) - unique,
        "counterfactual_snapshot_reduction_fraction": round((len(records) - unique) / len(records), 6),
        "recorded_snapshot_ms": round(snapshot_ms, 3),
        "recorded_lookup_ms": round(lookup_ms, 3),
        "recorded_registration_ms": round(registration_ms, 3),
        "checkpoint_db": str(db),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(title for _, title in fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for key, _ in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}".rstrip("0").rstrip(".")
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the advisor-requested Exact-vs-Relaxed real-trajectory search table.")
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/relaxed-search-final"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--checkpoint-stride", type=int, default=10)
    ap.add_argument("--snapshot-mib", type=float, default=27.6, help="Illustrative prior measured snapshot-equivalent size; used only for estimated disk savings.")
    ap.add_argument("--model-substring", default="qwen", help="Prefer trajectories whose summary model contains this substring.")
    ap.add_argument("--read-root", action="append", dest="read_roots")
    args = ap.parse_args()
    if args.limit < 1 or args.checkpoint_stride < 1:
        raise ValueError("limit and checkpoint stride must be >= 1")
    read_roots = tuple(args.read_roots or DEFAULT_READ_ROOTS)

    paths = sorted(args.results_root.rglob("trajectory.json"))
    audits: list[TrajectoryAudit] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            audit = audit_trajectory(path, read_roots)
        except Exception as exc:
            errors.append({"trajectory": str(path), "error": str(exc)})
            continue
        if audit.tool_steps == 0:
            continue
        audits.append(audit)

    preferred = [a for a in audits if args.model_substring.lower() in a.model.lower()]
    pool = preferred or audits
    pool = sorted(pool, key=lambda a: (a.tool_steps, a.input_tokens, a.output_tokens), reverse=True)

    selected: list[TrajectoryAudit] = []
    seen_tasks: set[str] = set()
    for audit in pool:
        if audit.task_id in seen_tasks:
            continue
        selected.append(audit)
        seen_tasks.add(audit.task_id)
        if len(selected) >= args.limit:
            break
    if len(selected) < args.limit:
        selected_paths = {a.path for a in selected}
        for audit in pool:
            if audit.path in selected_paths:
                continue
            selected.append(audit)
            selected_paths.add(audit.path)
            if len(selected) >= args.limit:
                break

    rows = [per_trajectory_row(a, stride=args.checkpoint_stride, snapshot_mib=args.snapshot_mib) for a in selected]

    pair_rows: list[dict[str, Any]] = []
    groups: dict[str, list[TrajectoryAudit]] = defaultdict(list)
    for audit in pool:
        groups[audit.task_id].append(audit)
    for group in groups.values():
        if len(group) < 2:
            continue
        # Limit quadratic work to the 12 longest real trajectories for a task.
        group = sorted(group, key=lambda a: a.tool_steps, reverse=True)[:12]
        for source in group:
            for query in group:
                if source.path == query.path or not compatible_env(source, query):
                    continue
                pair_rows.append(cross_run_pair_row(source, query, stride=args.checkpoint_stride))
    pair_rows.sort(
        key=lambda row: (row["relaxed_only_hits"], row["relaxed_hit_gain"], row["query_tool_steps"]),
        reverse=True,
    )
    pair_rows = pair_rows[: args.limit]

    measured_rows = []
    for audit in audits:
        row = measured_checkpoint_db_row(audit, read_roots)
        if row is not None:
            measured_rows.append(row)
    measured_rows.sort(key=lambda row: (row["logical_checkpoints"], row["recorded_snapshot_ms"]), reverse=True)

    aggregate = {
        "trajectory_count": len(rows),
        "tool_steps": sum(int(row["tool_steps"]) for row in rows),
        "safe_read_steps": sum(int(row["safe_read_steps"]) for row in rows),
        "mutation_steps": sum(int(row["mutation_steps"]) for row in rows),
        "barrier_steps": sum(int(row["barrier_steps"]) for row in rows),
        "zero_diff_projection_steps": sum(int(row["zero_diff_projection_steps"]) for row in rows),
        "sim_logical_checkpoints": sum(int(row["sim_logical_checkpoints"]) for row in rows),
        "sim_exact_physical_snapshots": sum(int(row["sim_exact_physical_snapshots"]) for row in rows),
        "sim_relaxed_physical_snapshots": sum(int(row["sim_relaxed_physical_snapshots"]) for row in rows),
        "sim_relaxed_snapshots_avoided": sum(int(row["sim_relaxed_snapshots_avoided"]) for row in rows),
        "sim_disk_saved_est_mib": round(sum(float(row["sim_disk_saved_est_mib"]) for row in rows), 3),
        "exact_current_state_hits": sum(int(row["exact_current_state_hits"]) for row in rows),
        "relaxed_current_state_hits": sum(int(row["relaxed_current_state_hits"]) for row in rows),
        "relaxed_only_current_state_hits": sum(int(row["relaxed_only_current_state_hits"]) for row in rows),
        "relaxed_replay_tool_steps_saved": sum(int(row["relaxed_replay_tool_steps_saved"]) for row in rows),
        "api_calls": sum(int(row["api_calls"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
    }
    logical = aggregate["sim_logical_checkpoints"]
    aggregate["sim_snapshot_reduction_fraction"] = round(
        aggregate["sim_relaxed_snapshots_avoided"] / logical, 6
    ) if logical else 0.0
    tool_steps = aggregate["tool_steps"]
    aggregate["safe_read_fraction"] = round(aggregate["safe_read_steps"] / tool_steps, 6) if tool_steps else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol": {
            "results_root": str(args.results_root),
            "selected_limit": args.limit,
            "checkpoint_stride": args.checkpoint_stride,
            "read_roots": list(read_roots),
            "snapshot_mib_estimate": args.snapshot_mib,
            "selection": "prefer longest Qwen trajectories with unique tasks, then fill remaining slots",
            "exact_hit_definition": "full-depth current-state exact prefix equal to a previously materialized checkpoint; self-hits excluded",
            "relaxed_hit_definition": "current Relaxed Projection state equal to a previously materialized checkpoint while target model history is not reused",
            "disk_note": "sim_disk_saved_est_mib is an extrapolation using the prior measured 27.6 MiB snapshot-equivalent size, not per-task physical disk measurement",
        },
        "aggregate": aggregate,
        "trajectories": rows,
        "cross_run_pairs": pair_rows,
        "measured_checkpoint_dbs": measured_rows,
        "parse_errors": errors,
    }
    (args.output_dir / "real_trajectory_search.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(args.output_dir / "real_trajectory_search.csv", rows)
    write_csv(args.output_dir / "cross_run_pairs.csv", pair_rows)
    write_csv(args.output_dir / "measured_checkpoint_dbs.csv", measured_rows)

    report = [
        "# Exact vs Relaxed Prefix Search — Real-Trajectory Audit\n",
        "This report is the 5–10-example experiment requested in the advisor meeting. It uses recorded real trajectories; it does not spend new model/API budget.\n",
        "## Protocol\n",
        f"- Selected real trajectories: **{len(rows)}** (prefer Qwen, unique tasks first).\n",
        f"- Logical checkpoint simulation: every **{args.checkpoint_stride} tool results**, plus the final tool boundary.\n",
        "- `Exact current-state hit`: current full trajectory prefix exactly equals an earlier materialized checkpoint prefix (self-hit excluded).\n",
        "- `Relaxed current-state hit`: current environment projection equals an earlier materialized checkpoint after dropping only proven safe reads. Model history/KV is never reused for a relaxed hit.\n",
        "- `zero_diff_projection_steps` means the conservative effect grammar proves the action does not mutate trusted environment state; actual Docker workspace-digest equality is validated separately by the real Docker smoke.\n",
        f"- Disk MiB is only an **illustrative extrapolation** at {args.snapshot_mib:.1f} MiB per prior measured snapshot-equivalent; snapshot counts are the primary storage result.\n",
        "\n## Aggregate\n",
        f"- Tool steps: **{aggregate['tool_steps']}**\n",
        f"- Proven safe-read / zero-diff projection steps: **{aggregate['safe_read_steps']} ({aggregate['safe_read_fraction']:.2%})**\n",
        f"- Simulated logical checkpoints: **{aggregate['sim_logical_checkpoints']}**\n",
        f"- Relaxed physical snapshots: **{aggregate['sim_relaxed_physical_snapshots']}**\n",
        f"- Snapshots avoided vs Exact: **{aggregate['sim_relaxed_snapshots_avoided']} ({aggregate['sim_snapshot_reduction_fraction']:.2%})**\n",
        f"- Relaxed-only current-state hits: **{aggregate['relaxed_only_current_state_hits']}**\n",
        f"- Replay tool steps avoided by those hits: **{aggregate['relaxed_replay_tool_steps_saved']}**\n",
        f"- Recorded API calls represented: **{aggregate['api_calls']}**\n",
        f"- Recorded input/output tokens represented: **{aggregate['input_tokens']} / {aggregate['output_tokens']}**\n",
        "\n## Per-trajectory results\n",
        md_table(rows, [
            ("task", "Task"),
            ("tool_steps", "Steps"),
            ("safe_read_fraction", "Safe-read frac"),
            ("longest_zero_diff_run", "Longest zero-diff run"),
            ("sim_logical_checkpoints", "Logical CP"),
            ("sim_relaxed_snapshots_avoided", "CP saved"),
            ("sim_snapshot_reduction_fraction", "CP reduction"),
            ("exact_current_state_hits", "Exact current hits"),
            ("relaxed_only_current_state_hits", "Relaxed-only hits"),
            ("relaxed_replay_tool_steps_saved", "Replay steps saved"),
            ("partial_score", "Partial"),
        ]),
        "\n## Cross-run same-task search pairs\n",
        md_table(pair_rows, [
            ("task", "Task"),
            ("source_run", "Source"),
            ("query_run", "Query"),
            ("source_checkpoints", "Source CP"),
            ("exact_full_depth_hits", "Exact hits"),
            ("relaxed_current_state_hits", "Relaxed hits"),
            ("relaxed_only_hits", "Relaxed-only"),
            ("query_relaxed_hit_rate", "Relaxed hit rate"),
            ("env_key_verified_equal", "Env verified"),
        ]),
        "\n## Recorded checkpoint databases\n",
        md_table(measured_rows, [
            ("task", "Task"),
            ("run", "Run"),
            ("logical_checkpoints", "Logical CP"),
            ("unique_relaxed_states", "Unique relaxed states"),
            ("counterfactual_snapshots_avoided", "CP avoidable"),
            ("counterfactual_snapshot_reduction_fraction", "Reduction"),
            ("recorded_snapshot_ms", "Snapshot ms"),
            ("recorded_lookup_ms", "Lookup ms"),
            ("recorded_registration_ms", "Registration ms"),
        ]),
        "\n## Claim boundary\n",
        "The audit supports **environment snapshot reuse**, not model-history or KV-cache reuse. A relaxed match may restore the matched environment snapshot while the continuation must keep the target branch's own model-facing history.\n",
    ]
    (args.output_dir / "report.md").write_text("".join(report), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected": len(rows),
        "aggregate": aggregate,
        "cross_run_pairs": len(pair_rows),
        "measured_checkpoint_dbs": len(measured_rows),
        "parse_errors": len(errors),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
