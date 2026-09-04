#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ash_sandbox import CheckpointStore
from ash_sandbox.relaxed_prefix import project_environment_prefix


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tool_result_prefixes(messages: list[dict[str, Any]]):
    prefix: list[dict[str, Any]] = []
    tool_index = 0
    for message_index, message in enumerate(messages):
        prefix.append(message)
        if message.get("role") != "tool_result":
            continue
        tool_index += 1
        yield tool_index, message_index, list(prefix)


def main() -> int:
    ap = argparse.ArgumentParser(description="Find real cross-run Relaxed Projection matches against live checkpoint indexes.")
    ap.add_argument("--old-trajectory", required=True, type=Path)
    ap.add_argument("--checkpoint-db", required=True, type=Path)
    ap.add_argument("--task-id", required=True)
    ap.add_argument(
        "--read-root",
        action="append",
        dest="read_roots",
        help="Trusted read-only filesystem root for shell proof; may be repeated. Default: /app",
    )
    ap.add_argument("--max-matches", type=int, default=30)
    args = ap.parse_args()
    read_roots = tuple(args.read_roots or ["/app"])

    old = load(args.old_trajectory)
    messages = list(old.get("messages") or [])
    with CheckpointStore(args.checkpoint_db) as checkpoints:
        records = checkpoints.list_for_task(args.task_id)
    if not records:
        raise RuntimeError("no retry checkpoint records found")
    envs = {record.env_fingerprint for record in records}
    if len(envs) != 1:
        raise RuntimeError(f"expected one environment fingerprint, found {len(envs)}")

    # Recompute checkpoint projections with the *current* proof grammar instead of
    # trusting hashes persisted by an older grammar version. This makes the audit
    # useful while the conservative shell recognizer evolves.
    retry_states: dict[str, list[Any]] = {}
    retry_projection_rows = []
    for record in records:
        projection = project_environment_prefix(
            record.trajectory_prefix,
            allow_safe_shell=True,
            workspace_roots=read_roots,
        )
        retry_states.setdefault(projection.state_hash, []).append(record)
        retry_projection_rows.append({
            "step": record.step_id,
            "snapshot_id": record.snapshot_id,
            "state_hash": projection.state_hash,
            "tool_steps": projection.tool_steps,
            "state_steps": projection.state_steps,
            "ignored_read_steps": projection.ignored_read_steps,
        })

    matches = []
    projected_states = set()
    safe_read_steps = 0
    tool_steps_total = 0
    for tool_index, message_index, prefix in tool_result_prefixes(messages):
        projection = project_environment_prefix(
            prefix,
            allow_safe_shell=True,
            workspace_roots=read_roots,
        )
        projected_states.add(projection.state_hash)
        safe_read_steps = projection.ignored_read_steps
        tool_steps_total = projection.tool_steps
        for record in retry_states.get(projection.state_hash, []):
            matches.append({
                "old_tool_index": tool_index,
                "old_message_index": message_index,
                "old_tool_name": message.get("tool_name"),
                "projection_tool_steps": projection.tool_steps,
                "projection_state_steps": projection.state_steps,
                "projection_ignored_read_steps": projection.ignored_read_steps,
                "state_hash": projection.state_hash,
                "retry_snapshot_id": record.snapshot_id,
                "retry_checkpoint_step": record.step_id,
                "retry_trajectory_id": record.trajectory_id,
            })

    # Collapse consecutive old tool results that resolve to the same materialized
    # retry checkpoint; keep first/last positions so read-only stretches are visible.
    grouped: list[dict[str, Any]] = []
    for row in matches:
        key = (row["retry_snapshot_id"], row["state_hash"])
        if grouped and grouped[-1]["_key"] == key:
            grouped[-1]["old_tool_index_last"] = row["old_tool_index"]
            grouped[-1]["old_message_index_last"] = row["old_message_index"]
            grouped[-1]["match_count"] += 1
        else:
            grouped.append({
                **row,
                "_key": key,
                "old_tool_index_first": row["old_tool_index"],
                "old_tool_index_last": row["old_tool_index"],
                "old_message_index_first": row["old_message_index"],
                "old_message_index_last": row["old_message_index"],
                "match_count": 1,
            })
    for row in grouped:
        row.pop("_key", None)
        row.pop("old_tool_index", None)
        row.pop("old_message_index", None)

    output = {
        "task_id": args.task_id,
        "old_trajectory": str(args.old_trajectory),
        "retry_checkpoint_db": str(args.checkpoint_db),
        "read_roots": list(read_roots),
        "retry_checkpoint_count": len(records),
        "retry_unique_projected_states": len(retry_states),
        "retry_projection_rows": retry_projection_rows,
        "old_tool_results": tool_steps_total,
        "old_ignored_safe_reads_at_end": safe_read_steps,
        "old_unique_projected_states": len(projected_states),
        "raw_match_positions": len(matches),
        "grouped_match_regions": len(grouped),
        "matches": grouped[: args.max_matches],
        "historical_model_prefix_reconstructable": False,
        "continuation_eligible": False,
        "note": "Historical trajectory lacks complete assistant tool_calls; matches prove environment equivalence opportunities only.",
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
