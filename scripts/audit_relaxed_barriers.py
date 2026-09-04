#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ash_sandbox import CheckpointStore
from ash_sandbox.relaxed_prefix import (
    BARRIER,
    MUTATION,
    SAFE_READ,
    canonical_tool_event,
    normalize_tool_result_messages,
)


def shell_family(command: str) -> str:
    command = str(command or "").strip()
    if not command:
        return "<empty>"
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "<parse-error>"
    if not tokens:
        return "<empty>"
    if tokens[0] == "cd" and "&&" in tokens:
        idx = tokens.index("&&")
        if idx + 1 < len(tokens):
            return tokens[idx + 1]
    return tokens[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Relaxed Change effect classifications on a real checkpointed trajectory.")
    ap.add_argument("--checkpoint-db", required=True, type=Path)
    ap.add_argument("--task-id", required=True)
    ap.add_argument(
        "--read-root",
        action="append",
        dest="read_roots",
        help="Trusted read-only filesystem root for shell proof; may be repeated. Default: /app",
    )
    ap.add_argument("--sample-per-family", type=int, default=5)
    args = ap.parse_args()
    read_roots = tuple(args.read_roots or ["/app"])

    with CheckpointStore(args.checkpoint_db) as store:
        rows = store.list_for_task(args.task_id)
    if not rows:
        raise RuntimeError("no checkpoints found")
    last = max(rows, key=lambda row: row.step_id)

    effect_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    barrier_tool_counts: Counter[str] = Counter()
    shell_barrier_families: Counter[str] = Counter()
    shell_safe_families: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)

    normalized = list(normalize_tool_result_messages(last.trajectory_prefix))
    for message in normalized:
        event = canonical_tool_event(
            message,
            allow_safe_shell=True,
            workspace_roots=read_roots,
        )
        effect = str(event["effect"])
        tool = str(event["tool_name"])
        effect_counts[effect] += 1
        tool_counts[tool] += 1
        if effect == BARRIER:
            barrier_tool_counts[tool] += 1
        if tool == "shell":
            command = str((event.get("tool_args") or {}).get("command") or "")
            family = shell_family(command)
            if effect == BARRIER:
                shell_barrier_families[family] += 1
                if len(samples[family]) < args.sample_per_family:
                    samples[family].append(command)
            elif effect == SAFE_READ:
                shell_safe_families[family] += 1

    output: dict[str, Any] = {
        "task_id": args.task_id,
        "checkpoint_step": last.step_id,
        "checkpoint_count": len(rows),
        "tool_result_count": len(normalized),
        "effect_counts": dict(effect_counts),
        "safe_read_fraction": round(effect_counts[SAFE_READ] / len(normalized), 6) if normalized else 0.0,
        "mutation_fraction": round(effect_counts[MUTATION] / len(normalized), 6) if normalized else 0.0,
        "barrier_fraction": round(effect_counts[BARRIER] / len(normalized), 6) if normalized else 0.0,
        "tool_counts": dict(tool_counts.most_common()),
        "barrier_tool_counts": dict(barrier_tool_counts.most_common()),
        "safe_shell_families": dict(shell_safe_families.most_common()),
        "barrier_shell_families": dict(shell_barrier_families.most_common()),
        "barrier_shell_samples": {family: samples[family] for family, _ in shell_barrier_families.most_common()},
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
