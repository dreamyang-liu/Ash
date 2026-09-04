#!/usr/bin/env python3
"""Build the fixed 10-task, all-available-rollout cache evaluation corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    old = json.loads(
        (repo / "results/report-ready-cache-10task/corpus-manifest.json").read_text()
    )
    images = {row["task"]: row["environment_image"] for row in old["rows"]}
    trajectories = [
        *sorted((repo / "results/swemarathon-9x2-lightstage/qwen").glob("*/trajectory.json")),
        *sorted((repo / "results/swemarathon-9x2-lightstage/luna").glob("*/trajectory.json")),
        *sorted((repo / "results/report-ready-cache-10task/rollouts").glob("*/*/trajectory.json")),
    ]
    rows = []
    for path in trajectories:
        trajectory = json.loads(path.read_text())
        task = path.parent.name.rsplit("-2026", 1)[0]
        if task not in images:
            continue
        model = str(trajectory.get("model") or trajectory.get("metadata", {}).get("model") or "")
        policy = "qwen" if "qwen" in path.parts else "luna"
        calls = sum(
            isinstance(message, dict) and message.get("role") == "tool_result"
            for message in trajectory.get("messages", [])
        )
        rows.append({
            "task": task,
            "run_id": path.parent.name,
            "policy": policy,
            "model": model or ("openai/qwen3.8-27b" if policy == "qwen" else "gpt-5.6-luna"),
            "trajectory": str(path.relative_to(repo)),
            "trajectory_sha256": sha256(path),
            "tool_boundaries": calls,
            "environment_image": images[task],
        })
    rows.sort(key=lambda row: (row["task"], row["policy"], row["run_id"]))
    tasks = sorted({row["task"] for row in rows})
    manifest = {
        "schema_version": 2,
        "design": "10 tasks; every available independent light-stage rollout; full recorded tool trajectory",
        "statistical_unit": "task",
        "task_count": len(tasks),
        "rollout_count": len(rows),
        "tool_boundary_count": sum(row["tool_boundaries"] for row in rows),
        "pairing": "all ordered source/query rollout pairs within the same task; no self pairs",
        "rows": rows,
    }
    if len(tasks) != 10:
        raise RuntimeError(f"expected 10 tasks, found {len(tasks)}: {tasks}")
    output = repo / "results/paper-cache-10task/corpus-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(output)
    print(json.dumps({key: manifest[key] for key in ("task_count", "rollout_count", "tool_boundary_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
