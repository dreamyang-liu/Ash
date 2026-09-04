#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import uuid
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=True)


def parse_size_bytes(value: str) -> int:
    value = value.strip()
    units = {"B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2, "GB": 1000**3}
    for unit in ("GB", "MB", "kB", "KB", "B"):
        if value.endswith(unit):
            return int(float(value[:-len(unit)].strip()) * units[unit])
    raise ValueError(value)


def top_layer_bytes(image: str) -> int:
    p = run(["docker", "history", "--no-trunc", "--format", "{{.Size}}", image])
    first = next((line.strip() for line in p.stdout.splitlines() if line.strip()), "0B")
    return parse_size_bytes(first)


def one_trial(base_image: str, condition: str, trial: int) -> dict[str, float | int]:
    suffix = uuid.uuid4().hex[:10]
    container = f"ash-storage-probe-{suffix}"
    image = f"ash-storage-probe:{condition}-{suffix}"
    try:
        run(["docker", "run", "-d", "--name", container, base_image, "sleep", "infinity"])
        if condition == "readonly":
            run([
                "docker", "exec", container, "sh", "-lc",
                "ls -la /app >/dev/null; cat /app/rfc8878.txt >/dev/null; grep -n Frames /app/rfc8878.txt >/dev/null",
            ])
        elif condition == "mutation":
            run(["docker", "exec", container, "sh", "-lc", "printf x > /app/.ash_storage_probe"])
        elif condition != "clean":
            raise ValueError(condition)

        t0 = time.perf_counter()
        run(["docker", "commit", container, image])
        commit_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "trial": trial,
            "commit_ms": round(commit_ms, 3),
            "top_layer_bytes": top_layer_bytes(image),
        }
    finally:
        subprocess.run(["docker", "rm", "-f", container], text=True, capture_output=True, check=False)
        subprocess.run(["docker", "rmi", "-f", image], text=True, capture_output=True, check=False)


def summarize(rows: list[dict[str, float | int]]) -> dict[str, object]:
    times = [float(r["commit_ms"]) for r in rows]
    sizes = [int(r["top_layer_bytes"]) for r in rows]
    return {
        "trials": rows,
        "commit_ms_mean": round(statistics.fmean(times), 3),
        "commit_ms_median": round(statistics.median(times), 3),
        "commit_ms_min": round(min(times), 3),
        "commit_ms_max": round(max(times), 3),
        "top_layer_bytes_mean": statistics.fmean(sizes),
        "top_layer_bytes_median": statistics.median(sizes),
        "top_layer_bytes_min": min(sizes),
        "top_layer_bytes_max": max(sizes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-image", default="ash-swemarathon:lightstage-zstd-decoder-fdfb234a79f9")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", default="results/prefix-baseline-comparison/snapshot_storage_probe.json")
    args = ap.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")

    result: dict[str, object] = {
        "base_image": args.base_image,
        "repeats": args.repeats,
        "conditions": {},
        "note": "top_layer_bytes is the writable diff reported by docker history for the committed image; shared base layers are excluded.",
    }
    for condition in ("clean", "readonly", "mutation"):
        rows = [one_trial(args.base_image, condition, i + 1) for i in range(args.repeats)]
        result["conditions"][condition] = summarize(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
