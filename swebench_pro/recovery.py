"""Read-only cohort reconciliation and registry inventory before Pro recovery."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


def durable_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict | None:
    if not path.exists() or not path.stat().st_size:
        return None
    return json.loads(path.read_text())


def reconcile(source: Path, output: Path) -> dict:
    manifest = read_json(source / "manifest.json")
    if not manifest:
        raise ValueError("Missing source manifest")
    rows = []
    preserved = {}
    for item in manifest["tasks"]:
        directory = source / f"shard-{item['index']:03d}" / item["id"]
        grade_path = directory / "grade.json"
        grade = read_json(grade_path)
        row = {**item, "source_directory": str(directory)}
        if grade and grade.get("evidence_valid"):
            verdict = grade["grade"]
            artifact = Path(verdict["verifier_artifacts"])
            if verdict.get("error") or verdict.get("verifier_artifact_error"):
                raise ValueError(f"Inconsistent retained grade: {grade_path}")
            for path in (grade_path, artifact / "metadata.json", artifact / "grade.json",
                         artifact / "verifier-logs.tar.gz", directory / "parent.jsonl"):
                preserved[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            row.update(action="preserve", resolved=verdict["resolved"], grade_path=str(grade_path))
        else:
            journal = directory / "parent.jsonl"
            events = []
            if journal.exists():
                for line in journal.read_text().splitlines():
                    if line.strip():
                        events.append(json.loads(line))
                preserved[str(journal)] = hashlib.sha256(journal.read_bytes()).hexdigest()
            started = any(event.get("type") == "run.started" for event in events)
            finished = any(event.get("type") == "run.finished" for event in events)
            snapshots = [event for event in events if event.get("type") == "checkpoint.captured"
                         and event.get("reason") == "captured" and event.get("snapshot_id")]
            row.update(action="regrade" if finished else "resume" if started else "fresh",
                       last_snapshot=snapshots[-1] if snapshots else None,
                       journal=str(journal) if journal.exists() else None)
        rows.append(row)
    report = {"source": str(source), "created_at": datetime.now(timezone.utc).isoformat(),
              "total_tasks": len(rows), "actions": dict(Counter(row["action"] for row in rows)),
              "preserved_resolved": sum(row.get("resolved") is True for row in rows),
              "tasks": rows, "preserved_sha256": preserved,
              "all_images_required_before_actor": True}
    durable_json(output / "recovery-plan.json", report)
    return report


def inventory(source: Path, output: Path, regctl: Path, conversion_index: Path, workers: int) -> dict:
    manifest = read_json(source / "manifest.json")
    cached = set(json.loads(conversion_index.read_text())["cached_source_digests"])

    def inspect(item: dict) -> dict:
        path = output / "registry-manifests" / f"{item['id']}.json"
        previous = read_json(path)
        if previous and previous.get("image") == item["image"] and previous.get("ok"):
            return previous
        result = subprocess.run([str(regctl), "manifest", "get", item["image"], "--format", "raw-body"],
                                capture_output=True, text=True, timeout=90)
        row = {"instance_id": item["id"], "image": item["image"], "ok": False}
        if result.returncode == 0:
            body = json.loads(result.stdout)
            if isinstance(body.get("layers"), list):
                row.update(ok=True, manifest=body)
            else:
                row["error"] = "Registry response lacks a concrete layer manifest"
        else:
            row["error"] = result.stderr[-1000:]
        durable_json(path, row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(inspect, item) for item in manifest["tasks"]]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 100 == 0:
                print(f"inventoried {len(rows)}/{len(futures)}", flush=True)
    layers = {layer["digest"]: layer["size"] for row in rows if row.get("ok") for layer in row["manifest"]["layers"]}
    missing = {digest: size for digest, size in layers.items() if digest not in cached}
    report = {"at": datetime.now(timezone.utc).isoformat(), "images": len(rows),
              "manifests_ok": sum(row["ok"] for row in rows),
              "errors": [row for row in rows if not row["ok"]],
              "unique_layers": len(layers), "uncached_layers": len(missing),
              "registry_compressed_bytes": sum(layers.values()),
              "uncached_registry_compressed_bytes": sum(missing.values()),
              "disk_free_bytes": shutil.disk_usage(output).free,
              "note": "Registry compressed bytes are an inventory measure, not exact converted disk requirements. "
                      "Cache credit is optimistic: any existing conversion of the source layer is credited, "
                      "even if a different parent chain may require another conversion. "
                      "No layer blobs downloaded, no images prepared, no actors launched."}
    durable_json(output / "image-inventory.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["audit", "inventory"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--regctl", type=Path)
    parser.add_argument("--conversion-index", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "audit":
        report = reconcile(args.source.resolve(), args.out.resolve())
        print(json.dumps({key: report[key] for key in ("total_tasks", "actions", "preserved_resolved")}))
    else:
        if not args.regctl or not args.conversion_index or args.workers < 1:
            parser.error("inventory requires --regctl, --conversion-index and positive workers")
        report = inventory(args.source.resolve(), args.out.resolve(), args.regctl, args.conversion_index, args.workers)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
