"""Oracle/no-op checks through the same preparation, snapshot and verifier path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4

from harness.execution.session import SandboxSession
from swebench.fork_eval import backend_for
from swebench_pro.bench import SWEbenchPro
from swebench_pro.grade import checked
from swebench_pro.tasks import Task


def check(bench: SWEbenchPro, task: Task, backend: dict, mode: str, directory: Path) -> dict:
    instance = bench.instance(task)
    snapshot_id = bench.prepare_image(instance, backend, directory / "preparation")
    if mode == "oracle":
        patch = task.sample.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError(f"Missing gold patch for {task.instance_id}")
        patch_path = directory / "gold.patch"
        patch_path.write_text(patch)
        session = SandboxSession(quiet=True, backend=dict(backend))
        try:
            if not session.create(snapshot_id, bench.resources(instance)):
                raise RuntimeError(f"Could not restore oracle VM: {session.create_error}")
            if not session.upload_file(patch_path, "/tmp/ash-pro-gold.patch"):
                raise RuntimeError("Could not upload gold patch")
            checked(session, "cd /app && git apply /tmp/ash-pro-gold.patch")
            snapshot = session.snapshot(name=f"pro-oracle-{uuid4().hex}", disk_only=True)
            if snapshot is None:
                raise RuntimeError("Could not snapshot oracle")
            snapshot_id = snapshot.id
        finally:
            session.destroy()
    instance["verifier_artifacts_dir"] = str(directory / "verifier")
    grade = bench.grade(snapshot_id, instance, backend)
    expected = mode == "oracle"
    record = {"instance_id": task.instance_id, "mode": mode, "snapshot_id": snapshot_id,
              "ok": grade.error is None and grade.verifier_artifact_error is None
              and grade.resolved == expected, "expected_resolved": expected, "grade": asdict(grade)}
    (directory / "gate.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro-repo", required=True)
    parser.add_argument("--pro-data")
    parser.add_argument("--pro-dataset-revision")
    parser.add_argument("--pro-cpus", type=int, default=4)
    parser.add_argument("--pro-memory-mb", type=int, default=16384)
    parser.add_argument("--pro-verifier-timeout", type=int, default=3600)
    parser.add_argument("--pro-block-network", action="store_true")
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    parser.add_argument("--instance", required=True, help="Comma-separated task IDs to check")
    parser.add_argument("--mode", choices=["both", "oracle", "nop"], default="both")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()
    args.timeout = args.pro_verifier_timeout + 600
    bench = SWEbenchPro(args)
    catalogue = bench.catalogue(args)
    wanted = args.instance.split(",")
    if set(wanted) - catalogue.keys():
        parser.error("Unknown task IDs")
    backend = backend_for(args, bench)
    modes = ["oracle", "nop"] if args.mode == "both" else [args.mode]
    passed = True
    for instance_id in wanted:
        for mode in modes:
            directory = Path(args.out).resolve() / instance_id / f"{mode}-{uuid4().hex[:8]}"
            directory.mkdir(parents=True)
            try:
                record = check(bench, catalogue[instance_id], backend, mode, directory)
            except Exception as exc:
                record = {"instance_id": instance_id, "mode": mode, "ok": False, "error": str(exc)}
                (directory / "gate.json").write_text(json.dumps(record, indent=2))
            passed = passed and record["ok"]
            print(json.dumps({key: value for key, value in record.items() if key != "grade"}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
