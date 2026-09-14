import json
from pathlib import Path

from swebench_pro.recovery import durable_json, reconcile


def test_reconciliation_preserves_results_and_distinguishes_resume_from_fresh(tmp_path: Path) -> None:
    source = tmp_path / "old"
    output = tmp_path / "new"
    tasks = [{"index": index, "id": f"task-{index}"} for index in range(3)]
    durable_json(source / "manifest.json", {"tasks": tasks})
    directory = source / "shard-000/task-0"
    artifacts = directory / "verifier"
    artifacts.mkdir(parents=True)
    for name in ("metadata.json", "grade.json", "verifier-logs.tar.gz"):
        (artifacts / name).write_text("evidence")
    (directory / "parent.jsonl").write_text('{"type":"run.finished"}\n')
    durable_json(directory / "grade.json", {"evidence_valid": True, "grade": {
        "resolved": True, "verifier_artifacts": str(artifacts)}})
    interrupted = source / "shard-001/task-1"
    interrupted.mkdir(parents=True)
    (interrupted.parent / "worker.json").write_text("")
    (interrupted / "parent.jsonl").write_text(
        '{"type":"run.started"}\n'
        '{"type":"checkpoint.captured","reason":"captured","snapshot_id":"snapshot","step":4}\n')
    before = (directory / "grade.json").read_bytes()
    report = reconcile(source, output)
    assert report["actions"] == {"preserve": 1, "resume": 1, "fresh": 1}
    assert report["preserved_resolved"] == 1
    assert report["tasks"][1]["last_snapshot"]["snapshot_id"] == "snapshot"
    assert report["all_images_required_before_actor"]
    assert (directory / "grade.json").read_bytes() == before
    assert (interrupted.parent / "worker.json").read_bytes() == b""


def test_durable_json_replaces_only_the_target(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    durable_json(target, {"step": 1})
    durable_json(target, {"step": 2})
    assert json.loads(target.read_text()) == {"step": 2}
    assert list(tmp_path.iterdir()) == [target]
