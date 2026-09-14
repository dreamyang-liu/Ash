from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from scripts import official_swebench as official
from scripts import official_swebench_runner as runner
from scripts.official_swebench_runner import limit_container_creation, prepare_images


def attempt(task: str, role: str = "parent", **changes: object) -> dict:
    return {"source": "main", "task": task, "attempt": role, "role": role,
            "branch_slot": 0 if role == "parent" else 1, "ok": True,
            "prediction": {"instance_id": task, "model_name_or_path": f"main/{role}",
                           "model_patch": "patch"}, **changes}


def write_report(root: Path, record: dict, resolved: bool) -> Path:
    batch = "parent" if record["role"] == "parent" else "branch-slot-01"
    path = official.report_path(root, f"run-{batch}", record["prediction"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({record["task"]: {"resolved": resolved}}))
    return path


def test_summary_keeps_missing_exports_and_unseen_parents_unmeasured(tmp_path: Path) -> None:
    records = [attempt("success"), attempt("failure"), attempt("missing"),
               attempt("export", ok=False), attempt("empty", prediction={}),
               attempt("failure", "branch"), attempt("missing", "branch")]
    write_report(tmp_path, records[0], True)
    write_report(tmp_path, records[1], False)
    write_report(tmp_path, records[5], True)
    expected = {"success", "failure", "missing", "export", "empty", "unseen"}
    summary, rows = official.summarize_attempts(tmp_path, "run", records, expected)
    assert summary["parent_submitted"] == 3
    assert summary["parent_completed"] == 2
    assert summary["parent_resolved"] == 1
    assert summary["parent_unresolved"] == 1
    assert summary["parent_missing_reports"] == 1
    assert summary["parent_export_errors"] == 1
    assert summary["parent_empty_patches"] == 1
    assert summary["missing_parent_ids"] == ["unseen"]
    assert summary["parent_final_rate"] is None
    assert summary["parent_resolved_total_lower_bound"] == 1 / 6
    assert summary["parent_resolved_completed_rate"] == 0.5
    assert summary["combined_resolved_tasks"] == 2
    assert summary["missing_attempt_reports"] == 2
    assert len(rows) == len(records)
    assert rows[2]["official_resolved"] is None
    assert rows[4]["official_resolved"] is False


def test_empty_patch_is_a_measured_failure_without_an_official_report(tmp_path: Path) -> None:
    summary, rows = official.summarize_attempts(
        tmp_path, "run", [attempt("empty", prediction={})], {"empty"})
    assert summary["parent_complete"]
    assert summary["parent_final_rate"] == 0
    assert summary["parent_resolved_completed_rate"] is None


def test_duplicate_parent_cohorts_are_not_silently_overwritten(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate parent"):
        official.summarize_attempts(tmp_path, "run", [attempt("task"), attempt("task")], {"task"})


@pytest.mark.parametrize("body", ["{", "[]", '{}', '{"task": {"resolved": "false"}}'])
def test_malformed_reports_are_not_counted_as_failures_or_skipped(tmp_path: Path, body: str) -> None:
    record = attempt("task")
    path = write_report(tmp_path, record, False)
    path.write_text(body)
    with pytest.raises(ValueError, match="Invalid official report"):
        official.summarize_attempts(tmp_path, "run", [record], {"task"})


def args(**changes: object) -> SimpleNamespace:
    return SimpleNamespace(**{**dict(batches=None, phase="evaluate", max_workers=32,
                                    prepare_workers=4, startup_workers=4, retry_workers=8, retry_rounds=2,
                                    cleanup_timeout=1, docker_client_timeout=600,
                                    docker_max_pool_size=128, namespace="swebench", timeout=1800,
                                    cache_level="instance", clean=False, stop_on_error=False),
                              **changes})


def test_retry_runs_only_missing_reports_at_lower_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [attempt("done"), attempt("pending1"), attempt("pending2")]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(record["prediction"]) + "\n" for record in records))
    write_report(tmp_path, records[0], False)
    calls = []
    monkeypatch.setattr(official, "cleanup_run_containers", lambda *args: None)

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        submitted = Path(command[command.index("--predictions_path") + 1])
        ids = [json.loads(line)["instance_id"] for line in submitted.read_text().splitlines()]
        preparing = "--prepare-only" in command
        calls.append((preparing, ids, command[command.index("--max_workers") + 1]))
        if not preparing:
            write_report(tmp_path, next(record for record in records if record["task"] == ids[0]), True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", run)
    official.run_batches(tmp_path, "run", "dataset", "test", {"parent": str(predictions)}, args())
    assert calls == [(True, ["pending1", "pending2"], "32"),
                     (False, ["pending1", "pending2"], "32"),
                     (True, ["pending2"], "8"), (False, ["pending2"], "8")]
    assert len((tmp_path / "official/retry-runs.jsonl").read_text().splitlines()) == 2


def test_failed_image_preparation_never_launches_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(attempt("pending")["prediction"]) + "\n")
    calls = []
    monkeypatch.setattr(official, "cleanup_run_containers", lambda *args: None)

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(official.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="incomplete after bounded retries"):
        official.run_batches(tmp_path, "run", "dataset", "test", {"parent": str(predictions)},
                             args(retry_rounds=1))
    assert len(calls) == 2
    assert all("--prepare-only" in command for command in calls)


def test_successful_process_without_reports_is_still_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(attempt("pending")["prediction"]) + "\n")
    monkeypatch.setattr(official, "cleanup_run_containers", lambda *args: None)
    monkeypatch.setattr(official.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="incomplete after bounded retries"):
        official.run_batches(tmp_path, "run", "dataset", "test", {"parent": str(predictions)},
                             args(retry_rounds=0))


def test_image_preparation_reuses_cache_and_records_failures(tmp_path: Path) -> None:
    class MissingImage(Exception):
        pass

    cached = {"cached"}
    pulls = []

    def get(name: str) -> str:
        if name not in cached:
            raise MissingImage(name)
        return name

    def pull(name: str, **kwargs: object) -> None:
        pulls.append(name)
        if name == "broken":
            raise TimeoutError("registry timed out")
        cached.add(name)

    specs = [SimpleNamespace(instance_id=name, instance_image_key=name,
                             platform="linux/amd64", is_remote_image=True)
             for name in ["cached", "new", "broken"]]
    client = SimpleNamespace(images=SimpleNamespace(get=get, pull=pull))
    output = tmp_path / "images.jsonl"
    assert not prepare_images(specs, client, None, MissingImage, 2, output)
    assert sorted(pulls) == ["broken", "new"]
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["ok"] for row in results] == [True, True, False]
    assert "TimeoutError" in results[-1]["error"]


def test_cleanup_does_not_touch_other_runs_and_has_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        if command[1] == "ps":
            return SimpleNamespace(stdout='{"ID":"own","Names":"sweb.eval.task.run-parent","Status":"Exited"}\n'
                                   '{"ID":"other","Names":"sweb.eval.task.run-parent-extra","Status":"Exited"}\n')
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(official.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        official.cleanup_run_containers("run-parent", timeout=1)
    assert calls[-1][0] == ["docker", "rm", "-f", "own"]
    assert all(options["timeout"] == 1 for command, options in calls)


def test_container_creation_limit_releases_slots_after_failure() -> None:
    lock = Lock()
    saturated = Event()
    release = Event()
    active = 0
    peak = 0

    def build(number: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                saturated.set()
        try:
            assert release.wait(5)
            if number == 0:
                raise RuntimeError("Docker failed")
            return number
        finally:
            with lock:
                active -= 1

    limited = limit_container_creation(build, 2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(limited, number) for number in range(8)]
        try:
            assert saturated.wait(5)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="Docker failed"):
            futures[0].result()
        assert [future.result() for future in futures[1:]] == list(range(1, 8))
    assert peak == 2


def test_runner_wires_creation_limit_into_official_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    original = lambda *args, **kwargs: "container"
    docker_build = SimpleNamespace(build_container=original)
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(from_env=lambda **kwargs: None))
    monkeypatch.setitem(sys.modules, "swebench.harness", SimpleNamespace(docker_build=docker_build))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["runner", "--startup-workers", "2", "--max_workers", "32"])
    invoked = []

    def run_module(name: str, **kwargs: object) -> None:
        assert docker_build.build_container is not original
        assert docker_build.build_container() == "container"
        assert sys.argv == ["runner", "--max_workers", "32"]
        invoked.append(name)

    monkeypatch.setattr(runner.runpy, "run_module", run_module)
    assert runner.main() == 0
    assert invoked == ["swebench.harness.run_evaluation"]
