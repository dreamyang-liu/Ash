"""Queued DeepSWE grading uses frozen assets and worker-owned VM sessions."""

from dataclasses import asdict
import hashlib
import json
import subprocess

import pytest

from deepswe.tests.test_grade import FakeSession, result, reward
from deepswe.tests.test_tasks import IMAGE, make_task
from runstore.deepswe import task_manifest
from runstore.grading import grade
from runstore.specs import GradeSpec, JobSpec


@pytest.fixture
def frozen_task(tmp_path):
    repo = tmp_path / "dataset-repo"
    task = make_task(repo / "tasks")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Test",
        "-c", "user.email=test@example.invalid", "commit", "-qm", "task",
    ], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([{
        "instance_id": "demo-task", "task_files_sha256": task_manifest(task),
    }]))
    spec = GradeSpec(
        benchmark="deepswe", instance_id="demo-task", snapshot_id="actor-snapshot",
        dataset_path=str(dataset), dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        grader_revision=revision, harness_repo=str(repo),
        resources={"cpu": 2, "memory_mb": 8192},
    )
    return task, asdict(spec)


@pytest.mark.parametrize("resolved", [True, False])
def test_queued_grader_tracks_collector_and_verifier_and_returns_real_verdict(tmp_path, frozen_task, resolved):
    _, spec = frozen_task
    FakeSession.instances = []

    class Session(FakeSession):
        def execute(self, tool, args, timeout=None):
            if args["command"].startswith("cat /logs/verifier/reward.json"):
                return result(json.dumps(reward(reward=int(resolved), f2p_passed=3 if resolved else 0)))
            return super().execute(tool, args, timeout)

    JobSpec(kind="grade", spec=spec).validate()
    verdict = grade(spec, tmp_path / "attempt", session_factory=Session)
    assert verdict["status"] == "completed"
    assert verdict["resolved"] is resolved
    assert verdict["failure_kind"] is None
    assert [session.created for session in FakeSession.instances] == ["actor-snapshot", IMAGE]
    assert all(session.destroyed for session in FakeSession.instances)
    assert all(session.backend["microvm"]["allow_internet"] is False for session in FakeSession.instances)
    assert all(session.backend["microvm"]["sandbox_ttl"] >= spec["timeout_s"] + 600
               for session in FakeSession.instances)
    assert FakeSession.instances[1].resources == {"cpu": 2, "memory_mb": 8192}


def test_missing_verifier_output_does_not_become_zero_reward(tmp_path, frozen_task):
    _, spec = frozen_task

    class Session(FakeSession):
        def execute(self, tool, args, timeout=None):
            if args["command"].startswith("cat /logs/verifier/reward.json"):
                return result("", success=False, error="missing reward")
            return super().execute(tool, args, timeout)

    verdict = grade(spec, tmp_path / "attempt", session_factory=Session)
    assert verdict["status"] == "error"
    assert verdict["resolved"] is None
    assert verdict["failure_kind"] == "infrastructure"


@pytest.mark.parametrize("change", ["asset", "revision", "resources", "network"])
def test_changed_frozen_grading_inputs_fail_before_allocating_vm(tmp_path, frozen_task, change):
    task, spec = frozen_task
    if change == "asset":
        (task / "tests" / "grader.py").write_text("changed")
    elif change == "revision":
        spec["grader_revision"] = "wrong-revision"
    elif change == "resources":
        spec["resources"]["memory_mb"] = 4096
    else:
        spec["verifier_network"] = "allow"

    def forbidden_session(**kwargs):
        pytest.fail("Frozen grading validation must precede VM allocation")

    with pytest.raises(ValueError, match="DeepSWE"):
        grade(spec, tmp_path / "attempt", session_factory=forbidden_session)


def test_deepswe_requires_harness_repository(frozen_task):
    _, spec = frozen_task
    spec["harness_repo"] = None
    with pytest.raises(ValueError, match="harness_repo"):
        JobSpec(kind="grade", spec=spec).validate()
