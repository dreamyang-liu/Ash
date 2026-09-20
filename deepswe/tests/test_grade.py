"""reward.json -> Grade, and the two-VM choreography against a fake session."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import deepswe.grade as grade_mod
from deepswe.grade import (VerifierOutcome, grade_from_verifier, grade_snapshot,
                           shell_parts)
from deepswe.tests.test_tasks import make_task
from deepswe.tasks import load_task


def result(output="", success=True, error=None):
    return SimpleNamespace(output=output, success=success, error=error)


def envelope(stdout="", stderr="", exit_code=0):
    return result(json.dumps({"stdout": stdout, "stderr": stderr,
                              "exit_code": exit_code}))


# --- shell_parts -------------------------------------------------------------

def test_shell_parts_reads_the_envelope_or_bare_stdout():
    assert shell_parts(result("plain\n")) == ("plain\n", "", 0)
    assert shell_parts(envelope("out", "err", 3)) == ("out", "err", 3)
    assert shell_parts(result("", success=False, error="boom")) == ("", "boom", 1)
    # A file whose CONTENT is JSON without a stdout key is still bare stdout.
    assert shell_parts(result('{"reward": 1}')) == ('{"reward": 1}', "", 0)


# --- grade_from_verifier ----------------------------------------------------

def reward(**kw):
    base = {"reward": 0, "f2p_total": 3, "f2p_passed": 0, "p2p_total": 2,
            "p2p_passed": 2, "f2p": 0.0, "p2p": 1.0, "partial": 0.4}
    base.update(kw)
    return base


def test_reward_one_is_resolved():
    g = grade_from_verifier("diff", VerifierOutcome(reward=reward(
        reward=1, f2p_passed=3)), {"uncommitted_paths": 0, "branch": "feat",
                                   "commits_since_base": 1})
    assert g.resolved and g.f2p_pass and g.p2p_pass and g.p2p_ran
    assert g.error is None and g.patch == "diff"
    assert "reward.json" in g.detail and "UNCOMMITTED" not in g.detail


def test_target_pass_with_regression_is_score_two_shape():
    g = grade_from_verifier("diff", VerifierOutcome(
        reward=reward(f2p_passed=3, p2p_passed=1),
        failed_tests=["p2p:pkg.TestOld"]))
    assert not g.resolved and g.f2p_pass and not g.p2p_pass
    assert g.broken == ["p2p:pkg.TestOld"]


def test_apply_failed_is_a_failed_attempt_not_a_grading_error():
    g = grade_from_verifier("garbage", VerifierOutcome(reward=reward(apply_failed=1)))
    assert g.error is None and not g.resolved and not g.p2p_ran
    assert "did not apply" in g.detail


def test_uncommitted_work_is_called_out():
    g = grade_from_verifier("", VerifierOutcome(reward=reward()),
                            {"uncommitted_paths": 7, "branch": "develop",
                             "commits_since_base": 0})
    assert "7 uncommitted" in g.detail and "UNCOMMITTED WORK IS NOT GRADED" in g.detail
    assert "empty model.patch" in g.detail


def test_no_reward_json_is_a_grading_error():
    g = grade_from_verifier("diff", VerifierOutcome(error="verifier produced no reward.json",
                                                    output="crash log"))
    assert g.error and "reward.json" in g.error
    assert "crash log" in g.detail and not g.resolved


# --- grade_snapshot against a fake session -----------------------------------

class FakeSession:
    """Records what grade_snapshot asks of each VM; answers from a script."""
    instances: list = []

    def __init__(self, quiet=True, backend=None):
        self.backend = backend
        self.created = None
        self.resources = None
        self.commands: list = []
        self.uploads: list = []
        self.destroyed = False
        self.create_error = ""
        FakeSession.instances.append(self)

    def create(self, image, resources=None):
        self.created, self.resources = image, resources
        return True

    def destroy(self):
        self.destroyed = True

    def upload_file(self, source, destination):
        self.uploads.append((Path(source).read_text(), destination))
        return True

    def download_file(self, source, destination):
        assert not self.destroyed
        Path(destination).write_bytes(b"raw test log\x00" * 1000)
        return True

    def execute(self, tool, args, timeout=None):
        assert tool == "shell", tool
        cmd = args["command"]
        self.commands.append(cmd)
        if cmd.startswith("cat /logs/artifacts/model.patch"):
            return result("diff --git a/x b/x\n+new\n")
        if cmd.startswith("cd /app && git status --porcelain"):
            return result("0\nfeature/x\nabc123\n1\n")
        if cmd.startswith("cat /logs/verifier/reward.json"):
            return result(json.dumps(reward(reward=1, f2p_passed=3)))
        if cmd.startswith("cat /logs/verifier/ctrf.json"):
            return result(json.dumps({"results": {"tests": [
                {"name": "pkg.TestA", "status": "passed", "suite": "f2p"}]}}))
        if cmd.startswith("cat "):
            return envelope("", "no such file", 1)
        return envelope("ok")


def test_grade_snapshot_collects_in_the_snapshot_and_verifies_on_a_pristine_vm(
        tmp_path, monkeypatch):
    FakeSession.instances = []
    monkeypatch.setattr(grade_mod, "SandboxSession", FakeSession)
    task = load_task(make_task(tmp_path))
    backend = {"backend": "microvm", "microvm": {"allow_internet": False}}

    grade = grade_snapshot("snap-123", task, backend, artifacts_dir=tmp_path / "logs")

    collect, verify = FakeSession.instances
    # 1. the snapshot: their collect command, then the patch and the repo state
    assert collect.created == "snap-123" and collect.destroyed
    assert collect.commands[0] == task.collect[0].command
    assert any(c.startswith("cat /logs/artifacts/model.patch") for c in collect.commands)
    # 2. a fresh VM from the TASK IMAGE with the task's shape, same (offline) backend
    assert verify.created == task.image
    assert verify.resources == {"cpu": 2, "memory_mb": 8192}
    assert verify.backend == backend and verify.destroyed
    # tests/Dockerfile replayed: every COPY placed where it says, RUN executed
    placed = {dst: text for text, dst in verify.uploads}
    assert set(placed) == {"/tests/test.sh", "/tests/test.patch", "/tests/grader.py",
                           "/tests/config.json", "/logs/artifacts/model.patch"}
    assert placed["/logs/artifacts/model.patch"] == "diff --git a/x b/x\n+new\n"
    assert placed["/tests/grader.py"] == "# grader.py\n"
    assert "chmod +x /tests/test.sh" in verify.commands
    assert "bash /tests/test.sh" in verify.commands
    # the RUN step comes before the entrypoint, the entrypoint before reading reward
    order = verify.commands
    assert order.index("chmod +x /tests/test.sh") < order.index("bash /tests/test.sh")
    assert order.index("bash /tests/test.sh") < next(
        i for i, c in enumerate(order) if c.startswith("cat /logs/verifier/reward.json"))
    # 3. the verdict is theirs
    assert grade.resolved and grade.patch.startswith("diff --git")
    assert "branch=feature/x" in grade.detail
    assert (Path(grade.verifier_artifacts) / "verifier-logs.tar.gz").read_bytes() == b"raw test log\x00" * 1000


def test_grade_snapshot_reports_a_snapshot_that_will_not_restore(monkeypatch, tmp_path):
    class Refusing(FakeSession):
        def create(self, image, resources=None):
            self.create_error = "gone"
            return False
    FakeSession.instances = []
    monkeypatch.setattr(grade_mod, "SandboxSession", Refusing)
    task = load_task(make_task(tmp_path))
    grade = grade_snapshot("snap-x", task, {})
    assert grade.error and "snap-x" in grade.error and "gone" in grade.error


def test_verifier_uses_its_own_declared_resources(monkeypatch, tmp_path):
    from dataclasses import replace
    FakeSession.instances = []
    monkeypatch.setattr(grade_mod, "SandboxSession", FakeSession)
    task = replace(load_task(make_task(tmp_path)), verifier_cpus=4, verifier_memory_mb=16384)
    grade_snapshot("snap-x", task, {}, artifacts_dir=tmp_path / "logs")
    assert FakeSession.instances[-1].resources == {"cpu": 4, "memory_mb": 16384}


@pytest.mark.parametrize("failure", [None, "timeout", "transport", "setup", "invalid_reward", "interrupt"])
def test_verifier_exports_raw_logs_before_cleanup_on_every_exit(tmp_path, monkeypatch, failure):
    events = []

    class LoggedSession(FakeSession):
        def execute(self, tool, args, timeout=None):
            command = args["command"]
            if command == "bash /tests/test.sh":
                if failure == "interrupt":
                    raise KeyboardInterrupt()
                if failure == "transport":
                    raise TimeoutError("transport lost")
                response = envelope("base mode rc=0", exit_code=137 if failure == "timeout" else 0)
                response.outcome = SimpleNamespace(timed_out=failure == "timeout", running=False)
                return response
            if failure == "setup" and command.startswith("chmod "):
                return envelope(stderr="chmod failed", exit_code=1)
            if command == "cat /logs/verifier/reward.json":
                if failure == "timeout":
                    return envelope(stderr="missing", exit_code=1)
                if failure == "invalid_reward":
                    return result("not json")
            return super().execute(tool, args, timeout)

        def download_file(self, source, destination):
            events.append("export")
            return super().download_file(source, destination)

        def destroy(self):
            events.append("destroy")
            assert events[-2] == "export"
            super().destroy()

    monkeypatch.setattr(grade_mod, "SandboxSession", LoggedSession)
    task = load_task(make_task(tmp_path))
    artifact_root = tmp_path / "artifacts"
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            grade_mod.verify_patch(task, "diff", {}, artifacts_dir=artifact_root)
    else:
        outcome = grade_mod.verify_patch(task, "diff", {}, artifacts_dir=artifact_root)
        assert bool(outcome.error) == (failure is not None)
        assert outcome.artifact_error is None
        grade = grade_from_verifier("diff", outcome)
        assert grade.verifier_artifacts == outcome.artifacts_dir
        assert grade.verifier_artifact_error is None
    directory = next(artifact_root.iterdir())
    assert (directory / "verifier-logs.tar.gz").read_bytes() == b"raw test log\x00" * 1000
    metadata = json.loads((directory / "metadata.json").read_text())
    assert bool(metadata["grading_error"]) == (failure is not None)
    if failure == "timeout":
        assert metadata["timed_out"] is True and metadata["exit_code"] == 137
        assert (directory / "stdout.txt").read_text() == "base mode rc=0"
    assert events == ["export", "destroy"]


def test_failed_log_export_is_visible_without_changing_reward_or_leaking_vm(tmp_path, monkeypatch, capsys):
    class BrokenDownload(FakeSession):
        def download_file(self, source, destination):
            raise OSError("disk full")

    FakeSession.instances = []
    monkeypatch.setattr(grade_mod, "SandboxSession", BrokenDownload)
    task = load_task(make_task(tmp_path))
    outcome = grade_mod.verify_patch(task, "diff", {}, artifacts_dir=tmp_path / "logs")
    grade = grade_from_verifier("diff", outcome)
    assert grade.resolved and grade.error is None
    assert "disk full" in grade.verifier_artifact_error
    metadata = json.loads((Path(grade.verifier_artifacts) / "metadata.json").read_text())
    assert "disk full" in metadata["artifact_error"]
    assert metadata["archive"] is None
    assert FakeSession.instances[-1].destroyed
    assert "Verifier log retention failed" in capsys.readouterr().err


def test_repeated_verification_does_not_overwrite_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(grade_mod, "SandboxSession", FakeSession)
    task = load_task(make_task(tmp_path))
    first = grade_mod.verify_patch(task, "first", {}, artifacts_dir=tmp_path / "logs")
    second = grade_mod.verify_patch(task, "second", {}, artifacts_dir=tmp_path / "logs")
    assert first.artifacts_dir != second.artifacts_dir
    assert (Path(first.artifacts_dir) / "metadata.json").exists()
    assert (Path(second.artifacts_dir) / "metadata.json").exists()
