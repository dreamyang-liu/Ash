import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from swebench.fork_eval import Grade
from swebench.verifier_logs import VerifierLogSession


class LocalSession:
    sandbox_id = "local-test"

    def execute(self, tool_name, args, **kwargs):
        assert tool_name == "shell"
        process = subprocess.Popen(args["command"], shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=args.get("timeout", 30))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        output = json.dumps({"stdout": stdout.decode()[:64], "stderr": stderr.decode()[:64],
                             "exit_code": process.returncode, "timed_out": timed_out, "running": False})
        return SimpleNamespace(success=process.returncode == 0, output=output, error=None)

    def download_file(self, source, destination):
        shutil.copyfile(source, destination)
        return True


def test_full_separate_streams_and_nonzero_exit(tmp_path):
    wrapped = VerifierLogSession(LocalSession(), tmp_path / "host", str(tmp_path))
    result = wrapped.execute("shell", {"command": "printf out; printf err >&2; exit 7"})
    assert json.loads(result.output)["exit_code"] == 7
    grade = Grade()
    wrapped.finish(grade)
    assert grade.verifier_artifact_error is None
    with tarfile.open(Path(grade.verifier_artifacts) / "verifier-logs.tar.gz") as archive:
        assert archive.extractfile("./0001.stdout").read() == b"out"
        assert archive.extractfile("./0001.stderr").read() == b"err"


def test_large_output_is_not_limited_by_transport(tmp_path):
    wrapped = VerifierLogSession(LocalSession(), tmp_path / "host", str(tmp_path))
    wrapped.execute("shell", {"command": "python3 -c 'print(\"x\" * 2000000)'"})
    grade = Grade()
    wrapped.finish(grade)
    with tarfile.open(Path(grade.verifier_artifacts) / "verifier-logs.tar.gz") as archive:
        assert archive.extractfile("./0001.stdout").read() == b"x" * 2000000 + b"\n"


def test_timeout_keeps_partial_logs(tmp_path):
    wrapped = VerifierLogSession(LocalSession(), tmp_path / "host", str(tmp_path))
    result = wrapped.execute("shell", {"command": "printf before-timeout; sleep 30", "timeout": 0.2})
    assert json.loads(result.output)["timed_out"]
    grade = Grade(error="timeout")
    wrapped.finish(grade)
    with tarfile.open(Path(grade.verifier_artifacts) / "verifier-logs.tar.gz") as archive:
        assert archive.extractfile("./0001.stdout").read() == b"before-timeout"
    assert grade.error == "timeout"


def test_export_failure_does_not_change_grade(tmp_path, monkeypatch):
    session = LocalSession()
    wrapped = VerifierLogSession(session, tmp_path / "host", str(tmp_path))
    wrapped.execute("shell", {"command": "echo output"})
    monkeypatch.setattr(session, "download_file", lambda *args: False)
    grade = Grade(resolved=True)
    wrapped.finish(grade)
    assert grade.resolved
    assert grade.verifier_artifact_error == "archive download failed"


def test_grade_uses_recording_and_full_two_phase_lease(tmp_path, monkeypatch):
    import harness.execution.session as session_module
    import swebench.fork_eval as fork_eval

    observed = {}

    class Session:
        def __init__(self, **kwargs):
            observed["backend"] = kwargs["backend"]

        def create(self, image):
            return True

        def execute(self, *args, **kwargs):
            return SimpleNamespace(success=True, output="", error=None)

        def destroy(self):
            observed["destroyed"] = True

    class Recorder:
        def __init__(self, session, root):
            self.session = session
            observed["root"] = root

        def execute(self, *args, **kwargs):
            return self.session.execute(*args, **kwargs)

        def finish(self, grade):
            assert not observed.get("destroyed")
            observed["retained"] = True

    monkeypatch.setattr(session_module, "SandboxSession", Session)
    monkeypatch.setattr("swebench.verifier_logs.VerifierLogSession", Recorder)
    monkeypatch.setattr(fork_eval, "_install_runner", lambda *args: None)
    monkeypatch.setattr(fork_eval, "_run_tests", lambda *args: SimpleNamespace(success=True, output="", error=None))
    fork_eval.grade_snapshot("snap", {"repo": "astropy/astropy", "f2p": ["test_one"],
                                     "p2p": ["test_two"], "verifier_artifacts_dir": str(tmp_path)},
                            {"backend": "microvm", "microvm": {"sandbox_ttl": 2400}})
    assert observed["backend"]["microvm"]["sandbox_ttl"] == 4800
    assert observed["retained"] and observed["destroyed"]
