"""Collect a snapshot's diff and run official Pro tests in a fresh microVM."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any
from uuid import uuid4

from harness.execution.session import SandboxSession
from harness.execution.backends import with_sandbox_budget
from swebench.fork_eval import Grade
from swebench_pro.tasks import Task


def shell(session: Any, command: str, timeout: int = 120) -> tuple[str, str, int]:
    result = session.execute("shell", {"command": command, "timeout": timeout}, timeout=timeout + 60)
    outcome = getattr(result, "outcome", None)
    if outcome is not None:
        if outcome.running or outcome.timed_out or outcome.exit_code is None or outcome.truncated:
            raise RuntimeError("Pro command incomplete, timed out or truncated")
        return outcome.stdout, outcome.stderr, outcome.exit_code
    try:
        payload = json.loads(result.output)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and "stdout" in payload:
        if (payload.get("running") or payload.get("timed_out") or payload.get("stdout_truncated")
                or payload.get("stderr_truncated") or payload.get("exit_code") is None):
            raise RuntimeError("Pro command incomplete, timed out or truncated")
        return str(payload["stdout"]), str(payload.get("stderr") or ""), int(payload["exit_code"])
    if not result.success:
        raise RuntimeError(result.error or result.output or "Pro tool execution failed")
    return result.output, "", 0


def checked(session: Any, command: str, timeout: int = 120) -> str:
    stdout, stderr, code = shell(session, command, timeout)
    if code != 0:
        raise RuntimeError(f"Pro setup/collection exited {code}: {(stderr or stdout)[-2000:]}")
    return stdout


def reset_base(session: Any, task: Task) -> None:
    checked(session, f"cd /app && git reset --hard {task.base_commit} && git checkout {task.base_commit}")


def prepare_image(task: Task, backend: dict, resources: dict, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    session = SandboxSession(quiet=True, backend=dict(backend))
    metadata = {**task.provenance, "instance_id": task.instance_id, "resources": resources}
    try:
        if not session.create(task.image, resources):
            raise RuntimeError(f"Could not prepare Pro actor image: {session.create_error}")
        reset_base(session, task)
        snapshot = session.snapshot(name=f"pro-base-{uuid4().hex}", disk_only=True)
        if snapshot is None:
            raise RuntimeError("Could not checkpoint prepared Pro actor image")
        metadata["snapshot_id"] = snapshot.id
        return snapshot.id
    except Exception as exc:
        metadata["error"] = str(exc)
        raise
    finally:
        try:
            session.destroy()
        finally:
            (directory / f"{uuid4().hex}.json").write_text(json.dumps(metadata, indent=2))


def collect_patch(session: Any, baseline: list[str], task: Task, directory: Path) -> str:
    config = directory / "collect.json"
    config.write_text(json.dumps({"base": task.base_commit, "baseline": baseline}))
    git_directory = checked(session, "cd /app && git rev-parse --absolute-git-dir").strip()
    if not git_directory.startswith("/"):
        raise RuntimeError("Pro collector did not resolve an absolute Git directory")
    guest = f"{git_directory}/ash-pro-collect-{uuid4().hex}"
    if not session.upload_file(config, guest + ".json"):
        raise RuntimeError("Could not upload Pro collection baseline")
    script = """import json, os, subprocess, sys, tempfile
config = json.load(open(sys.argv[1]))
os.chdir('/app')
with tempfile.TemporaryDirectory(dir=os.path.dirname(sys.argv[1])) as temporary:
    environment = dict(os.environ, GIT_INDEX_FILE=temporary + '/index')
    def git(*arguments):
        return subprocess.check_output(['git', *arguments], env=environment)
    git('read-tree', config['base'])
    git('add', '-u')
    baseline = set(config['baseline'])
    paths = git('ls-files', '--others', '--exclude-standard', '-z').decode('utf-8', 'surrogateescape').split('\\0')
    for path in paths:
        if path and path not in baseline:
            git('add', '--', path)
    with open(sys.argv[2], 'wb') as output:
        output.write(git('diff', '--cached', '--binary', config['base'], '--'))
"""
    checked(session, f"python - {shlex.quote(guest + '.json')} {shlex.quote(guest + '.patch')} <<'PY'\n{script}\nPY")
    patch_path = directory / "model.patch"
    if not session.download_file(guest + ".patch", patch_path):
        raise RuntimeError("Could not download Pro patch")
    return patch_path.read_bytes().decode("utf-8", "surrogateescape")


def assemble(task: Task, patch: str, directory: Path) -> str:
    request = directory / "request.json"
    request.write_text(json.dumps({"sample": task.sample, "patch": patch}))
    command = [sys.executable, str(Path(__file__).with_name("official.py")),
               str(task.harness_repo), str(request), str(directory)]
    with (directory / "assembler.log").open("w") as log:
        subprocess.run(command, cwd=task.harness_repo, check=True, timeout=120,
                       stdout=log, stderr=subprocess.STDOUT)
    return (directory / "entryscript.sh").read_text()


def guarded_entryscript(original: str, patch: str) -> str:
    apply_command = "git apply -v /workspace/patch.diff"
    test_commands = [line for line in original.splitlines() if line.startswith("bash /workspace/run_script.sh ")]
    if original.splitlines().count(apply_command) != 1 or len(test_commands) != 1:
        raise ValueError("Unexpected official Pro entryscript structure")
    apply_guard = (apply_command + " || { echo apply_failed > /workspace/ash-stage; exit 41; }") if patch.strip() else ":"
    test_command = test_commands[0]
    lines = []
    environment_prelude = True
    for line in original.splitlines():
        if line.strip() in ("# apply patch", "cd /app"):
            environment_prelude = False
        legacy_export = re.fullmatch(r"export\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+)", line)
        if environment_prelude and legacy_export:
            name, value = legacy_export.groups()
            value = " ".join(shlex.split(value))
            line = f"export {name}={shlex.quote(value)}"
        lines.append(line)
    guarded = "\n".join(lines).replace(apply_command, apply_guard)
    guarded = guarded.replace(test_command, test_command + " || test_exit=$?\n"
                              "printf '%s\\n' \"${test_exit:-0}\" > /workspace/test_exit_code.txt")
    return "set -e\n" + guarded


def grade_output(patch: str, output: Any, task: Task) -> Grade:
    grade = Grade(patch=patch)
    tests = output.get("tests") if isinstance(output, dict) else None
    if not isinstance(tests, list) or any(
        not isinstance(test, dict) or not isinstance(test.get("name"), str)
        or not isinstance(test.get("status"), str) for test in tests
    ):
        grade.error = "Malformed official Pro parser output"
        return grade
    passed = {test["name"] for test in tests if test["status"] == "PASSED"}
    failed_f2p = sorted(set(task.f2p) - passed)
    failed_p2p = sorted(set(task.p2p) - passed)
    grade.resolved = not failed_f2p and not failed_p2p
    grade.f2p_pass = not failed_f2p
    grade.p2p_ran = True
    grade.p2p_pass = not failed_p2p
    grade.broken = failed_p2p
    grade.detail = json.dumps({"f2p_passed": len(set(task.f2p) & passed), "f2p_total": len(set(task.f2p)),
                               "p2p_passed": len(set(task.p2p) & passed), "p2p_total": len(set(task.p2p)),
                               "failed_f2p": failed_f2p, "failed_p2p": failed_p2p}, ensure_ascii=False)
    return grade


def verify_in_session(session: Any, task: Task, patch: str, directory: Path, timeout: int) -> Grade:
    original = assemble(task, patch, directory)
    effective_patch = (directory / "patch.diff").read_bytes().decode("utf-8", "surrogateescape")
    guarded = directory / "guarded-entryscript.sh"
    guarded.write_text(guarded_entryscript(original, effective_patch))
    checked(session, "mkdir -p /workspace && rm -f /workspace/output.json /workspace/ash-stage")
    for name in ("patch.diff", "run_script.sh", "parser.py", "guarded-entryscript.sh"):
        if not session.upload_file(directory / name, f"/workspace/{name}"):
            raise RuntimeError(f"Could not upload verifier file {name}")
    stdout, stderr, code = shell(
        session, "bash /workspace/guarded-entryscript.sh > /workspace/entry.stdout 2> /workspace/entry.stderr", timeout)
    if code == 41:
        stage, stderr, stage_code = shell(session, "cat /workspace/ash-stage")
        if stage_code == 0 and stage.strip() == "apply_failed":
            return Grade(patch=patch, detail="model.patch did not apply to the pristine Pro base")
    if code != 0:
        return Grade(patch=patch, error=f"Pro verifier setup/parser exited {code}; see retained workspace logs")
    output_path = directory / "output.json"
    if not session.download_file("/workspace/output.json", output_path):
        return Grade(patch=patch, error="Pro verifier produced no downloadable output.json")
    return grade_output(patch, json.loads(output_path.read_text()), task)


def retain(session: Any, directory: Path) -> str | None:
    try:
        archive = f"/tmp/ash-pro-logs-{uuid4().hex}.tar.gz"
        checked(session, f"tar -czf {archive} -C /workspace .")
        if not session.download_file(archive, directory / "verifier-logs.tar.gz"):
            return "Pro verifier archive download failed"
    except Exception as exc:
        return f"Pro verifier archive failed: {exc}"
    return None


def grade_snapshot(snapshot_id: str, task: Task, backend: dict, *,
                   resources: dict, timeout: int = 3600,
                   artifacts_dir: str | Path | None = None,
                   collector_backend: dict | None = None,
                   session_factory=None) -> Grade:
    root = Path(artifacts_dir or "runs/pro-verifier")
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="verify-", dir=root)).resolve()
    grade = Grade(verifier_artifacts=str(directory))
    verifier_backend = with_sandbox_budget(backend, timeout + 600)
    factory = session_factory or SandboxSession
    verifier = factory(quiet=True, backend=verifier_backend)
    collector_config = dict(backend if collector_backend is None else collector_backend)
    collector = factory(quiet=True, backend=collector_config)
    verifier_ready = False
    metadata = {**task.provenance, "snapshot_id": snapshot_id, "resources": resources,
                "timeout": timeout, "runtime_port": backend.get("microvm", {}).get("runtime_port", 3000),
                "collector_runtime_port": collector_config.get("microvm", {}).get("runtime_port", 3000),
                "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        if not verifier.create(task.image, resources):
            raise RuntimeError(f"Could not create Pro verifier: {verifier.create_error}")
        verifier_ready = True
        checked(verifier, "mkdir -p /workspace")
        reset_base(verifier, task)
        checked(verifier, "cd /app && git ls-files --others --exclude-standard -z > /workspace/baseline.paths")
        baseline_file = directory / "baseline.paths"
        if not verifier.download_file("/workspace/baseline.paths", baseline_file):
            raise RuntimeError("Could not download pristine Pro baseline")
        baseline = [path for path in baseline_file.read_bytes().decode("utf-8", "surrogateescape").split("\0") if path]
        try:
            if not collector.create(snapshot_id, resources):
                raise RuntimeError(f"Could not restore Pro attempt: {collector.create_error}")
            grade.patch = collect_patch(collector, baseline, task, directory)
        finally:
            collector.destroy()
        grade = verify_in_session(verifier, task, grade.patch, directory, timeout)
    except Exception as exc:
        grade.error = f"{type(exc).__name__}: {exc}"
    finally:
        grade.verifier_artifacts = str(directory)
        if verifier_ready:
            grade.verifier_artifact_error = retain(verifier, directory)
        try:
            verifier.destroy()
        except Exception as exc:
            grade.error = f"{grade.error or ''}; verifier cleanup failed: {exc}".strip("; ")
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(),
                        error=grade.error, artifact_error=grade.verifier_artifact_error,
                        resolved=grade.resolved)
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (directory / "grade.json").write_text(json.dumps(grade.__dict__, indent=2))
    return grade
