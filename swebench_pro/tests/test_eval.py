from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from harness.core.result import CommandOutcome, ToolResult
from swebench import fork_eval
from swebench_pro import grade as grading
from swebench_pro.bench import SWEbenchPro
from swebench_pro.tasks import string_list, task_from_row


def make_task(root: Path, **changes: object):
    row = {"instance_id": "instance_demo-123", "repo": "org/demo", "base_commit": "a" * 40,
           "dockerhub_tag": "demo-123", "problem_statement": "Fix the behavior",
           "requirements": "Preserve ordering", "interface": "Function: solve",
           "before_repo_set_cmd": "git reset --hard " + "a" * 40 + "\ngit checkout gold -- test.py",
           "fail_to_pass": '["new"]', "pass_to_pass": '["old"]',
           "selected_test_files_to_run": '["test.py"]', "patch": "PRIVATE_GOLD", "test_patch": "PRIVATE_TEST"}
    row.update(changes)
    for relative in ["swe_bench_pro_eval.py", "helper_code/image_uri.py",
                     "run_scripts/instance_demo-123/run_script.sh", "run_scripts/instance_demo-123/parser.py",
                     "dockerfiles/base_dockerfile/instance_demo-123/Dockerfile",
                     "dockerfiles/instance_dockerfile/instance_demo-123/Dockerfile"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    return task_from_row(row, root, "fixture")


def options(root: Path, **changes: object) -> SimpleNamespace:
    return SimpleNamespace(**{**dict(pro_repo=str(root), benchmark="swebench-pro", runtime_bin="runtime/ash-runtime",
                                    timeout=10800, slot="claude-code", model="test", rounds=0,
                                    branch_count_mode="adaptive", parent_from=None), **changes})


def test_runtime_port_override_is_opt_in_and_preserves_legacy_collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_args = options(tmp_path)
    assert "runtime_port" not in fork_eval.backend_for(legacy_args, SWEbenchPro(legacy_args))["microvm"]
    args = options(tmp_path, pro_runtime_port=34122, pro_collector_runtime_port=3000)
    bench = SWEbenchPro(args)
    backend = fork_eval.backend_for(args, bench)
    assert backend["microvm"]["runtime_port"] == 34122
    captured = {}

    def grade(snapshot: str, task: object, verifier: dict, **kwargs: object) -> fork_eval.Grade:
        captured.update(verifier=verifier, **kwargs)
        return fork_eval.Grade()

    monkeypatch.setattr(grading, "grade_snapshot", grade)
    bench.grade("legacy-snapshot", {"task": object()}, backend)
    assert captured["verifier"]["microvm"]["runtime_port"] == 34122
    assert captured["collector_backend"]["microvm"]["runtime_port"] == 3000
    assert backend["microvm"]["runtime_port"] == 34122
    with pytest.raises(ValueError, match="runtime ports"):
        SWEbenchPro(options(tmp_path, pro_runtime_port=65536))


def test_prompt_includes_public_spec_but_no_verifier_or_reference_patch(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    bench = fork_eval.select_benchmark(options(tmp_path))
    instance = bench.instance(task)
    prompt = bench.prompt(instance)
    assert all(text in prompt for text in ["Fix the behavior", "Preserve ordering", "Function: solve", "/app"])
    assert all(text not in prompt for text in ["PRIVATE_GOLD", "PRIVATE_TEST", "git checkout gold", "/testbed"])
    assert "uncommitted" in prompt
    assert bench.branch_prompt(instance, "PRIVATE_VERDICT", "Inspect ordering.", truncated=True).startswith("Inspect ordering.")
    backend = fork_eval.backend_for(options(tmp_path), bench)
    assert backend["microvm"]["image_env"]
    assert "allow_internet" not in backend["microvm"]
    offline = SWEbenchPro(options(tmp_path, pro_block_network=True))
    assert fork_eval.backend_for(options(tmp_path), offline)["microvm"]["allow_internet"] is False


@pytest.mark.parametrize("value", ["__import__('os').system('false')", "null", "[1]", None])
def test_test_lists_are_parsed_without_eval(value: object) -> None:
    with pytest.raises(ValueError):
        string_list(value, "tests")


@pytest.mark.parametrize("changes", [{"base_commit": "HEAD;false"}, {"dockerhub_tag": "../../bad"},
                                     {"instance_id": "../../bad"}, {"fail_to_pass": "[]"}])
def test_invalid_dataset_fields_are_rejected(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        make_task(tmp_path, **changes)


@pytest.mark.parametrize("tests,resolved,f2p,p2p", [
    ([{"name": "new", "status": "PASSED"}, {"name": "old", "status": "PASSED"}], True, True, True),
    ([{"name": "new", "status": "PASSED"}], False, True, False),
    ([{"name": "old", "status": "PASSED"}], False, False, True),
    ([], False, False, False),
])
def test_grade_matches_official_set_inclusion(tmp_path: Path, tests: list, resolved: bool, f2p: bool, p2p: bool) -> None:
    grade = grading.grade_output("patch", {"tests": tests}, make_task(tmp_path))
    assert (grade.resolved, grade.f2p_pass, grade.p2p_pass) == (resolved, f2p, p2p)
    assert grade.error is None


@pytest.mark.parametrize("output", [None, {}, {"tests": [{}]}, {"tests": "PASSED"}])
def test_malformed_parser_output_is_unmeasured(tmp_path: Path, output: object) -> None:
    assert grading.grade_output("patch", output, make_task(tmp_path)).error


def test_guard_preserves_test_failures_for_parser_but_stops_setup_errors(tmp_path: Path) -> None:
    script = """cd /app
git reset --hard BASE
git checkout BASE
git apply -v /workspace/patch.diff
git checkout gold -- test.py
bash /workspace/run_script.sh test.py > /workspace/stdout.log 2> /workspace/stderr.log
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    guarded = grading.guarded_entryscript(script, "patch")
    assert guarded.startswith("set -e\n")
    assert "git apply -v /workspace/patch.diff || { echo apply_failed" in guarded
    assert "|| test_exit=$?" in guarded
    assert "git checkout gold -- test.py" in guarded
    assert "git apply" not in grading.guarded_entryscript(script, "")
    with pytest.raises(ValueError):
        grading.guarded_entryscript("echo unexpected", "patch")


def test_legacy_env_conversion_is_limited_to_official_prelude() -> None:
    original = ("export LANG en_US.UTF-8\nexport LC_ALL POSIX\n"
                "export PATH=/usr/bin:$PATH\n# apply patch\ncd /app\n"
                "git apply -v /workspace/patch.diff\nexport EXISTING ANOTHER\n"
                "bash /workspace/run_script.sh tests\n")
    guarded = grading.guarded_entryscript(original, "patch")
    assert "export LANG=en_US.UTF-8\n" in guarded
    assert "export LC_ALL=POSIX\n" in guarded
    assert "export PATH=/usr/bin:$PATH\n" in guarded
    assert "export EXISTING ANOTHER\n" in guarded


def test_official_assembler_preserves_non_utf8_patch_bytes(tmp_path: Path) -> None:
    task = make_task(tmp_path / "upstream")
    (task.harness_repo / "swe_bench_pro_eval.py").write_text(
        "def assemble_workspace_files(uid, scripts_dir, patch, sample):\n"
        "    return {'patch.diff': patch, 'entryscript.sh': 'echo ready'}, 'echo ready'\n")
    output = tmp_path / "output"
    output.mkdir()
    raw = b"diff --git a/legacy b/legacy\n+copyright \xa9\r\n"
    assert grading.assemble(task, raw.decode("utf-8", "surrogateescape"), output) == "echo ready"
    assert (output / "patch.diff").read_bytes() == raw


def test_actor_preparation_is_consumed_and_grading_errors_do_not_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = make_task(tmp_path / "repo")
    bench = SWEbenchPro(options(tmp_path))
    prepared = []
    seen = []
    monkeypatch.setattr(bench, "prepare_image", lambda *args: prepared.append(args) or "prepared-snapshot")

    def run_attempt(*args: object, **kwargs: object) -> SimpleNamespace:
        seen.append(kwargs)
        return SimpleNamespace(journal_path=tmp_path / "parent.jsonl", checkpoints=2)

    monkeypatch.setattr(fork_eval, "run_attempt", run_attempt)
    monkeypatch.setattr(fork_eval, "grade_attempt", lambda *args: fork_eval.Grade(error="Docker unavailable"))
    monkeypatch.setattr(fork_eval, "report", lambda *args: None)
    monkeypatch.setattr(fork_eval, "ask_analyst", lambda *args, **kwargs: pytest.fail("infra must not trigger branching"))
    attempts = fork_eval.run_one(None, options(tmp_path, rounds=2), task, [3], tmp_path / "out", bench)
    assert len(attempts) == 1 and attempts[0].grade.error
    assert len(prepared) == 1 and seen[0]["image"] == "prepared-snapshot"


def test_branch_run_keeps_snapshot_and_native_cut(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    bench = SWEbenchPro(options(tmp_path))
    specs = []
    orch = SimpleNamespace(run=lambda spec: specs.append(spec) or "outcome")
    result = fork_eval.run_attempt(orch, options(tmp_path), bench.instance(task), name="r1b1-fix",
                                  prompt="hint", image="branch-snapshot", out_dir=tmp_path,
                                  resume="native-session", resume_at="cut-uuid", fork=True, bench=bench)
    assert result == "outcome"
    assert specs[0].sandbox_image == "branch-snapshot"
    assert specs[0].resume_session_id == "native-session" and specs[0].fork
    assert specs[0].extra["resume_session_at"] == "cut-uuid"


def test_summary_does_not_publish_errors_as_a_final_score(tmp_path: Path) -> None:
    bench = SWEbenchPro(options(tmp_path))
    report = bench.summary([{"instance": "task", "resolved": False,
                             "attempts": [{"grading_error": "timeout"}]}])
    assert report["final_resolved_rate"] is None
    assert report["resolved_lower_bound"] == 0
    assert report["grading_error_ids"] == ["task"]


def test_summary_waits_for_the_entire_selected_cohort(tmp_path: Path) -> None:
    bench = SWEbenchPro(options(tmp_path))
    report = bench.summary([{"instance": "done", "resolved": True, "attempts": [{}]}],
                           expected_ids=["done", "pending"])
    assert not report["grading_complete"] and report["final_resolved_rate"] is None
    assert report["resolved_lower_bound"] == 0.5
    assert report["pending_task_ids"] == ["pending"]


@pytest.mark.parametrize("linked_worktree", [False, True])
def test_snapshot_collection_includes_committed_staged_and_untracked_without_image_baggage(tmp_path: Path, linked_worktree: bool) -> None:
    repo = tmp_path / "working"
    repo.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "tracked.txt").write_text("base\n")
    (repo / "legacy.txt").write_bytes(b"old copyright \xa9\r\n")
    git("add", "tracked.txt", "legacy.txt")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    if linked_worktree:
        git("worktree", "add", "--detach", str(tmp_path / "linked"), base)
        repo = tmp_path / "linked"
    (repo / "committed.txt").write_text("committed\n")
    git("add", "committed.txt")
    git("commit", "-qm", "agent")
    (repo / "tracked.txt").write_text("changed\n")
    (repo / "legacy.txt").write_bytes(b"new copyright \xa9\r\n")
    (repo / "staged.txt").write_text("staged\n")
    git("add", "staged.txt")
    (repo / "new 'file.txt").write_text("new\n")
    (repo / "baggage.txt").write_text("image baggage\n")
    index_path = Path(git("rev-parse", "--git-path", "index"))
    if not index_path.is_absolute():
        index_path = repo / index_path
    index_before = index_path.read_bytes()
    guest_tmp = tmp_path / "guest-tmp"
    guest_tmp.mkdir()
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(f"#!{sys.executable}\nimport subprocess,sys,shutil\n"
                    f"result=subprocess.run([{shutil.which('git')!r},*sys.argv[1:]])\n"
                    "if sys.argv[1:2] == ['read-tree']:\n"
                    f"    shutil.rmtree({str(guest_tmp)!r})\n"
                    f"    __import__('pathlib').Path({str(guest_tmp)!r}).mkdir()\n"
                    "raise SystemExit(result.returncode)\n")
    shim.chmod(0o755)
    task = make_task(tmp_path / "assets", base_commit=base)
    directory = tmp_path / "artifacts"
    directory.mkdir()

    class LocalSession:
        def mapped(self, remote: str) -> Path:
            if remote.startswith("/tmp/ash-pro-collect-"):
                return guest_tmp / Path(remote).name
            return Path(remote)

        def upload_file(self, source: Path, destination: str) -> bool:
            self.mapped(destination).write_bytes(source.read_bytes())
            return True

        def download_file(self, source: str, destination: Path) -> bool:
            destination.write_bytes(self.mapped(source).read_bytes())
            return True

        def execute(self, name: str, args: dict, **kwargs: object) -> ToolResult:
            command = args["command"].replace("/tmp/ash-pro-collect-", str(guest_tmp / "ash-pro-collect-"))
            command = command.replace("os.chdir('/app')", f"os.chdir({str(repo)!r})")
            command = command.replace("cd /app", f"cd {shlex.quote(str(repo))}")
            command = command.replace("python -", f"{sys.executable} -", 1)
            environment = dict(os.environ, TMPDIR=str(guest_tmp), PATH=f"{shim_dir}:{os.environ['PATH']}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, env=environment)
            return ToolResult(success=result.returncode == 0, output=result.stdout,
                              outcome=CommandOutcome(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))

    patch = grading.collect_patch(LocalSession(), ["baggage.txt"], task, directory)
    assert all(name in patch for name in ["tracked.txt", "committed.txt", "staged.txt", "new 'file.txt"])
    assert "baggage" not in patch
    assert index_path.read_bytes() == index_before
    assert b"new copyright \xa9\r\n" in patch.encode("utf-8", "surrogateescape")
    assert (directory / "model.patch").read_bytes() == patch.encode("utf-8", "surrogateescape")
    pristine = tmp_path / "pristine"
    git("worktree", "add", "--detach", str(pristine), base)
    subprocess.run(["git", "-C", str(pristine), "apply", str(directory / "model.patch")], check=True)
    assert (pristine / "legacy.txt").read_bytes() == (repo / "legacy.txt").read_bytes()


@pytest.mark.parametrize("running,timed_out", [(True, False), (False, True)])
def test_unfinished_execution_is_an_error(running: bool, timed_out: bool) -> None:
    result = ToolResult(success=True, output="", outcome=CommandOutcome(exit_code=0, running=running, timed_out=timed_out))
    with pytest.raises(RuntimeError, match="incomplete"):
        grading.shell(SimpleNamespace(execute=lambda *args, **kwargs: result), "true")


@pytest.mark.parametrize("collector_port", [None, 3000])
def test_two_vm_grading_preserves_resources_artifacts_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collector_port: int | None) -> None:
    task = make_task(tmp_path / "upstream")
    sessions = []

    class FakeSession:
        def __init__(self, **kwargs: object):
            self.backend = kwargs["backend"]
            self.created = None
            self.destroyed = False
            sessions.append(self)

        def create(self, image: str, resources: dict) -> bool:
            self.created = image
            self.resources = resources
            return True

        def execute(self, *args: object, **kwargs: object) -> ToolResult:
            return ToolResult(success=True, output="")

        def download_file(self, source: str, destination: Path) -> bool:
            destination.write_text("baggage\0" if source.endswith("baseline.paths") else "archive")
            return True

        def destroy(self) -> None:
            self.destroyed = True

    monkeypatch.setattr(grading, "SandboxSession", FakeSession)

    def collect(session: FakeSession, baseline: list, *args: object) -> str:
        assert session.created == "actor-snapshot" and baseline == ["baggage"]
        return "collected diff"

    def verify(session: FakeSession, task: object, patch: str, directory: Path, timeout: int) -> fork_eval.Grade:
        assert sessions[1].destroyed
        assert session.created != "actor-snapshot"
        assert patch == "collected diff" and timeout == 3600
        return fork_eval.Grade(patch=patch, resolved=True)

    monkeypatch.setattr(grading, "collect_patch", collect)
    monkeypatch.setattr(grading, "verify_in_session", verify)
    collector_backend = None if collector_port is None else {"backend": "microvm", "microvm": {"runtime_port": collector_port}}
    result = grading.grade_snapshot("actor-snapshot", task, {"backend": "microvm", "microvm": {"runtime_port": 34122}},
                                    resources={"cpu": 4, "memory_mb": 16384}, artifacts_dir=tmp_path / "output",
                                    collector_backend=collector_backend)
    assert result.resolved and not result.error and not result.verifier_artifact_error
    assert all(session.destroyed for session in sessions)
    assert all(session.resources == {"cpu": 4, "memory_mb": 16384} for session in sessions)
    assert sessions[0].backend["microvm"]["sandbox_ttl"] >= 4200
    assert sessions[0].backend["microvm"]["runtime_port"] == 34122
    assert sessions[1].backend["microvm"]["runtime_port"] == (collector_port or 34122)
    assert (Path(result.verifier_artifacts) / "metadata.json").exists()


@pytest.mark.parametrize("error,expected_code", [(None, 0), ("verifier unavailable", 2)])
def test_real_cli_wires_loader_preparation_actor_snapshot_and_grade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str | None, expected_code: int,
) -> None:
    task = make_task(tmp_path / "upstream")
    calls = []
    monkeypatch.setattr("swebench_pro.bench.load_tasks", lambda *args: {task.instance_id: task})
    monkeypatch.setattr(grading, "prepare_image", lambda *args: "prepared-base")

    class Orchestrator:
        def __init__(self, **kwargs: object):
            pass

        def run(self, spec: object) -> SimpleNamespace:
            calls.append(spec)
            Path(spec.journal_path).write_text(json.dumps({"type": "checkpoint.captured", "step": 1,
                                                        "snapshot_id": "actor-final", "reason": "captured"}) + "\n")
            return SimpleNamespace(journal_path=spec.journal_path, status="completed", checkpoints=1)

    def grade(snapshot: str, raw: object, backend: dict, **kwargs: object) -> fork_eval.Grade:
        assert snapshot == "actor-final" and raw is task
        assert backend["microvm"]["image_env"]
        assert backend["microvm"]["runtime_port"] == 34122
        assert kwargs["resources"] == {"cpu": 4, "memory_mb": 16384}
        assert kwargs["artifacts_dir"].endswith("parent.verifier")
        return fork_eval.Grade(resolved=error is None, error=error)

    monkeypatch.setattr(fork_eval, "Orchestrator", Orchestrator)
    monkeypatch.setattr(grading, "grade_snapshot", grade)
    monkeypatch.setattr(fork_eval, "report", lambda *args: None)
    code = fork_eval.main(["--benchmark", "swebench-pro", "--pro-repo", str(tmp_path / "upstream"),
                           "--instance", "all", "--slot", "claude-code", "--rounds", "0",
                           "--pro-runtime-port", "34122",
                           "--volatile-ok", "-o", str(tmp_path / "out")])
    assert code == expected_code
    assert len(calls) == 1 and calls[0].sandbox_image == "prepared-base"
    assert calls[0].backend["microvm"]["runtime_port"] == 34122
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["benchmark"] == "swebench-pro"
    assert summary["final_resolved_rate"] == (1.0 if error is None else None)
    assert summary["instances"][0]["attempts"][0]["grading_snapshot"]["snapshot_id"] == "actor-final"


@pytest.mark.parametrize("apply_code,test_code,setup_command,expected", [
    (0, 1, "true", 0), (0, 0, "false", 1), (1, 0, "true", 41),
])
def test_guarded_shell_runs_parser_after_failed_tests_only(
    tmp_path: Path, apply_code: int, test_code: int, setup_command: str, expected: int,
) -> None:
    original = ("export LANG en_US.UTF-8\nexport LC_ALL POSIX\n"
                f"git() {{ if [ \"$1\" = apply ]; then return {apply_code}; fi; return 0; }}\n"
                "cd /app\ngit reset --hard base\ngit checkout base\ngit apply -v /workspace/patch.diff\n"
                f"{setup_command}\n"
                "bash /workspace/run_script.sh test.py > /workspace/stdout.log 2> /workspace/stderr.log\n"
                "python /workspace/parser.py\n")
    (tmp_path / "run_script.sh").write_text(f"exit {test_code}\n")
    (tmp_path / "parser.py").write_text("from pathlib import Path\nPath('parsed').write_text('yes')\n")
    guarded = grading.guarded_entryscript(original, "patch")
    guarded = guarded.replace("/workspace", str(tmp_path)).replace("/app", str(tmp_path))
    guarded = guarded.replace("python ", sys.executable + " ")
    result = subprocess.run(["bash", "-c", guarded], capture_output=True, text=True)
    assert result.returncode == expected
    assert (tmp_path / "parsed").exists() == (expected == 0)


def test_local_dataset_is_complete_and_duplicates_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from swebench_pro import tasks

    task = make_task(tmp_path / "upstream")
    checked = []
    monkeypatch.setattr(tasks, "validate_repo", lambda repo: checked.append(repo))
    data = tmp_path / "data.jsonl"
    data.write_text(task.sample_json + "\n")
    loaded = tasks.load_tasks(task.harness_repo, data)
    assert list(loaded) == [task.instance_id]
    assert "sha256" in loaded[task.instance_id].data_source
    assert checked == [task.harness_repo]
    data.write_text(task.sample_json + "\n" + task.sample_json + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        tasks.load_tasks(task.harness_repo, data)


def test_gate_uses_the_benchmark_snapshot_grader(tmp_path: Path) -> None:
    from swebench_pro.gate import check

    task = make_task(tmp_path / "upstream")
    calls = []
    bench = SimpleNamespace(instance=lambda task: {"task": task},
                            prepare_image=lambda *args: "prepared-nop",
                            grade=lambda *args: calls.append(args) or fork_eval.Grade())
    record = check(bench, task, {"backend": "microvm"}, "nop", tmp_path)
    assert record["ok"] and not record["expected_resolved"]
    assert calls[0][0] == "prepared-nop"
    assert calls[0][1]["verifier_artifacts_dir"] == str(tmp_path / "verifier")
