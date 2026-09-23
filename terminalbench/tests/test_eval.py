from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from terminalbench.eval import audit_agentenv, build_config, parse_args, preflight, run_job, summarize


def manifest(output: Path, expected: int = 3) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "ash-eval.json").write_text(json.dumps({
        "dataset": "terminal-bench/terminal-bench@4.0.0", "expected_trials": expected,
        "task_filter": [], "attempts_per_task": 1,
    }))


def report(output: Path, name: str, reward: object, **changes: object) -> None:
    directory = output / name
    directory.mkdir(exist_ok=True)
    (directory / "result.json").write_text(json.dumps({
        "task_name": name, "trial_name": name, "finished_at": "2026-09-09",
        "verifier_result": {"rewards": {"reward": reward}}, "exception_info": None, **changes,
    }))


def test_job_config_pins_v4_and_preserves_task_resource_limits(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    args = parse_args(["--model", "test-model", "--output", str(tmp_path / "job"),
                       "--workers", "32", "--attempts", "3", "--task", "one", "--env", "docker"])
    config = build_config(args)
    assert config.datasets[0].name == "terminal-bench/terminal-bench"
    assert config.datasets[0].version == "4.0.0"
    assert config.datasets[0].task_names == ["one"]
    assert config.agents[0].import_path == "terminalbench.agent:AshClaudeCode"
    assert config.agents[0].override_timeout_sec is None
    assert config.environment.override_gpus is None
    assert config.environment.override_cpus is None
    assert config.timeout_multiplier == 1
    assert config.n_concurrent_trials == 32 and config.n_attempts == 3
    assert config.retry.max_retries == 0


def test_summary_never_turns_errors_or_missing_reports_into_final_zero(tmp_path: Path) -> None:
    manifest(tmp_path)
    report(tmp_path, "good", 1)
    report(tmp_path, "broken", None, exception_info={"exception_type": "BuildError"})
    result = summarize(tmp_path)
    assert result["completed_trials"] == 1
    assert result["error_trials"] == 1
    assert result["missing_or_incomplete_trials"] == 1
    assert result["mean_reward_lower_bound"] == 1 / 3
    assert result["mean_reward_completed"] == 1
    assert result["final_mean_reward"] is None


@pytest.mark.parametrize("reward", [None, True, "1", -1, 2, float("nan")])
def test_invalid_rewards_are_grading_errors(tmp_path: Path, reward: object) -> None:
    manifest(tmp_path, 1)
    report(tmp_path, "task", reward)
    result = summarize(tmp_path)
    assert result["error_trials"] == 1 and not result["complete"]


def test_valid_zero_and_fractional_rewards_are_preserved(tmp_path: Path) -> None:
    manifest(tmp_path)
    for name, reward in [("zero", 0), ("partial", 0.5), ("solved", 1)]:
        report(tmp_path, name, reward)
    result = summarize(tmp_path)
    assert result["complete"]
    assert result["resolved_trials"] == 1
    assert result["final_mean_reward"] == 0.5


def test_interrupted_trial_is_incomplete_even_with_a_reward(tmp_path: Path) -> None:
    manifest(tmp_path, 1)
    report(tmp_path, "task", 1, finished_at=None)
    assert summarize(tmp_path)["missing_or_incomplete_trials"] == 1


def test_harbor_factory_loads_the_ash_agent(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    from terminalbench.agent import AshClaudeCode
    from harbor.agents.factory import AgentFactory

    agent = AgentFactory.create_agent_from_import_path(
        "terminalbench.agent:AshClaudeCode", logs_dir=tmp_path, model_name="test-model")
    assert isinstance(agent, AshClaudeCode)


def test_job_calls_official_lifecycle_and_summarizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("harbor")
    from harbor.job import Job

    output = tmp_path / "job"
    args = parse_args(["--agent", "nop", "--phase", "run", "--output", str(output), "--env", "docker"])
    config = build_config(args)
    calls = []

    class FakeJob:
        def __len__(self) -> int:
            return 1

        async def run(self) -> None:
            calls.append("run")
            report(output, "task", 0)

    async def create(received: object) -> FakeJob:
        assert received == config
        output.mkdir(exist_ok=True)
        calls.append("create")
        return FakeJob()

    monkeypatch.setattr(Job, "create", create)
    result = asyncio.run(run_job(args, config))
    assert calls == ["create", "run"]
    assert result["complete"] and result["final_mean_reward"] == 0
    with pytest.raises(FileExistsError):
        asyncio.run(run_job(args, config))
    stored = json.loads((output / "ash-eval.json").read_text())
    stored["config"]["retry"]["exclude_exceptions"].reverse()
    (output / "ash-eval.json").write_text(json.dumps(stored))
    args.resume = True
    assert asyncio.run(run_job(args, config))["complete"]
    config.n_attempts = 2
    with pytest.raises(ValueError, match="Resume config differs"):
        asyncio.run(run_job(args, config))


def test_missing_compose_stops_before_job_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "docker compose version")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Docker Compose v2"):
        preflight(parse_args(["--agent", "nop", "--env", "docker"]))


@pytest.mark.parametrize("disk,supported", [(10240, True), (65536, True), (1536, False)])
def test_admission_accepts_small_disks_but_rejects_unaligned_disks(tmp_path, disk, supported):
    pytest.importorskip("harbor")
    source = tmp_path / "task"
    shutil.copytree(Path(__file__).parent / "agentenv_fixtures" / "shell-task", source)
    definition = source / "task.toml"
    definition.write_text(definition.read_text().replace("storage_mb = 65536", f"storage_mb = {disk}"))
    runtime = tmp_path / "runtime"
    runtime.touch()
    args = parse_args(["--agent", "nop", "--output", str(tmp_path), "--runtime-bin", str(runtime)])
    config = build_config(args)
    job = SimpleNamespace(job_dir=tmp_path, _task_download_results={"task": SimpleNamespace(path=source)})
    report = audit_agentenv(job, config)
    assert report[0]["supported"] is supported
    if not supported:
        assert "divisible by 1024" in report[0]["errors"][0]
