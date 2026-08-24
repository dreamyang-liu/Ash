"""Loading SWE-Marathon tasks, and grading with their own verifiers."""

import json

import pytest

from swebench.marathon import (LOGS_MOUNT, TESTS_MOUNT, MarathonError,
                               discover_tasks, grade, image_tag, load_task)


def write_task(root, name="zstd-decoder", *, instruction="Build a decoder.",
               toml=None, tests=("test.sh",)):
    directory = root / "tasks" / name
    (directory / "environment").mkdir(parents=True)
    (directory / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (directory / "instruction.md").write_text(instruction)
    (directory / "task.toml").write_text(toml if toml is not None else """
[task]
name = "abundant/zstd-decoder"

[metadata]
difficulty = "hard"
expert_time_estimate_hours = 12
tags = ["c"]

[verifier]
timeout_sec = 1200.0

[agent]
timeout_sec = 18000.0

[environment]
cpus = 4
memory_mb = 16384
""")
    tests_dir = directory / "tests"
    tests_dir.mkdir()
    for filename in tests:
        (tests_dir / filename).write_bytes(b"#!/bin/bash\nexit 0\n")
    return directory


# --- loading --------------------------------------------------------------- #

def test_task_directory_is_parsed(tmp_path):
    task = load_task(write_task(tmp_path))
    assert task.name == "abundant/zstd-decoder"
    assert task.instance_id == "zstd-decoder", "directory name is the stable id"
    assert task.instruction == "Build a decoder."
    assert (task.agent_timeout_sec, task.verifier_timeout_sec) == (18000.0, 1200.0)
    assert (task.cpus, task.memory_mb) == (4, 16384)
    assert task.metadata["expert_time_estimate_hours"] == 12


def test_incomplete_task_fails_before_anything_expensive(tmp_path):
    """Building an image takes minutes; a task with no instruction should be
    rejected on the way in, not after."""
    directory = write_task(tmp_path)
    (directory / "instruction.md").unlink()
    with pytest.raises(MarathonError, match="instruction.md"):
        load_task(directory)


def test_discovery_finds_tasks_and_skips_other_directories(tmp_path):
    write_task(tmp_path, "zstd-decoder")
    write_task(tmp_path, "slack-clone")
    (tmp_path / "tasks" / "README.md").write_text("not a task")
    (tmp_path / "tasks" / "scratch").mkdir()      # no task.toml
    found = [t.instance_id for t in discover_tasks(tmp_path)]
    assert found == ["slack-clone", "zstd-decoder"], "name-sorted, real only"


def test_image_reference_is_per_task(tmp_path):
    a = load_task(write_task(tmp_path, "zstd-decoder"))
    b = load_task(write_task(tmp_path, "slack-clone"))
    assert image_tag(a) != image_tag(b)
    assert image_tag(a, registry="reg:5001").startswith("reg:5001/")


# --- grading --------------------------------------------------------------- #

class FakeResult:
    def __init__(self, output="", error=None):
        self.output = output
        self.error = error
        self.success = not error


class FakeSession:
    """Records what grading did, and replays canned verifier artifacts."""

    def __init__(self, reward="1", metrics=None, verifier_output="",
                 can_upload=True):
        self.reward = reward
        self.metrics = metrics if metrics is not None else {
            "total_passed": 43, "total_tests": 43, "partial_score": 1.0}
        self.verifier_output = verifier_output
        self.can_upload = can_upload
        self.commands: list[str] = []
        self.uploads: list[tuple] = []

    def supports_upload(self):
        return self.can_upload

    def upload_file(self, source, destination):
        if not self.can_upload:
            return False
        self.uploads.append((str(source), destination))
        return True

    def execute(self, tool_name, args, timeout=None):
        command = args.get("command", "")
        self.commands.append(command)
        if "test.sh" in command:
            return FakeResult(self.verifier_output)
        if "reward.txt" in command:
            return FakeResult(self.reward)
        if "metrics.json" in command:
            return FakeResult(json.dumps(self.metrics) if self.metrics else "")
        return FakeResult("")


def test_grading_stages_fixtures_and_runs_the_task_script(tmp_path):
    task = load_task(write_task(tmp_path, tests=("test.sh", "expected.bin")))
    session = FakeSession()
    result = grade(session, task)

    # Fixtures reach the mount the script expects, including binary ones a
    # tool call could not carry.
    assert {destination for _, destination in session.uploads} == {
        f"{TESTS_MOUNT}/test.sh", f"{TESTS_MOUNT}/expected.bin"}
    assert any(f"bash {TESTS_MOUNT}/test.sh" in c for c in session.commands)
    assert any(f"{LOGS_MOUNT}/verifier" in c for c in session.commands)
    assert result.reward == 1.0 and result.resolved


def test_partial_score_survives_a_zero_reward(tmp_path):
    """Marathon's reward is all-or-nothing and almost always zero; without the
    partial score, 37 of 43 tests and none of them look identical."""
    task = load_task(write_task(tmp_path))
    session = FakeSession(reward="0", metrics={
        "total_passed": 37, "total_tests": 43, "partial_score": 0.860})
    result = grade(session, task)
    assert result.reward == 0.0 and not result.resolved
    assert result.partial_score == 0.860
    assert result.metrics["total_passed"] == 37


def test_missing_reward_file_falls_back_to_the_printed_line(tmp_path):
    task = load_task(write_task(tmp_path))
    session = FakeSession(reward="", metrics={},
                          verifier_output="Reward: 0.0 (2/43 tests passed)")
    result = grade(session, task)
    assert result.reward == 0.0
    assert result.error is None, "a printed verdict is a verdict"


def test_unrunnable_verifier_reports_instead_of_raising(tmp_path):
    """An ungradeable attempt is a zero; raising here would lose the
    trajectory that produced it."""
    task = load_task(write_task(tmp_path))
    session = FakeSession(reward="", metrics={}, verifier_output="bash: not found")
    result = grade(session, task)
    assert result.reward == 0.0
    assert result.error and "no reward" in result.error


def test_staging_failure_is_reported_not_silently_skipped(tmp_path):
    """A verifier run without its fixtures would grade the wrong thing and
    report a confident zero."""
    task = load_task(write_task(tmp_path))
    session = FakeSession(can_upload=False)
    result = grade(session, task)
    assert result.reward == 0.0
    assert result.error and "staging failed" in result.error
    assert not any("test.sh" in c for c in session.commands), (
        "the verifier must not run against missing fixtures")


# --- harness wiring -------------------------------------------------------- #

def test_marathon_harness_is_registered_and_mounts_the_guard():
    import inspect
    from swebench.harnesses import HARNESSES
    from swebench.harnesses import marathon as harness_module

    assert "marathon" in HARNESSES
    source = inspect.getsource(harness_module)
    # The reason this harness exists rather than a driver script: the shared
    # machinery comes along instead of being re-mounted by hand each time.
    assert "make_context_window_guard(" in source
    assert "install_checkpoints(" in source
    assert "grade(session, task)" in source


def test_harness_budgets_are_not_swebench_shaped():
    """A 250-step default would end a 5-hour task a third of the way in."""
    from swebench.harnesses.marathon import MarathonHarness
    import inspect
    source = inspect.getsource(MarathonHarness._attempt)
    assert 'step_limit", 1000' in source
    assert 'cost_limit", 50.0' in source


def test_marathon_summarizes_context_while_benchmarks_elide():
    """Different horizons want different folding. On marathon tasks the facts
    worth keeping live only in tool output -- measured on a real 133-step
    attempt, 7 of 8 sampled facts (build flags, expected hashes, which
    interpreters the image lacks) were gone after elision, tool calls
    included -- and rediscovering one costs more steps than the summary costs
    to write. Benchmark-length runs never fold at all, so they stay free."""
    import inspect
    from swebench.harnesses import litellm as benchmark
    from swebench.harnesses import marathon as long_horizon

    assert 'context_strategy", "summarize"' in inspect.getsource(long_horizon)
    assert 'context_strategy", "elide"' in inspect.getsource(benchmark)
