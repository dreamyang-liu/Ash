"""Loading a task directory: what we read, what we refuse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepswe.tasks import TaskError, load_task, load_tasks, parse_verifier_dockerfile

IMAGE = "public.ecr.aws/x/y:abc-v1.1"
BASE = "452382821dd9dae7cc36995960656bb94dc47212"

TASK_TOML = f"""\
schema_version = "1.3"
[task]
name = "datacurve/demo-task"
[metadata]
task_id = "demo-task"
language = "go"
repository_url = "https://github.com/carvel-dev/ytt"
base_commit_hash = "{BASE}"
[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0
[[verifier.collect]]
command = "cd /app && mkdir -p /logs/artifacts && git diff --binary {BASE} HEAD > /logs/artifacts/model.patch"
timeout_sec = 300.0
[agent]
network_mode = "no-network"
timeout_sec = 10800.0
[environment]
docker_image = "{IMAGE}"
cpus = 2
memory_mb = 8192
"""

DOCKERFILE = f"""\
# Verifier image: the pinned task image with the hidden tests baked in.
FROM {IMAGE}

COPY test.sh /tests/test.sh
COPY test.patch /tests/test.patch
COPY grader.py /tests/grader.py
COPY config.json /tests/config.json
RUN chmod +x /tests/test.sh
"""


def make_task(root: Path, name: str = "demo-task", *, dockerfile: str = DOCKERFILE,
              toml: str = TASK_TOML) -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(toml.replace('task_id = "demo-task"',
                                                 'task_id = "%s"' % name))
    (task / "instruction.md").write_text("Add `Query` to orderedmap.\n")
    (task / "tests" / "Dockerfile").write_text(dockerfile)
    for f in ("test.sh", "test.patch", "grader.py"):
        (task / "tests" / f).write_text("# %s\n" % f)
    (task / "tests" / "config.json").write_text(json.dumps({
        "base_commit": BASE,
        "f2p_node_ids": ["pkg.TestA", "pkg.TestB"],
        "p2p_node_ids": ["pkg.TestOld"],
    }))
    return task


def test_load_task_reads_everything_the_grader_needs(tmp_path):
    task = load_task(make_task(tmp_path))
    assert task.task_id == "demo-task"
    assert task.repo == "carvel-dev/ytt"
    assert task.image == IMAGE
    assert task.base_commit == BASE
    assert (task.cpus, task.memory_mb) == (2, 8192)
    assert task.agent_timeout_s == 10800.0 and task.no_network
    assert task.verifier_timeout_s == 1800.0
    assert [s.command for s in task.collect][0].startswith("cd /app && mkdir -p")
    # The verifier's file placement comes from ITS Dockerfile, not from a list
    # we maintain -- so a new file they add is copied without a code change.
    assert [(f.source.name, f.destination) for f in task.verifier_files] == [
        ("test.sh", "/tests/test.sh"), ("test.patch", "/tests/test.patch"),
        ("grader.py", "/tests/grader.py"), ("config.json", "/tests/config.json")]
    assert task.verifier_run_steps == ("chmod +x /tests/test.sh",)
    assert task.f2p == ("pkg.TestA", "pkg.TestB") and task.p2p == ("pkg.TestOld",)
    assert task.instruction.startswith("Add `Query`")


def test_verifier_image_must_be_the_task_image(tmp_path):
    other = DOCKERFILE.replace(IMAGE, "public.ecr.aws/x/y:other")
    with pytest.raises(TaskError, match="differs from environment image"):
        load_task(make_task(tmp_path, dockerfile=other))


def test_dockerfile_instructions_we_cannot_replay_are_refused(tmp_path):
    with_env = DOCKERFILE + "ENV FOO=bar\n"
    with pytest.raises(TaskError, match="ENV"):
        load_task(make_task(tmp_path, dockerfile=with_env))


def test_missing_copied_file_is_an_error(tmp_path):
    task = make_task(tmp_path)
    (task / "tests" / "grader.py").unlink()
    with pytest.raises(TaskError, match="grader.py"):
        load_task(task)


def test_no_collect_step_is_an_error(tmp_path):
    toml = TASK_TOML.replace("[[verifier.collect]]", "[[verifier.other]]")
    with pytest.raises(TaskError, match="collect"):
        load_task(make_task(tmp_path, toml=toml))


def test_parse_dockerfile_rejects_multistage_and_flags():
    with pytest.raises(TaskError, match="multi-stage"):
        parse_verifier_dockerfile("FROM a\nFROM b\n")
    with pytest.raises(TaskError, match="COPY"):
        parse_verifier_dockerfile("FROM a\nCOPY --chown=1 x /y\n")
    with pytest.raises(TaskError, match="no FROM"):
        parse_verifier_dockerfile("COPY x /y\n")


def test_load_tasks_skips_non_task_dirs_but_not_broken_tasks(tmp_path):
    make_task(tmp_path, "a-task")
    make_task(tmp_path, "b-task")
    (tmp_path / "tools").mkdir()            # dataset keeps helpers beside tasks
    assert [t.task_id for t in load_tasks(tmp_path)] == ["a-task", "b-task"]
    assert [t.task_id for t in load_tasks(tmp_path, wanted=["b-task"])] == ["b-task"]
    with pytest.raises(TaskError, match="not in"):
        load_tasks(tmp_path, wanted=["c-task"])
    (tmp_path / "b-task" / "tests" / "Dockerfile").write_text("FROM other\n")
    with pytest.raises(TaskError):
        load_tasks(tmp_path)
