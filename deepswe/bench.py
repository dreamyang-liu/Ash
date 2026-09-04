"""The DeepSWE adapter ``swebench.fork_eval`` drives when ``--benchmark deepswe``.

What the loop asks of a benchmark, and what DeepSWE answers:

    catalogue(args)          every task under --tasks-dir, by id
    instance(raw)            the dict the loop prints and passes around
    prompt(instance)         instruction.md verbatim + the facts of THIS sandbox
    branch_prompt(...)       the same, continuing from an earlier attempt
    resources(instance)      the shape task.toml declares (2 CPU / 8 GB)
    grade(snapshot, ...)     deepswe.grade: collect -> their verifier -> Grade
    no_network               True: every task declares network_mode no-network

The prompt adds as little as possible to the task's own instruction: where the
repository is, that there is no internet, that only committed work is graded
(the instruction says so too), and how the two MCP tools behave -- the last is
unavoidable, since the agent's own file tools cannot see the sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from swebench.fork_eval import TOOL_PRIMER, Grade
from deepswe.grade import grade_snapshot
from deepswe.tasks import Task, load_tasks

REPO_DIR = "/app"

#: The SWE-bench primer, re-pointed at /app. Same tools, same quirks.
PRIMER = TOOL_PRIMER.replace("/testbed", REPO_DIR)

PROMPT = """\
You are working in the {repo} repository, checked out at {repo_dir} inside your \
sandbox. The sandbox has NO internet access: everything the project needs is \
already installed, so do not try to download or install anything.

## Task
{instruction}

## How your work is graded
Only COMMITTED work counts. The grader takes `git diff <base> HEAD` from \
{repo_dir}, applies it to a pristine copy of this environment, and runs hidden \
tests there. Uncommitted changes are invisible to it. Work on a new branch and \
commit everything before you finish.

{primer}"""

BRANCH_PROMPT = """\
You are continuing work on a task in the {repo} repository at {repo_dir} in your \
sandbox. The filesystem already holds an earlier attempt's work -- ITS EDITS ARE \
ON DISK, some perhaps uncommitted. Start from the files as they are: run \
`git status` and `git diff` in {repo_dir} to see what the earlier attempt \
changed before you touch anything. The sandbox has NO internet access.

## Task
{instruction}

## What the earlier attempt produced, and how it was graded
{verdict}

## Your direction for this attempt
{hint}

Only COMMITTED work is graded: commit everything on a branch before you finish.

{primer}"""


class DeepSWE:
    name = "deepswe"
    #: Every task in the dataset declares ``network_mode = "no-network"`` for
    #: both agent and verifier; the loop turns sandbox egress off for both.
    no_network = True
    #: Their verifier runs in a container built FROM the task image, so it sees
    #: the image's ENV -- a venv on PATH, /root/go/bin, node_modules/.bin. The
    #: gate found `pytest: not found` (exit 127) the first time the runtime ran
    #: without it. Templates for this benchmark launch the runtime under the
    #: image's ENV; see harness/execution/templates.py.
    image_env = True

    def __init__(self, tasks_dir: "str | Path"):
        if not tasks_dir:
            raise SystemExit("--benchmark deepswe needs --tasks-dir "
                             "(the dataset's tasks/ directory)")
        self.tasks_dir = Path(tasks_dir).expanduser()

    def catalogue(self, args) -> Dict[str, Task]:
        return {task.task_id: task for task in load_tasks(self.tasks_dir)}

    def instance(self, task: Task) -> dict:
        if not task.no_network:
            print("   note: %s declares agent network_mode=%r; running it offline "
                  "anyway (whole batch shares one policy)"
                  % (task.task_id, task.agent_network_mode))
        return {
            "instance_id": task.task_id,
            "repo": task.repo,
            "image": task.image,
            "problem": task.instruction,
            "f2p": list(task.f2p),
            "p2p": list(task.p2p),
            "task": task,
        }

    def prompt(self, instance: dict) -> str:
        return PROMPT.format(repo=instance["repo"], repo_dir=REPO_DIR,
                             instruction=instance["problem"], primer=PRIMER)

    def branch_prompt(self, instance: dict, verdict: str, hint: str) -> str:
        return BRANCH_PROMPT.format(repo=instance["repo"], repo_dir=REPO_DIR,
                                    instruction=instance["problem"],
                                    verdict=verdict, hint=hint, primer=PRIMER)

    def resources(self, instance: dict) -> Optional[dict]:
        task: Task = instance["task"]
        return {"cpu": task.cpus, "memory_mb": task.memory_mb}

    def grade(self, snapshot_id: str, instance: dict, backend: dict) -> Grade:
        return grade_snapshot(snapshot_id, instance["task"], backend)
