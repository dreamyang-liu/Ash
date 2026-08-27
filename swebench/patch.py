"""Extracting the agent's work as a patch — one implementation, two callers.

A prediction should be everything the agent changed, and nothing else. Both
halves are easy to get wrong in opposite directions.

`git add -A` errs one way: it stages every untracked path, including ones that
were already there before the agent started. A `psf/requests` prediction came out
at 872,340 characters across 66 files, of which one was a source change and 65
were a `build/` tree that ships **inside the SWE-bench image** — `git status` in a
freshly created sandbox already reports `?? build/`. The official grader applies
whatever it is handed, so nothing downstream catches this.

Staging only tracked files (`git add -u`) errs the other way: it silently drops a
source file the agent created, and only the agent knows whether a new file is
part of the answer or a scratch script. mini-swe-agent avoids the whole question
by having the model produce the diff itself (`git diff -- <files it edited>`),
which is the most honest answer available — the author of a change knows its
extent.

This takes the same position without depending on the model getting a submission
ritual right: record which paths are untracked when the sandbox is created, and
treat only *newly* appeared paths as the agent's. Image baggage is excluded
because it predates the agent, not because it is untracked; a source file the
agent adds is included because the agent added it. The prompt still asks for
scratch to be cleaned up, so the two reinforce rather than substitute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

#: Where SWE-bench images check out the repository under test.
WORKDIR = "/testbed"

Shell = Callable[[str], object]     # command -> ToolResult-like (.success, .output)

__all__ = ["WORKDIR", "extract_patch", "baseline_untracked", "added_paths",
           "select_added", "stage_commands", "diff_command", "format_patch",
           "UNTRACKED_LIST"]

#: Untracked paths, respecting .gitignore. Read-only: never touches the index.
UNTRACKED_LIST = "git ls-files --others --exclude-standard"

#: Paths never worth submitting whatever their status: caches and compiled
#: output are not a change to the repository under any reading, and an agent that
#: forgets to clean one up should not have its answer buried.
NEVER_SUBMIT = (
    "__pycache__/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
    ".tox/", ".eggs/", "*.egg-info/", "*.pyc", "*.pyo", "*.so", "*.o",
)


def _output(result) -> str:
    return (getattr(result, "output", "") or "") if getattr(result, "success", False) else ""


def _lines(result) -> list[str]:
    return [line.strip() for line in _output(result).splitlines() if line.strip()]


def baseline_untracked(shell: Shell) -> set[str]:
    """Untracked paths that exist before the agent runs.

    Called once, at sandbox creation. A SWE-bench image can arrive with a
    `build/` tree or a stray artifact already in place; those are the image's,
    not the agent's, and the difference is invisible later.
    """
    return set(_lines(shell(UNTRACKED_LIST)))


def _is_noise(path: str) -> bool:
    from fnmatch import fnmatch
    for pattern in NEVER_SUBMIT:
        if pattern.endswith("/"):
            if path.startswith(pattern) or f"/{pattern}" in f"/{path}":
                return True
        elif fnmatch(path, pattern) or fnmatch(path.rsplit("/", 1)[-1], pattern):
            return True
    return False


def select_added(untracked: Iterable[str], baseline: Iterable[str] = ()) -> list[str]:
    """The agent's new paths, given the current and baseline untracked lists.

    Pure, so the async caller can do its own listing and still apply exactly the
    same rules -- the part that decides what a prediction contains is shared even
    though the two callers cannot share an executor.
    """
    before = set(baseline)
    return sorted(p for p in (s.strip() for s in untracked)
                  if p and p not in before and not _is_noise(p))


def added_paths(shell: Shell, baseline: Iterable[str] = ()) -> list[str]:
    """Untracked paths the agent created, minus caches and compiled output."""
    return select_added(_lines(shell(UNTRACKED_LIST)), baseline)


def stage_commands(added: Iterable[str] = ()) -> list[str]:
    """Commands that stage the agent's work.

    `-u` covers modifications and deletions of tracked files; each newly added
    path is then staged by name. Named explicitly rather than with `-A` so a path
    that was already there cannot ride along.
    """
    commands = ["git add -u"]
    paths = [p for p in added if p]
    if paths:
        quoted = " ".join(f"'{p}'" for p in paths)
        commands.append(f"git add -- {quoted}")
    return commands


def diff_command(base_commit: str = "") -> str:
    return f"git diff --cached {base_commit or 'HEAD'}"


def format_patch(diff_output: str) -> str:
    """Normalise a diff for a prediction: trailing newline, or empty."""
    patch = (diff_output or "").rstrip("\r\n")
    return patch + "\n" if patch else ""


def extract_patch(shell: Shell, base_commit: str = "",
                  baseline: Iterable[str] = ()) -> tuple[str, list[str]]:
    """(patch, paths added by the agent) — the diff of everything it changed.

    ``shell`` runs one command in the repository and returns something with
    ``.success`` and ``.output``. The async caller (the MCP proxy) drives
    ``stage_commands``/``diff_command`` directly.
    """
    added = added_paths(shell, baseline)
    for command in stage_commands(added):
        shell(command)
    return format_patch(_output(shell(diff_command(base_commit)))), added


# ---------------------------------------------------------------------------
# Async extraction (sandbox rather than a sync shell)
# ---------------------------------------------------------------------------

async def extract_patch_async(sandbox, base_commit: str, *,
                        baseline_untracked: "set[str] | None" = None) -> str:
    """Build this repository's diff from a live sandbox.

    A plain function of a sandbox, deliberately: with per-step snapshots the
    harness can restore any step and call this, so extraction does not have to
    happen while the agent runs and can be re-run after the fact (a fix to what
    counts as part of a patch then applies to runs already finished).

    ``baseline_untracked`` names paths that were already untracked before the
    agent started, which must not be staged -- a SWE-bench image can ship a
    ``build/`` tree, and without the baseline it is indistinguishable from a file
    the agent created. Prefer :func:`untracked_baseline` against the run's step-0
    snapshot; passing ``None`` stages every untracked path it finds.

    Returns "" when there is nothing to diff (no repository, or a failed probe):
    ``format_patch`` cannot tell a shell error payload from a diff, and such a
    payload written to a .diff file used to be read back as a prediction.
    """
    listed = await sandbox.call("shell", command=f"cd {WORKDIR} && {UNTRACKED_LIST}")
    if listed.is_error:
        return ""
    added = select_added((listed.output or "").splitlines(), baseline_untracked or set())
    for command in stage_commands(added):
        await sandbox.call("shell", command=f"cd {WORKDIR} && {command}")
    result = await sandbox.call(
        "shell", command=f"cd {WORKDIR} && {diff_command(base_commit)}")
    return "" if result.is_error else format_patch(result.output)


async def untracked_baseline(sandbox) -> "set[str]":
    """Untracked paths present in a pristine sandbox (i.e. a step-0 snapshot)."""
    probe = await sandbox.call("shell", command=f"cd {WORKDIR} && {UNTRACKED_LIST}")
    if probe.is_error:
        return set()
    return {line.strip() for line in (probe.output or "").splitlines() if line.strip()}
