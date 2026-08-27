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
# Sandbox observer
# ---------------------------------------------------------------------------

class PatchObserver:
    """Keeps a git diff per sandbox — SWE-bench's notion of "the answer".

    Mounted on the execution plane with
    ``--observer swebench.patch:patch_observer``; the proxy itself knows nothing
    about patches (see harness/execution/observers.py).

    Two things have to happen at the right moment, and both were learned the hard
    way. The untracked-file baseline must be taken *before* an agent runs, or a
    ``build/`` tree the image shipped is indistinguishable later from a file the
    agent created. And the diff is re-extracted after every mutating call rather
    than at shutdown: extraction at shutdown races the harness's read under load,
    and a killed run left nothing at all.
    """

    name = "swebench-patch"

    def __init__(self, patch_dir: "str | Path | None" = None) -> None:
        self.patch_dir = Path(patch_dir) if patch_dir else None
        self.patches: dict[str, str] = {}

    async def on_created(self, entry) -> None:
        probe = await entry.sandbox.call("shell", command=f"cd {WORKDIR} && {UNTRACKED_LIST}")
        baseline = set()
        if not probe.is_error:
            baseline = {line.strip() for line in (probe.output or "").splitlines() if line.strip()}
        entry.meta["baseline_untracked"] = baseline
        # An empty diff on disk immediately, so a reader always finds a current
        # file even if the agent never edits anything.
        await self._extract(entry)

    async def after_mutating_call(self, entry, tool_name: str, args: dict) -> None:
        await self._extract(entry)

    async def on_destroy(self, entry) -> None:
        await self._extract(entry)

    async def _extract(self, entry) -> str:
        listed = await entry.sandbox.call(
            "shell", command=f"cd {WORKDIR} && {UNTRACKED_LIST}")
        if listed.is_error:
            # No repository to diff (e.g. a plain image with no /testbed). Record
            # an empty patch rather than the shell's error payload: `format_patch`
            # does not distinguish them, so the error text used to be written into
            # the .diff and read back as if it were a prediction.
            self.patches[entry.id] = ""
            self._write(entry, "")
            return ""
        added = select_added((listed.output or "").splitlines(),
                             entry.meta.get("baseline_untracked", set()))
        for command in stage_commands(added):
            await entry.sandbox.call("shell", command=f"cd {WORKDIR} && {command}")
        result = await entry.sandbox.call(
            "shell", command=f"cd {WORKDIR} && {diff_command(entry.base_commit)}")
        patch = "" if result.is_error else format_patch(result.output)
        self.patches[entry.id] = patch
        self._write(entry, patch)
        return patch

    def _write(self, entry, patch: str) -> None:
        if self.patch_dir:
            self.patch_dir.mkdir(parents=True, exist_ok=True)
            (self.patch_dir / f"{entry.id}.diff").write_text(patch)


def patch_observer() -> PatchObserver:
    """Factory for ``--observer swebench.patch:patch_observer``.

    Reads ``ASH_PATCH_DIR`` because an observer spec carries no arguments; the
    harness that spawns the proxy sets it alongside the spec.
    """
    import os

    return PatchObserver(os.environ.get("ASH_PATCH_DIR"))
