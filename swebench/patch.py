"""Extracting the agent's work as a patch — one implementation, two callers.

A prediction is a diff of the *repository's* files. An agent also leaves things
behind that are not part of its answer: a reproduce script it wrote to see the
bug, `build/` from a `setup.py build`, `.pytest_cache`, `__pycache__`. Those are
how it worked, not what it changed.

`git add -A` cannot tell the difference — it stages every untracked file, and
force-stages ones the repo ignores. Measured on one run: a `psf/requests`
prediction came out at 872,340 characters across 66 files, of which **one** was a
source change and 65 were `build/` output. The official grader applies whatever we
hand it, so nothing downstream would have caught that.

So the patch is built from tracked files only (`git add -u`). Deliberately
excluding a newly added source file is a real trade-off, and the data says it is
the right one: across SWE-bench Verified, 1 gold patch in 500 adds a file (0.2%),
while an agent leaves scratch behind constantly. `untracked_paths` reports what
was left out, so a caller can log it rather than wonder.
"""

from __future__ import annotations

from typing import Callable

#: Where SWE-bench images check out the repository under test.
WORKDIR = "/testbed"

Shell = Callable[[str], object]     # command -> ToolResult-like (.success, .output)

__all__ = ["WORKDIR", "extract_patch", "untracked_paths", "patch_commands",
           "format_patch", "UNTRACKED_LIST"]

#: Lists the files a patch leaves out (respecting .gitignore).
UNTRACKED_LIST = "git ls-files --others --exclude-standard"


def _output(result) -> str:
    return (getattr(result, "output", "") or "") if getattr(result, "success", False) else ""


#: The commands that produce a patch, in order. Shared so the sync and async
#: callers cannot drift on the part that matters -- which files get staged.
def patch_commands(base_commit: str = "") -> tuple[str, str]:
    """(stage, diff) — the two commands, given a base commit.

    `git add -u`, not `-A`: stage modifications and deletions of files the repo
    already tracks, and leave the agent's scratch alone.
    """
    return "git add -u", f"git diff --cached {base_commit or 'HEAD'}"


def format_patch(diff_output: str) -> str:
    """Normalise a diff for a prediction: trailing newline, or empty."""
    patch = (diff_output or "").rstrip("\r\n")
    return patch + "\n" if patch else ""


def extract_patch(shell: Shell, base_commit: str = "") -> str:
    """The diff of tracked files against ``base_commit``.

    ``shell`` runs one command in the repository and returns something with
    ``.success`` and ``.output``. The async caller (the MCP proxy) uses
    ``patch_commands`` and ``format_patch`` directly instead of this wrapper.
    """
    stage, diff = patch_commands(base_commit)
    shell(stage)
    return format_patch(_output(shell(diff)))


def untracked_paths(shell: Shell, limit: int = 20) -> list[str]:
    """Paths the agent created that the patch leaves out.

    Worth logging: a reproduce script is expected, but a `build/` tree usually
    means a command was run that the task did not need, and an untracked file
    under a source directory may be a real answer this excludes (rare — see the
    module docstring).
    """
    listed = _output(shell(UNTRACKED_LIST))
    paths = [line.strip() for line in listed.splitlines() if line.strip()]
    return paths[:limit]
