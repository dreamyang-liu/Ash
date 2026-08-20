"""Patch extraction (swebench/patch.py).

A prediction is a diff of the repository's files. `git add -A` could not tell
those from the agent's scratch: one real run produced a psf/requests prediction of
872,340 characters across 66 files, of which one was a source change and 65 were
`build/` output from a `setup.py build`. The official grader applies whatever it
is handed, so nothing downstream would have caught it.

Uses a real git repository in a tmp dir — the behaviour under test is git's
staging semantics, which a stubbed shell could only restate.

Covered:
- a modified tracked file is in the patch
- a created file is not, however plausible it looks
- build output does not drown the real change
- deletions of tracked files survive
- the base commit is respected, so earlier commits are not re-diffed
- untracked_paths reports what was excluded
- both callers (session, MCP proxy) issue the same commands
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from swebench.patch import (
    UNTRACKED_LIST,
    extract_patch,
    format_patch,
    patch_commands,
    untracked_paths,
)


class Repo:
    """A real git repo plus the `shell` callable patch.py expects."""

    def __init__(self, path: Path):
        self.path = path
        self._git("init", "-q", ".")
        self._git("config", "user.email", "t@example.com")
        self._git("config", "user.name", "t")

    def _git(self, *args: str) -> str:
        return subprocess.run(("git",) + args, cwd=self.path,
                              capture_output=True, text=True).stdout

    def write(self, rel: str, text: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, *paths: str) -> str:
        self._git("add", *paths)
        self._git("commit", "-qm", "c")
        return self._git("rev-parse", "HEAD").strip()

    def shell(self, command: str):
        done = subprocess.run(command, cwd=self.path, shell=True,
                              capture_output=True, text=True)
        class R:
            success = done.returncode == 0
            output = done.stdout
        return R()


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path)
    r.write("pkg/mod.py", "def f():\n    return 1\n")
    r.write(".gitignore", "*.log\n")
    r.commit("pkg/mod.py", ".gitignore")
    return r


# --------------------------------------------------------------------------- #
#  What belongs in a prediction
# --------------------------------------------------------------------------- #

def test_a_modified_source_file_is_the_patch(repo: Repo):
    repo.write("pkg/mod.py", "def f():\n    return 2\n")
    patch = extract_patch(repo.shell)
    assert "pkg/mod.py" in patch
    assert "return 2" in patch


def test_build_output_does_not_drown_the_real_change(repo: Repo):
    """The case from the run: `setup.py build` left 65 files beside one fix."""
    repo.write("pkg/mod.py", "def f():\n    return 2\n")
    for i in range(65):
        repo.write(f"build/lib/pkg/gen{i}.py", "x = 1\n" * 200)

    patch = extract_patch(repo.shell)
    assert "pkg/mod.py" in patch
    assert "build/" not in patch
    assert patch.count("diff --git") == 1
    assert len(patch) < 1000, f"patch is {len(patch)} chars of noise"


def test_a_created_file_is_left_out(repo: Repo):
    """An agent's reproduce script is how it worked, not what it changed. Across
    SWE-bench Verified only 1 gold patch in 500 adds a file (0.2%), so this
    trade-off costs almost nothing and removes constant noise."""
    repo.write("reproduce_bug.py", "print('repro')\n")
    repo.write("pkg/new_helper.py", "def helper():\n    pass\n")
    patch = extract_patch(repo.shell)
    assert patch == ""                       # nothing tracked was touched


def test_a_deleted_tracked_file_is_in_the_patch(repo: Repo):
    """-u stages deletions too; a fix that removes a file must survive."""
    (repo.path / "pkg" / "mod.py").unlink()
    patch = extract_patch(repo.shell)
    assert "pkg/mod.py" in patch
    assert "deleted file" in patch


def test_only_changes_since_the_base_commit_are_included(repo: Repo):
    """Sandboxes are created from an image whose HEAD may already carry commits;
    the diff is against the commit the run started from."""
    repo.write("pkg/mod.py", "def f():\n    return 2\n")
    base = repo.commit("pkg/mod.py")         # a commit the agent did not make
    repo.write("pkg/mod.py", "def f():\n    return 3\n")

    patch = extract_patch(repo.shell, base_commit=base)
    # The diff spans base..working: `return 2` is the line being removed, and
    # `return 1` (from before the base commit) does not appear at all.
    assert "+    return 3" in patch
    assert "-    return 2" in patch
    assert "return 1" not in patch


def test_an_untouched_repository_yields_an_empty_patch(repo: Repo):
    assert extract_patch(repo.shell) == ""


# --------------------------------------------------------------------------- #
#  Reporting what was excluded
# --------------------------------------------------------------------------- #

def test_untracked_paths_names_what_the_patch_omits(repo: Repo):
    repo.write("reproduce_bug.py", "x\n")
    repo.write("build/lib/gen.py", "y\n")
    left_out = untracked_paths(repo.shell)
    assert "reproduce_bug.py" in left_out
    assert any(p.startswith("build/") for p in left_out)


def test_untracked_paths_respects_gitignore(repo: Repo):
    """A repo that already calls something an artifact is not worth reporting."""
    repo.write("debug.log", "noise\n")       # matched by .gitignore
    assert "debug.log" not in untracked_paths(repo.shell)


def test_untracked_paths_is_bounded(repo: Repo):
    for i in range(50):
        repo.write(f"build/f{i}.py", "x\n")
    assert len(untracked_paths(repo.shell, limit=5)) == 5


# --------------------------------------------------------------------------- #
#  One implementation, two callers
# --------------------------------------------------------------------------- #

def test_staging_never_force_adds():
    """`git add -A` is what caused this; a regression would be silent, since the
    patch still applies -- it just carries 98% noise."""
    stage, _ = patch_commands()
    assert stage == "git add -u"
    assert "-A" not in stage


def test_both_callers_use_the_shared_commands():
    """The session and the MCP proxy produce predictions for the same benchmark;
    they must not disagree about what a prediction contains."""
    root = Path(__file__).resolve().parents[1]
    for name in ("sandbox.py", "mcp_server.py"):
        source = (root / name).read_text()
        assert "git add -A" not in source, f"{name} force-adds untracked files again"
        assert "patch_commands" in source or "extract_patch" in source, \
            f"{name} should build its patch via swebench/patch.py"


def test_format_patch_normalises_the_diff():
    assert format_patch("") == ""
    assert format_patch("diff --git a/x b/x") == "diff --git a/x b/x\n"
    assert format_patch("d\n\n\n") == "d\n"        # one trailing newline


def test_untracked_list_respects_the_repo_and_stays_read_only():
    assert "--exclude-standard" in UNTRACKED_LIST   # honours .gitignore
    assert "--others" in UNTRACKED_LIST
    assert "add" not in UNTRACKED_LIST              # never mutates the index
