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
    NEVER_SUBMIT,
    UNTRACKED_LIST,
    baseline_untracked,
    extract_patch,
    format_patch,
    select_added,
    stage_commands,
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
    patch, _ = extract_patch(repo.shell)
    assert "pkg/mod.py" in patch
    assert "return 2" in patch


def test_caches_and_compiled_output_are_never_submitted(repo: Repo):
    """These are not a change to the repository under any reading, and an agent
    that forgets to clean one up should not have its answer buried."""
    repo.write("pkg/mod.py", "def f():\n    return 2\n")
    repo.write("pkg/__pycache__/mod.cpython-311.pyc", "\x00binary")
    repo.write(".pytest_cache/v/cache/lastfailed", "{}")
    repo.write("pkg/ext.so", "\x00elf")

    patch, added = extract_patch(repo.shell)
    assert "pkg/mod.py" in patch
    for noise in ("__pycache__", ".pytest_cache", "ext.so"):
        assert noise not in patch, f"{noise} reached the prediction"
    assert added == []


def test_a_file_the_agent_created_is_in_the_patch(repo: Repo):
    """Only the agent knows whether a new file is part of the answer. Dropping
    every new file would silently discard a legitimate one (rare, but real: 1
    gold patch in 500 adds a file) -- so a new path is included, and the prompt
    is where scratch gets discouraged."""
    repo.write("pkg/new_helper.py", "def helper():\n    pass\n")
    patch, added = extract_patch(repo.shell)
    assert "pkg/new_helper.py" in patch
    assert "new file mode" in patch
    assert added == ["pkg/new_helper.py"]


def test_what_the_image_already_had_is_not_the_agents_work(repo: Repo):
    """The case that started this: SWE-bench's psf/requests image ships a
    `build/` tree, so a fresh sandbox already reports `?? build/`. Staged with
    `-A`, that became 65 of a prediction's 66 files."""
    for i in range(30):
        repo.write(f"build/lib/pkg/gen{i}.py", "x = 1\n" * 50)
    baseline = baseline_untracked(repo.shell)     # recorded at sandbox creation

    repo.write("pkg/mod.py", "def f():\n    return 2\n")   # the agent's fix
    patch, added = extract_patch(repo.shell, baseline=baseline)

    assert "pkg/mod.py" in patch
    assert "build/" not in patch
    assert added == []


def test_a_deleted_tracked_file_is_in_the_patch(repo: Repo):
    """-u stages deletions too; a fix that removes a file must survive."""
    (repo.path / "pkg" / "mod.py").unlink()
    patch, _ = extract_patch(repo.shell)
    assert "pkg/mod.py" in patch
    assert "deleted file" in patch


def test_only_changes_since_the_base_commit_are_included(repo: Repo):
    """Sandboxes are created from an image whose HEAD may already carry commits;
    the diff is against the commit the run started from."""
    repo.write("pkg/mod.py", "def f():\n    return 2\n")
    base = repo.commit("pkg/mod.py")         # a commit the agent did not make
    repo.write("pkg/mod.py", "def f():\n    return 3\n")

    patch, _ = extract_patch(repo.shell, base_commit=base)
    # The diff spans base..working: `return 2` is the line being removed, and
    # `return 1` (from before the base commit) does not appear at all.
    assert "+    return 3" in patch
    assert "-    return 2" in patch
    assert "return 1" not in patch


def test_an_untouched_repository_yields_an_empty_patch(repo: Repo):
    assert extract_patch(repo.shell)[0] == ""


# --------------------------------------------------------------------------- #
#  Reporting what was excluded
# --------------------------------------------------------------------------- #

def test_baseline_records_what_was_there_first(repo: Repo):
    repo.write("build/lib/gen.py", "y\n")
    assert baseline_untracked(repo.shell) == {"build/lib/gen.py"}


def test_gitignored_files_are_not_the_agents_work(repo: Repo):
    """A repo that already calls something an artifact settles the question."""
    repo.write("debug.log", "noise\n")       # matched by .gitignore
    patch, added = extract_patch(repo.shell)
    assert added == [] and patch == ""





# --------------------------------------------------------------------------- #
#  One implementation, two callers
# --------------------------------------------------------------------------- #

def test_staging_never_force_adds():
    """`git add -A` is what caused this; a regression would be silent, since the
    patch still applies -- it just carries 98% noise."""
    for command in stage_commands(["pkg/new.py"]):
        assert "-A" not in command
    # New files are staged by name, so a path that was already there cannot ride
    # along on a wildcard.
    assert stage_commands(["pkg/new.py"])[-1] == "git add -- 'pkg/new.py'"
    assert stage_commands([]) == ["git add -u"]


def test_selection_is_pure_so_both_callers_share_the_rules():
    """The proxy is async and cannot share an executor, so the part that decides
    what a prediction contains is a pure function over the two listings."""
    current = ["build/lib/x.py", "repro.py", "pkg/new.py", "pkg/__pycache__/a.pyc"]
    assert select_added(current, baseline=["build/lib/x.py"]) == \
        ["pkg/new.py", "repro.py"]


def test_never_submit_covers_the_usual_suspects():
    for pattern in ("__pycache__/", "*.pyc", "*.so", ".pytest_cache/"):
        assert pattern in NEVER_SUBMIT


def test_both_callers_use_the_shared_commands():
    """The session and the MCP proxy produce predictions for the same benchmark;
    they must not disagree about what a prediction contains."""
    root = Path(__file__).resolve().parents[1]
    # The proxy's extraction moved into this module as PatchObserver (the generic
    # execution plane knows nothing about patches), so the two callers are the
    # harness session and that observer.
    for name in ("sandbox.py", "patch.py"):
        source = (root / name).read_text()
        # Only executable lines: patch.py *documents* why `git add -A` is the
        # wrong primitive, and that prose must not read as the mistake itself.
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("#", "`"))
        )
        assert 'call("shell", command="git add -A' not in code, \
            f"{name} force-adds untracked files again"
        assert "extract_patch" in source or "stage_commands" in source, \
            f"{name} should build its patch via swebench/patch.py"


def test_format_patch_normalises_the_diff():
    assert format_patch("") == ""
    assert format_patch("diff --git a/x b/x") == "diff --git a/x b/x\n"
    assert format_patch("d\n\n\n") == "d\n"        # one trailing newline


def test_untracked_list_respects_the_repo_and_stays_read_only():
    assert "--exclude-standard" in UNTRACKED_LIST   # honours .gitignore
    assert "--others" in UNTRACKED_LIST
    assert "add" not in UNTRACKED_LIST              # never mutates the index


def test_the_session_passes_its_baseline_to_the_extractor():
    """The wiring, not just the rule: a session that recorded the baseline at
    creation and then forgot to pass it would put the image's `build/` tree back
    into every prediction, and only a live run would notice."""
    from swebench.sandbox import AshSession

    calls: list[str] = []

    class FakeSandbox:
        # The session dispatches through call_agent_tool (it owns the
        # agent-facing tool surface), so that is what a stub must offer.
        async def call_agent_tool(self, name, args, registry=None, agent_id=""):
            calls.append(args.get("command", ""))
            class R:
                is_error = False
                # The listing that select_added reads.
                output = "build/lib/gen.py\nrepro.py\n"
                stdout = None
            return R()

    session = AshSession(quiet=True)
    session._sandbox = FakeSandbox()
    session._baseline_untracked = {"build/lib/gen.py"}
    session.get_patch()

    staged = [c for c in calls if c.startswith("git add -- ")]
    assert staged, "the agent's new file was never staged by name"
    assert "'repro.py'" in staged[0]
    assert "build/lib/gen.py" not in staged[0], \
        "the image's own untracked file was staged as the agent's work"
