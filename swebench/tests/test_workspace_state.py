from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from ash_sandbox.workspace_state import compute_git_workspace_fingerprint
from swebench.workspace_state import compute_session_git_workspace_fingerprint


class _LocalSession:
    def __init__(self, baseline=()):
        self._sandbox = object()
        self._baseline_untracked = set(baseline)

    def execute(self, tool_name, args):
        assert tool_name == "shell"
        proc = subprocess.run(
            args["command"],
            shell=True,
            cwd=args.get("working_dir") or None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return SimpleNamespace(success=proc.returncode == 0, output=proc.stdout, error="" if proc.returncode == 0 else proc.stdout)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "a.txt").write_text("one\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    return repo


def test_session_fingerprint_matches_local_reference_for_clean_repo(tmp_path):
    repo = _repo(tmp_path)
    local = compute_git_workspace_fingerprint(repo)
    remote = compute_session_git_workspace_fingerprint(_LocalSession(), workdir=str(repo))
    assert remote == local


def test_session_fingerprint_tracks_staged_unstaged_and_untracked_state(tmp_path):
    repo = _repo(tmp_path)
    session = _LocalSession()
    clean = compute_session_git_workspace_fingerprint(session, workdir=str(repo))

    (repo / "a.txt").write_text("staged\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    staged = compute_session_git_workspace_fingerprint(session, workdir=str(repo))
    assert staged.digest != clean.digest
    assert staged.index_diff_digest != clean.index_diff_digest

    (repo / "a.txt").write_text("unstaged\n")
    unstaged = compute_session_git_workspace_fingerprint(session, workdir=str(repo))
    assert unstaged.digest != staged.digest
    assert unstaged.worktree_diff_digest != staged.worktree_diff_digest

    (repo / "new.txt").write_text("new\n")
    untracked = compute_session_git_workspace_fingerprint(session, workdir=str(repo))
    assert untracked.digest != unstaged.digest
    assert untracked.untracked_count == 1


def test_baseline_untracked_paths_are_omitted_but_new_paths_are_not(tmp_path):
    repo = _repo(tmp_path)
    (repo / "image-baggage.txt").write_text("base\n")
    session = _LocalSession(baseline={"image-baggage.txt"})
    baseline = compute_session_git_workspace_fingerprint(session, workdir=str(repo))
    assert baseline.untracked_count == 0

    (repo / "agent-file.txt").write_text("agent\n")
    changed = compute_session_git_workspace_fingerprint(session, workdir=str(repo))
    assert changed.untracked_count == 1
    assert changed.digest != baseline.digest
