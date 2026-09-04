import subprocess

from ash_sandbox.workspace_state import compute_git_workspace_fingerprint
from ash_sandbox.relaxed_prefix import workspace_convergence_key


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("one\n")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-qm", "init")
    return repo


def tr(tool, args, content="ok", success=True):
    return {"role": "tool_result", "tool_name": tool, "tool_args": args, "content": content, "success": success}


def test_git_workspace_fingerprint_is_content_sensitive_and_deterministic(tmp_path):
    repo = init_repo(tmp_path)
    clean = compute_git_workspace_fingerprint(repo)
    assert clean.digest == compute_git_workspace_fingerprint(repo).digest
    (repo / "a.txt").write_text("two\n")
    modified = compute_git_workspace_fingerprint(repo)
    assert modified.digest != clean.digest
    (repo / "a.txt").write_text("one\n")
    assert compute_git_workspace_fingerprint(repo).digest == clean.digest


def test_git_workspace_fingerprint_includes_untracked_and_index_state(tmp_path):
    repo = init_repo(tmp_path)
    clean = compute_git_workspace_fingerprint(repo)
    (repo / "new.txt").write_text("x")
    untracked = compute_git_workspace_fingerprint(repo)
    assert untracked.digest != clean.digest and untracked.untracked_count == 1
    git(repo, "add", "new.txt")
    staged = compute_git_workspace_fingerprint(repo)
    assert staged.digest != untracked.digest


def test_different_file_edit_histories_can_converge_only_with_same_workspace_digest(tmp_path):
    repo = init_repo(tmp_path)
    (repo / "a.txt").write_text("final\n")
    final = compute_git_workspace_fingerprint(repo).digest
    history_a = [tr("text_editor", {"command": "write", "path": "/testbed/a.txt", "file_text": "final\n"}, "done")]
    history_b = [
        tr("text_editor", {"command": "str_replace", "path": "/testbed/a.txt", "old_str": "one", "new_str": "tmp"}, "done"),
        tr("text_editor", {"command": "str_replace", "path": "/testbed/a.txt", "old_str": "tmp", "new_str": "final"}, "done"),
    ]
    ka = workspace_convergence_key(env_fingerprint="env-1", workspace_digest=final, messages=history_a)
    kb = workspace_convergence_key(env_fingerprint="env-1", workspace_digest=final, messages=history_b)
    assert ka == kb


def test_external_barrier_prevents_false_convergence(tmp_path):
    repo = init_repo(tmp_path)
    digest = compute_git_workspace_fingerprint(repo).digest
    base = [tr("text_editor", {"command": "view", "path": "/testbed/a.txt"}, "one")]
    with_install = base + [tr("shell", {"command": "pip install x"}, "installed")]
    k1 = workspace_convergence_key(env_fingerprint="env-1", workspace_digest=digest, messages=base)
    k2 = workspace_convergence_key(env_fingerprint="env-1", workspace_digest=digest, messages=with_install)
    assert k1 != k2
