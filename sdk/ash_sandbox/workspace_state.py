from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _run_git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    return proc.stdout


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frame(label: bytes, payload: bytes) -> bytes:
    return label + b"\x00" + str(len(payload)).encode("ascii") + b"\x00" + payload


@dataclass(frozen=True)
class GitWorkspaceFingerprint:
    """Content fingerprint of Git-visible coding state.

    The digest includes HEAD identity, staged state, unstaged tracked state, and
    untracked file/symlink contents. It intentionally does not claim to capture
    installed packages, running processes, environment variables, or other state
    outside the repository; relaxed matching must pair it with an exact external
    side-effect/barrier hash.
    """

    root: str
    head: str
    digest: str
    index_diff_digest: str
    worktree_diff_digest: str
    untracked_digest: str
    untracked_count: int


def compute_git_workspace_fingerprint(workdir: str | Path) -> GitWorkspaceFingerprint:
    path = Path(workdir).expanduser().resolve()
    top = _run_git(path, "rev-parse", "--show-toplevel").decode("utf-8", "replace").strip()
    root = Path(top).resolve()
    head = _run_git(root, "rev-parse", "HEAD").decode("ascii", "replace").strip()

    common = ("--binary", "--full-index", "--no-ext-diff", "--no-textconv", "--no-renames")
    index_diff = _run_git(root, "diff", *common, "--cached", "HEAD", "--", ".")
    worktree_diff = _run_git(root, "diff", *common, "--", ".")
    untracked_raw = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    rel_paths = [p for p in untracked_raw.split(b"\x00") if p]
    rel_paths.sort()

    untracked_records = bytearray()
    for rel_raw in rel_paths:
        rel = rel_raw.decode("utf-8", "surrogateescape")
        full = root / rel
        st = full.lstat()
        mode = stat.S_IMODE(st.st_mode)
        if stat.S_ISLNK(st.st_mode):
            payload = os.readlink(full).encode("utf-8", "surrogateescape")
            kind = b"symlink"
        elif stat.S_ISREG(st.st_mode):
            payload = full.read_bytes()
            kind = b"file"
        else:
            # Git's untracked listing normally returns files/symlinks, not
            # directories. Keep unusual node types distinct without reading them.
            payload = b""
            kind = f"mode:{stat.S_IFMT(st.st_mode)}".encode("ascii")
        record = (
            _frame(b"path", rel_raw)
            + _frame(b"kind", kind)
            + _frame(b"mode", oct(mode).encode("ascii"))
            + _frame(b"content", payload)
        )
        untracked_records.extend(_frame(b"entry", record))

    untracked_bytes = bytes(untracked_records)
    combined = (
        _frame(b"head", head.encode("ascii"))
        + _frame(b"index", index_diff)
        + _frame(b"worktree", worktree_diff)
        + _frame(b"untracked", untracked_bytes)
    )
    return GitWorkspaceFingerprint(
        root=str(root),
        head=head,
        digest=_digest(combined),
        index_diff_digest=_digest(index_diff),
        worktree_diff_digest=_digest(worktree_diff),
        untracked_digest=_digest(untracked_bytes),
        untracked_count=len(rel_paths),
    )
