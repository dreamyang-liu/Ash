from __future__ import annotations

import base64
import json
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from ash_sandbox.workspace_state import GitWorkspaceFingerprint

from .patch import WORKDIR
from .sandbox import AshSession


# The program runs inside the sandbox. Keeping the state read-only is essential:
# relaxed matching must not change the Git index merely to fingerprint it.
_CONTAINER_FINGERPRINT_PROGRAM = r'''
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
baseline = set(json.loads(sys.argv[2]))

def run(*args):
    p = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace").strip())
    return p.stdout

def frame(label, payload):
    return label + b"\x00" + str(len(payload)).encode("ascii") + b"\x00" + payload

def digest(payload):
    return hashlib.sha256(payload).hexdigest()

head = run("rev-parse", "HEAD").decode("ascii", "replace").strip()
common = ("--binary", "--full-index", "--no-ext-diff", "--no-textconv", "--no-renames")
index_diff = run("diff", *common, "--cached", "HEAD", "--", ".")
worktree_diff = run("diff", *common, "--", ".")
untracked_raw = run("ls-files", "--others", "--exclude-standard", "-z")
rel_paths = sorted(p for p in untracked_raw.split(b"\x00") if p and p.decode("utf-8", "surrogateescape") not in baseline)

records = bytearray()
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
        payload = b""
        kind = f"mode:{stat.S_IFMT(st.st_mode)}".encode("ascii")
    record = (
        frame(b"path", rel_raw)
        + frame(b"kind", kind)
        + frame(b"mode", oct(mode).encode("ascii"))
        + frame(b"content", payload)
    )
    records.extend(frame(b"entry", record))

untracked = bytes(records)
combined = (
    frame(b"head", head.encode("ascii"))
    + frame(b"index", index_diff)
    + frame(b"worktree", worktree_diff)
    + frame(b"untracked", untracked)
)
print(json.dumps({
    "root": str(root),
    "head": head,
    "digest": digest(combined),
    "index_diff_digest": digest(index_diff),
    "worktree_diff_digest": digest(worktree_diff),
    "untracked_digest": digest(untracked),
    "untracked_count": len(rel_paths),
}, sort_keys=True))
'''


@dataclass(frozen=True)
class FilesystemWorkspaceFingerprint:
    root: str
    digest: str
    entry_count: int
    regular_file_bytes: int
    method: str = "filesystem-v1"


_FILESYSTEM_FINGERPRINT_PROGRAM = r'''
set -euo pipefail
root="$1"
cd "$root"
selection=( -mindepth 1 -xdev \( -type d -o -type f -o -type l \) )
digest="$({
  find . "${selection[@]}" -print0 \
    | sort -z \
    | while IFS= read -r -d '' p; do
        mode="$(stat -c '%f:%a:%s' -- "$p")"
        printf 'P\0%s\0M\0%s\0' "$p" "$mode"
        if [ -L "$p" ]; then
          target_hash="$(readlink -z -- "$p" | sha256sum | cut -d' ' -f1)"
          printf 'L\0%s\0' "$target_hash"
        elif [ -f "$p" ]; then
          file_hash="$(sha256sum -- "$p" | cut -d' ' -f1)"
          printf 'F\0%s\0' "$file_hash"
        else
          printf 'D\0'
        fi
      done
} | sha256sum | cut -d' ' -f1)"
entry_count="$(find . "${selection[@]}" -printf '.' | wc -c | tr -d ' ')"
regular_file_bytes="$(find . -mindepth 1 -xdev -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
printf '%s\t%s\t%s\n' "$digest" "$entry_count" "$regular_file_bytes"
'''


def compute_session_filesystem_workspace_fingerprint(
    session: AshSession,
    *,
    workdir: str = WORKDIR,
) -> FilesystemWorkspaceFingerprint:
    """Hash the full visible workspace tree without requiring Git or Python.

    This is the conservative portability fallback for minimal SWE-Marathon
    containers. It includes directory/file mode, regular-file content, symlink
    target, empty directories, and paths. It intentionally ignores timestamps.
    """
    if not workdir.startswith("/") or ".." in PurePosixPath(workdir).parts:
        raise ValueError("workdir must be an absolute normalized sandbox path")
    if not session._sandbox:
        raise RuntimeError("No active sandbox")
    encoded = base64.b64encode(_FILESYSTEM_FINGERPRINT_PROGRAM.encode("utf-8")).decode("ascii")
    command = (
        "printf %s "
        + shlex.quote(encoded)
        + " | base64 -d | bash -s -- "
        + shlex.quote(workdir)
    )
    result = session.execute("shell", {"command": command, "working_dir": workdir})
    if not result.success:
        raise RuntimeError(result.error or result.output or "filesystem fingerprint command failed")
    try:
        digest, count, size = result.output.strip().splitlines()[-1].split("\t")
    except Exception as exc:
        raise RuntimeError(f"invalid filesystem fingerprint output: {result.output!r}") from exc
    if len(digest) != 64:
        raise RuntimeError(f"invalid filesystem digest: {digest!r}")
    return FilesystemWorkspaceFingerprint(
        root=workdir,
        digest=digest,
        entry_count=int(count),
        regular_file_bytes=int(size),
    )


def compute_session_workspace_digest(
    session: AshSession,
    *,
    workdir: str = WORKDIR,
) -> str:
    """Return a conservative workspace digest on both rich and minimal images.

    Prefer the Git-visible fingerprint because it is much cheaper on large coding
    repositories and preserves index/worktree distinctions. If the image lacks
    Python/Git support or the directory is not a Git repository, fall back to a
    full-tree filesystem digest.
    """
    try:
        return compute_session_git_workspace_fingerprint(session, workdir=workdir).digest
    except RuntimeError:
        return compute_session_filesystem_workspace_fingerprint(session, workdir=workdir).digest


def compute_session_git_workspace_fingerprint(
    session: AshSession,
    *,
    workdir: str = WORKDIR,
) -> GitWorkspaceFingerprint:
    """Read a Git-visible workspace fingerprint from the active sandbox.

    The command is intentionally read-only. Baseline untracked paths that shipped
    with the benchmark image are omitted because ``env_fingerprint`` identifies
    that immutable base image separately; any untracked path created after spawn
    remains part of the digest.
    """
    if not workdir.startswith("/") or ".." in PurePosixPath(workdir).parts:
        raise ValueError("workdir must be an absolute normalized sandbox path")
    if not session._sandbox:
        raise RuntimeError("No active sandbox")

    encoded = base64.b64encode(_CONTAINER_FINGERPRINT_PROGRAM.encode("utf-8")).decode("ascii")
    baseline_json = json.dumps(sorted(session._baseline_untracked), ensure_ascii=False)
    command = (
        "python3 -c \"import base64;exec(base64.b64decode('"
        + encoded
        + "'))\" "
        + json.dumps(workdir)
        + " "
        + json.dumps(baseline_json)
    )
    result = session.execute("shell", {"command": command, "working_dir": workdir})
    if not result.success:
        raise RuntimeError(result.error or result.output or "workspace fingerprint command failed")
    try:
        data = json.loads(result.output.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"invalid workspace fingerprint output: {result.output!r}") from exc
    return GitWorkspaceFingerprint(
        root=str(data["root"]),
        head=str(data["head"]),
        digest=str(data["digest"]),
        index_diff_digest=str(data["index_diff_digest"]),
        worktree_diff_digest=str(data["worktree_diff_digest"]),
        untracked_digest=str(data["untracked_digest"]),
        untracked_count=int(data["untracked_count"]),
    )
