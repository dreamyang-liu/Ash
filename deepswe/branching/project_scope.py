"""Project-binding snapshots for Shepherd branches.

Shepherd forks the declared workspace binding, not the whole sandbox. This
module contains the backend-independent contract used by benchmark adapters:
one content-addressed project archive plus the exact conversation prefix at a
completed tool boundary. RAM, processes, files outside the binding, and
external effects are intentionally not part of the fork.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path, PurePosixPath
import shlex

UPSTREAM_SHEPHERD_EXPERIMENTS_COMMIT = "c12ebd1b774cf12f70ef2b4486e61e7052f3e3ab"


@dataclass(frozen=True)
class ProjectBindingSnapshot:
    workdir: str
    checkpoint_step: int
    archive: str
    archive_sha256: str
    archive_bytes: int
    conversation_session_id: str
    conversation_cut: str
    backend: str = "project-binding-tar-v1"

    def to_dict(self) -> dict:
        return asdict(self)


def checked_workdir(value: str) -> str:
    """Return a safe absolute project root suitable for replacement."""
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Project binding must be an absolute normalized path")
    normalized = str(path)
    if normalized in {"/", "/bin", "/boot", "/dev", "/etc", "/home",
                      "/proc", "/root", "/run", "/sys", "/tmp", "/usr", "/var"}:
        raise ValueError("Refusing a system-wide project binding: " + normalized)
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_command(workdir: str, remote_archive: str) -> str:
    """Build a command that captures only the declared project binding."""
    root = checked_workdir(workdir)
    archive = PurePosixPath(remote_archive)
    if not archive.is_absolute():
        raise ValueError("Remote archive path must be absolute")
    return "tar -czf %s -C %s ." % (shlex.quote(str(archive)), shlex.quote(root))


def restore_commands(workdir: str, remote_archive: str) -> tuple[str, str]:
    """Build checked commands that replace, then restore, one project root."""
    root = checked_workdir(workdir)
    archive = PurePosixPath(remote_archive)
    if not archive.is_absolute():
        raise ValueError("Remote archive path must be absolute")
    prepare = "mkdir -p %s && find %s -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +" % (
        shlex.quote(root), shlex.quote(root))
    restore = "tar -xzf %s -C %s" % (shlex.quote(str(archive)), shlex.quote(root))
    return prepare, restore


def verify_snapshot(snapshot: dict) -> ProjectBindingSnapshot:
    """Validate a durable snapshot manifest and its local archive."""
    required = set(ProjectBindingSnapshot.__dataclass_fields__)
    missing = required - set(snapshot)
    if missing:
        raise ValueError("Project snapshot manifest is incomplete: " + ", ".join(sorted(missing)))
    result = ProjectBindingSnapshot(**{key: snapshot[key] for key in required})
    checked_workdir(result.workdir)
    if type(result.checkpoint_step) is not int or result.checkpoint_step < 0:
        raise ValueError("Invalid project checkpoint step")
    archive = Path(result.archive)
    if not archive.is_file() or archive.stat().st_size != result.archive_bytes:
        raise ValueError("Project snapshot archive is missing or has the wrong size")
    if sha256_file(archive) != result.archive_sha256:
        raise ValueError("Project snapshot archive hash mismatch")
    if result.backend != "project-binding-tar-v1":
        raise ValueError("Unsupported project snapshot backend")
    return result
