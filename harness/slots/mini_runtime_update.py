"""Bring an owned snapshot's runtime to the requested version before agent work."""

import hashlib
from pathlib import Path
import re
import shlex

from harness.execution.templates import RUNTIME_PATH


def _run(session, command):
    result = session.execute("shell", {"command": command, "timeout": 20}, timeout=30)
    outcome = getattr(result, "outcome", None)
    if not result.success or (outcome is not None and (
        outcome.exit_code != 0 or outcome.running or outcome.timed_out or outcome.truncated
    )):
        detail = result.error or (getattr(outcome, "stderr", "") if outcome else "") or result.output
        raise ValueError(f"mini runtime preparation failed: {detail}")
    return outcome.stdout if outcome is not None else result.output


def _fingerprint(session, path):
    text = _run(session, "sha256sum " + path).split()
    if not text or not re.fullmatch(r"[0-9a-f]{64}", text[0]):
        raise ValueError("mini runtime fingerprint response is invalid")
    return text[0]


def ensure_runtime(session, runtime_bin, journal, claim=None, keep=False):
    requested = Path(runtime_bin)
    expected = hashlib.sha256(requested.read_bytes()).hexdigest()
    current = _fingerprint(session, '"/proc/$PPID/exe"')
    if current == expected:
        journal.emit("mini.runtime", sha256=expected, upgraded=False)
        return
    if not session.supports_upload() or not session.supports_snapshot():
        raise ValueError("mini runtime differs from the requested binary and cannot be refreshed safely")
    previous = session.sandbox_id
    temporary = RUNTIME_PATH + "." + expected + ".next"
    if not session.upload_file(requested, temporary):
        raise ValueError("mini runtime upload failed")
    _run(session, f"chmod 0755 {shlex.quote(temporary)} && mv -f {shlex.quote(temporary)} {shlex.quote(RUNTIME_PATH)}")
    if _fingerprint(session, shlex.quote(RUNTIME_PATH)) != expected:
        raise ValueError("mini uploaded runtime checksum differs")
    snapshot = session.snapshot(disk_only=True)
    if snapshot is None:
        raise ValueError("mini runtime preparation snapshot failed")
    if claim is not None:
        claim.snapshot(snapshot.id)

    def register(replacement):
        if claim is not None:
            claim.sandbox(session.sandbox_id, keep=keep)

    session.on_swap.append(register)
    try:
        if not session.swap_sandbox(snapshot):
            raise ValueError("mini runtime re-board failed")
    finally:
        session.on_swap.remove(register)
    if _fingerprint(session, '"/proc/$PPID/exe"') != expected:
        raise ValueError("mini running runtime checksum differs after re-board")
    journal.emit("mini.runtime", sha256=expected, previous_sha256=current, upgraded=True,
                 preparation_snapshot=snapshot.id, previous_sandbox=previous, sandbox_id=session.sandbox_id)
