"""Snapshot -> model.patch -> their verifier -> ``Grade``.

Two microVMs per verdict, mirroring Pier's ``environment_mode = "separate"``:

1. **Collect.** The attempt's last snapshot is restored and the task's own
   ``[[verifier.collect]]`` command runs in it -- ``git diff --binary <base>
   HEAD > /logs/artifacts/model.patch``. Only committed work is in that diff;
   that is the benchmark's rule, not ours, so the uncommitted state is merely
   *reported* (it is the first thing an analyst should know).
2. **Verify.** A fresh VM from the task's pristine image, offline, with the
   task's declared shape. ``tests/Dockerfile`` is replayed on it (COPY the
   verifier files, RUN its steps), model.patch is placed where the collect
   step would have left it, and ``bash /tests/test.sh`` runs. It writes
   ``/logs/verifier/reward.json``; that file is the verdict.

Nothing here interprets tests. ``grader.py`` is theirs and decides everything;
we read what it wrote.
"""

from __future__ import annotations

import json
import posixpath
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harness.execution.session import SandboxSession
from swebench.fork_eval import Grade
from deepswe.tasks import Task

ARTIFACTS_DIR = "/logs/artifacts"
VERIFIER_DIR = "/logs/verifier"
MODEL_PATCH = ARTIFACTS_DIR + "/model.patch"
VERIFIER_ENTRYPOINT = "bash /tests/test.sh"


def shell_parts(result) -> Tuple[str, str, int]:
    """``(stdout, stderr, exit_code)`` from a ``shell`` ToolResult.

    The runtime returns bare stdout on a clean exit and a JSON envelope
    (stdout/stderr/exit_code) when there is anything else to say. Reading
    ``result.success`` alone would score a failing verifier as a pass.
    """
    text = result.output or ""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and "stdout" in payload:
        return (str(payload.get("stdout") or ""), str(payload.get("stderr") or ""),
                int(payload.get("exit_code") or 0))
    if not result.success:
        return "", str(result.error or text), 1
    return text, "", 0


def _sh(session, command: str, timeout_s: float):
    return session.execute("shell", {"command": command, "timeout": int(timeout_s)},
                           timeout=timeout_s + 60)


def _cat(session, path: str, timeout_s: float = 120) -> Optional[str]:
    out, _, rc = shell_parts(_sh(session, "cat %s" % path, timeout_s))
    return out if rc == 0 else None


# --- 1. collect --------------------------------------------------------------

def collect_patch(session, task: Task) -> Tuple[str, Dict[str, object]]:
    """Run the task's collect step(s) in the restored snapshot; return the patch.

    Also reports the repository state -- branch, HEAD, how many paths are
    uncommitted -- because "the agent did the work and never committed" is a
    distinct failure from "the agent did the wrong work", and the verdict
    alone cannot tell them apart.
    """
    diagnostics: Dict[str, object] = {}
    for step in task.collect:
        out, err, rc = shell_parts(_sh(session, step.command, step.timeout_s))
        if rc != 0:
            diagnostics["collect_error"] = (err or out)[-2000:]
    patch = _cat(session, MODEL_PATCH) or ""
    state, _, rc = shell_parts(_sh(
        session,
        "cd /app && git status --porcelain | wc -l && "
        "git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && "
        "git log --oneline %s..HEAD | wc -l" % (task.base_commit or "HEAD"),
        120))
    if rc == 0:
        lines = state.split("\n")
        if len(lines) >= 4:
            diagnostics["uncommitted_paths"] = int(lines[0].strip() or 0)
            diagnostics["branch"] = lines[1].strip()
            diagnostics["head"] = lines[2].strip()
            diagnostics["commits_since_base"] = int(lines[3].strip() or 0)
    return patch, diagnostics


# --- 2. verify ---------------------------------------------------------------

@dataclass
class VerifierOutcome:
    reward: Optional[dict] = None
    output: str = ""
    error: Optional[str] = None
    failed_tests: List[str] = field(default_factory=list)
    exit_code: int = 0


def _place(session, source: Path, destination: str) -> bool:
    """Put a host file into the VM byte-exact; text_editor is the fallback."""
    if session.upload_file(source, destination):
        return True
    try:
        text = Path(source).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    result = session.execute("text_editor", {"command": "write",
                                             "path": destination,
                                             "file_text": text})
    return bool(result.success)


def _failed_from_ctrf(text: Optional[str], limit: int = 25) -> List[str]:
    """Names of non-passing tests in the ctrf.json their grader synthesises."""
    if not text:
        return []
    try:
        doc = json.loads(text)
    except ValueError:
        return []
    tests = ((doc.get("results") or {}).get("tests") or []) if isinstance(doc, dict) else []
    out: List[str] = []
    for entry in tests:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "").lower() != "passed":
            name = str(entry.get("name") or "")
            suite = str(entry.get("suite") or "")
            out.append("%s:%s" % (suite, name) if suite else name)
        if len(out) >= limit:
            break
    return out


def verify_patch(task: Task, patch: str, backend: dict) -> VerifierOutcome:
    """Replay tests/Dockerfile on a pristine offline VM and run the verifier."""
    outcome = VerifierOutcome()
    session = SandboxSession(quiet=True, backend=dict(backend))
    if not session.create(task.image, {"cpu": task.cpus, "memory_mb": task.memory_mb}):
        outcome.error = "could not start verifier VM from %s: %s" % (
            task.image, session.create_error)
        return outcome
    try:
        dirs = {posixpath.dirname(f.destination) for f in task.verifier_files}
        dirs |= {ARTIFACTS_DIR, VERIFIER_DIR}
        _sh(session, "mkdir -p %s" % " ".join(sorted(d for d in dirs if d)), 60)
        for item in task.verifier_files:
            if not _place(session, item.source, item.destination):
                outcome.error = "could not place %s at %s" % (item.source.name,
                                                             item.destination)
                return outcome
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False,
                                         encoding="utf-8", newline="") as tmp:
            tmp.write(patch)
            tmp_path = Path(tmp.name)
        try:
            if not _place(session, tmp_path, MODEL_PATCH):
                outcome.error = "could not place model.patch"
                return outcome
        finally:
            tmp_path.unlink(missing_ok=True)
        for command in task.verifier_run_steps:
            _, err, rc = shell_parts(_sh(session, command, 300))
            if rc != 0:
                outcome.error = "verifier Dockerfile RUN failed (%s): %s" % (command, err[-500:])
                return outcome

        result = _sh(session, VERIFIER_ENTRYPOINT, task.verifier_timeout_s)
        out, err, rc = shell_parts(result)
        outcome.exit_code = rc
        outcome.output = out + (("\n[stderr]\n" + err) if err else "")

        reward_text = _cat(session, VERIFIER_DIR + "/reward.json")
        if reward_text is None:
            sentinel = _cat(session, VERIFIER_DIR + "/reward.txt")
            outcome.error = ("verifier produced no reward.json (exit %d%s)"
                             % (rc, ", reward.txt=%s" % sentinel.strip()
                                if sentinel else ""))
            return outcome
        try:
            outcome.reward = json.loads(reward_text)
        except ValueError:
            outcome.error = "reward.json is not JSON: %r" % reward_text[:200]
            return outcome
        outcome.failed_tests = _failed_from_ctrf(_cat(session, VERIFIER_DIR + "/ctrf.json"))
    except Exception as exc:  # noqa: BLE001 - an ungradeable attempt is a zero
        outcome.error = "%s: %s" % (type(exc).__name__, exc)
    finally:
        session.destroy()
    return outcome


# --- 3. verdict --------------------------------------------------------------

def grade_from_verifier(patch: str, outcome: VerifierOutcome,
                        diagnostics: Optional[Dict[str, object]] = None) -> Grade:
    """Their reward.json -> the loop's ``Grade``. Pure, so it is testable.

    ``reward`` is binary and authoritative. f2p/p2p fractions fill the fields the
    analysts read; ``apply_failed`` means the patch never landed, which is a
    failed attempt rather than a grading error (the grader said so itself).
    """
    grade = Grade(patch=patch)
    notes = []
    diagnostics = diagnostics or {}
    if diagnostics:
        notes.append("repo state: branch=%s, %s commit(s) since base, %s uncommitted path(s)%s"
                     % (diagnostics.get("branch", "?"),
                        diagnostics.get("commits_since_base", "?"),
                        diagnostics.get("uncommitted_paths", "?"),
                        " -- UNCOMMITTED WORK IS NOT GRADED"
                        if diagnostics.get("uncommitted_paths") else ""))
        if diagnostics.get("collect_error"):
            notes.append("collect step failed: %s" % diagnostics["collect_error"])
    if outcome.error and outcome.reward is None:
        grade.error = outcome.error
        grade.detail = "\n".join(notes + ["", "verifier output (tail):",
                                          outcome.output[-4000:]])
        return grade

    reward = outcome.reward or {}
    binary = int(reward.get("reward") or 0)
    f2p_total = int(reward.get("f2p_total") or 0)
    f2p_passed = int(reward.get("f2p_passed") or 0)
    p2p_total = int(reward.get("p2p_total") or 0)
    p2p_passed = int(reward.get("p2p_passed") or 0)
    apply_failed = bool(reward.get("apply_failed"))

    grade.resolved = binary == 1
    grade.f2p_pass = f2p_total > 0 and f2p_passed == f2p_total
    grade.p2p_ran = not apply_failed
    grade.p2p_pass = grade.p2p_ran and p2p_passed == p2p_total
    grade.broken = [t for t in outcome.failed_tests if t.startswith("p2p:")]
    if apply_failed:
        notes.append("model.patch did not apply to the pristine base")
    elif not patch.strip():
        notes.append("empty model.patch: nothing committed on top of the base")
    notes.append("reward.json: %s" % json.dumps(reward, sort_keys=True))
    if outcome.failed_tests:
        notes.append("non-passing tests (%d shown): %s"
                     % (len(outcome.failed_tests), ", ".join(outcome.failed_tests)))
    grade.detail = "\n".join(notes + ["", "verifier output (tail):",
                                      outcome.output[-4000:]])
    return grade


def grade_snapshot(snapshot_id: str, task: Task, backend: dict) -> Grade:
    """The whole path: restore, collect, verify on a pristine VM, read reward."""
    session = SandboxSession(quiet=True, backend=dict(backend))
    if not session.create(snapshot_id):
        return Grade(error="could not restore %s: %s" % (snapshot_id,
                                                         session.create_error))
    try:
        patch, diagnostics = collect_patch(session, task)
    except Exception as exc:  # noqa: BLE001
        return Grade(error="collect failed: %s: %s" % (type(exc).__name__, exc))
    finally:
        session.destroy()
    return grade_from_verifier(patch, verify_patch(task, patch, backend), diagnostics)
