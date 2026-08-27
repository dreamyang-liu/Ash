"""Shared machinery for slots that drive an agent's *server* protocol.

``codex app-server`` (JSON-RPC over stdio) and ``opencode serve`` (HTTP + SSE)
both expose a versioned protocol with capabilities the one-shot CLIs do not:

    fork at a chosen point · rollback · resume · interrupt · mid-run steering
    approval requests the driver answers (the policy seam)
    token usage as notifications rather than a summary line

That is why the slots use them instead of ``codex exec --json`` /
``opencode run --format json``: those write to stdout in formats their own docs
call unstable, and neither can branch a run.

The two protocols are different enough that this module only holds what is
genuinely common -- process lifetime with a watchdog (a server that stops
emitting must not strand a batch), and a place for a policy callback -- while
each slot implements its own transport.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

from harness.core.journal import JournalWriter
from harness.core.slot import AgentSlot, McpWiring, SlotResult, TaskSpec

#: Verdict a policy callback returns for one requested action.
#: ``allow`` / ``deny`` mirror the interceptor pipeline's Continue / Reject, so a
#: single policy can serve every slot regardless of how its protocol spells it.
Verdict = str
ALLOW: Verdict = "allow"
DENY: Verdict = "deny"

#: ``(kind, payload) -> (verdict, reason)``. ``kind`` is normalized:
#: "command" | "patch" | "permission" | "tool".
PolicyCallback = Callable[[str, dict], "tuple[Verdict, Optional[str]]"]


def allow_all(kind: str, payload: dict):
    """Default policy: the sandbox is the boundary, not the agent's manners."""
    return ALLOW, None


class ServerProcess:
    """A managed agent-server subprocess.

    Owns the watchdog: reading a child's stream blocks, so a server that hangs
    without emitting would never be reclaimed and would stall a whole batch.
    """

    def __init__(
        self,
        command: List[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        stdin: bool = True,
        stderr_to: Optional[int] = None,
    ) -> None:
        self.command = list(command)
        self.proc = subprocess.Popen(
            self.command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_to if stderr_to is not None else subprocess.PIPE,
            text=False,
            bufsize=0,
            start_new_session=True,
        )
        self._stderr_tail: List[str] = []
        if self.proc.stderr is not None:
            threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._stderr_tail.append(line)
                del self._stderr_tail[:-40]

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail[-12:])

    def alive(self) -> bool:
        return self.proc.poll() is None

    def write_line(self, payload: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(payload + b"\n")
        self.proc.stdin.flush()

    def terminate(self, grace_s: float = 10.0) -> None:
        """SIGTERM the process group, then SIGKILL. Servers spawn children."""
        if self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self.proc.terminate()
        deadline = time.time() + grace_s
        while time.time() < deadline and self.proc.poll() is None:
            time.sleep(0.1)
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()


class ServerSlot(AgentSlot):
    """Base for protocol-driven slots.

    Subclasses implement :meth:`run`; this provides the shared process handle,
    the policy hook and a uniform timeout story.
    """

    def __init__(self, policy: Optional[PolicyCallback] = None) -> None:
        self.policy: PolicyCallback = policy or allow_all
        self._server: Optional[ServerProcess] = None
        self._killed = threading.Event()

    # --- policy ------------------------------------------------------------
    def decide(self, kind: str, payload: dict, journal: JournalWriter):
        """Apply the policy callback and journal the verdict.

        Recorded even when it allows: a trajectory that does not say a call was
        gated cannot be compared with one where nothing was.
        """
        try:
            verdict, reason = self.policy(kind, payload)
        except Exception as exc:  # noqa: BLE001 - a broken policy must not kill the run
            verdict, reason = ALLOW, "policy raised %s" % type(exc).__name__
        journal.emit(
            "policy.verdict",
            kind=kind,
            verdict=verdict,
            reason=reason,
            request=payload,
        )
        return verdict, reason

    # --- lifecycle ---------------------------------------------------------
    def kill(self) -> None:
        self._killed.set()
        if self._server is not None:
            self._server.terminate()

    def _fail(self, journal: JournalWriter, message: str) -> SlotResult:
        journal.emit("agent.error", message=message)
        return SlotResult(status="error", error=message)

    def build_env(self, task: TaskSpec) -> Dict[str, str]:
        env = dict(os.environ)
        env.update(task.env or {})
        # Colour codes in a machine-read stream are noise at best.
        env.setdefault("NO_COLOR", "1")
        return env


def find_free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def log_stderr(prefix: str, text: str) -> None:  # pragma: no cover - diagnostics
    if text:
        sys.stderr.write("[%s] %s\n" % (prefix, text))
        sys.stderr.flush()
