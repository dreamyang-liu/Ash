"""Shared driver for slots that are headless CLI processes emitting JSONL.

Covers codex (``codex exec --json``) and opencode (``opencode run --format
json``). claude-code does not use this path: it goes through the Agent SDK,
which gives typed messages plus hooks/canUseTool.

Responsibilities kept here (identical across CLI slots):
- spawn, stream stdout line-by-line, parse JSON, hand each event to the slot's
  normalizer, append results to the journal;
- keep stderr in the journal too (``slot.log``) -- upstream CLIs report auth and
  config failures there, and losing them makes debugging a batch run miserable;
- accumulate usage; derive final text; enforce timeout; best-effort kill.

Non-JSON stdout lines are journaled as ``slot.log`` rather than dropped: a CLI
that starts printing banners must be *visible* in the trajectory, not silently
skipped (that is how normalizer drift goes unnoticed).
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Callable, List, Optional, Tuple

from harness.core.events import (
    AGENT_MESSAGE,
    RUN_FINISHED,
    RUN_RESULT,
    RUN_STARTED,
    SESSION_REF,
    SLOT_LOG,
    TURN_COMPLETED,
    Usage,
)
from harness.core.journal import JournalWriter
from harness.core.slot import AgentSlot, McpWiring, SlotResult, TaskSpec

Normalizer = Callable[[dict], List[Tuple[str, dict]]]


class JsonlCliSlot(AgentSlot):
    """Base class: subclasses provide ``build_command`` and ``normalizer``."""

    #: set by subclasses
    normalizer: Optional[Normalizer] = None

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None

    # --- to implement -----------------------------------------------------
    def build_command(self, task: TaskSpec, mcp: Optional[McpWiring]) -> List[str]:
        raise NotImplementedError

    def build_env(self, task: TaskSpec, mcp: Optional[McpWiring]) -> dict:
        import os

        env = os.environ.copy()
        env.update(task.env)
        if mcp and mcp.env:
            env.update(mcp.env)
        return env

    def stdin_payload(self, task: TaskSpec) -> Optional[str]:
        """Return prompt text if the CLI reads it from stdin, else None."""
        return None

    # --- driver -----------------------------------------------------------
    def run(
        self,
        task: TaskSpec,
        journal: JournalWriter,
        mcp: Optional[McpWiring] = None,
    ) -> SlotResult:
        command = self.build_command(task, mcp)
        journal.emit(
            RUN_STARTED,
            slot=self.name,
            slot_version=self.version(),
            model=task.model,
            cwd=task.cwd,
            task_prompt=task.prompt,
            config={"command": command, "mcp": _mcp_summary(mcp)},
        )

        usage = Usage()
        session_id: Optional[str] = None
        last_text = ""
        result_text = ""
        stdin_text = self.stdin_payload(task)

        try:
            self._proc = subprocess.Popen(
                command,
                cwd=task.cwd,
                env=self.build_env(task, mcp),
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            journal.emit(RUN_FINISHED, status="error", error=str(exc), usage=usage.as_dict())
            return SlotResult(status="error", error=str(exc), usage=usage.as_dict())

        proc = self._proc
        stderr_lines: List[str] = []
        stderr_thread = threading.Thread(
            target=_drain_stderr, args=(proc, journal, stderr_lines), daemon=True
        )
        stderr_thread.start()

        if stdin_text is not None and proc.stdin:
            try:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            except BrokenPipeError:  # pragma: no cover - CLI died early
                pass

        # A watchdog, not a wait() timeout: reading stdout blocks, so a CLI that
        # hangs *without* emitting anything would otherwise never be reclaimed
        # and would stall an entire batch run.
        timed_out = threading.Event()

        def on_deadline() -> None:
            timed_out.set()
            self.kill()

        watchdog = threading.Timer(task.timeout_s, on_deadline) if task.timeout_s else None
        if watchdog is not None:
            watchdog.daemon = True
            watchdog.start()

        try:
            for line in proc.stdout or ():
                line = line.strip()
                if not line:
                    continue
                try:
                    native = json.loads(line)
                except ValueError:
                    journal.emit(SLOT_LOG, stream="stdout", text=line[:4000])
                    continue
                for event_type, payload in self.normalizer(native):
                    journal.emit(event_type, **payload)
                    if event_type == TURN_COMPLETED and payload.get("usage"):
                        usage.add_dict(payload["usage"])
                    elif event_type == SESSION_REF and payload.get("native_session_id"):
                        session_id = payload["native_session_id"]
                    elif event_type == AGENT_MESSAGE and payload.get("text"):
                        last_text = payload["text"]
                    elif event_type == RUN_RESULT and payload.get("text"):
                        result_text = payload["text"]
            exit_code = proc.wait()
        finally:
            if watchdog is not None:
                watchdog.cancel()
            stderr_thread.join(timeout=5)

        status = (
            "timeout" if timed_out.is_set() else ("completed" if exit_code == 0 else "error")
        )
        error = None
        if status == "error":
            error = "exit code %s" % exit_code
            tail = [ln for ln in stderr_lines[-5:] if ln]
            if tail:
                error += ": " + " | ".join(tail)

        final_text = result_text or last_text
        journal.emit(
            RUN_FINISHED,
            status=status,
            exit_code=exit_code,
            usage=usage.as_dict(),
            error=error,
        )
        return SlotResult(
            status=status,
            final_text=final_text,
            usage=usage.as_dict(),
            native_session_id=session_id,
            exit_code=exit_code,
            error=error,
        )

    def kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            proc.kill()


def _drain_stderr(proc: subprocess.Popen, journal: JournalWriter, sink: List[str]) -> None:
    for line in proc.stderr or ():
        line = line.rstrip()
        if not line:
            continue
        sink.append(line)
        journal.emit(SLOT_LOG, stream="stderr", text=line[:4000])


def _mcp_summary(mcp: Optional[McpWiring]) -> Optional[dict]:
    if mcp is None:
        return None
    return {"name": mcp.name, "command": mcp.command, "url": mcp.url}
