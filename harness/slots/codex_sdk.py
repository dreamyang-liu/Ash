"""codex slot driven through the official SDK (``openai-codex``).

Verified against openai-codex 0.147.0 / codex-cli 0.145.0. The SDK is a typed
wrapper over ``codex app-server`` (JSON-RPC), which is why this replaces the old
``codex exec --json`` driver: that parsed a stdout stream with no version field,
and could not branch a run at all.

What the protocol gives us, and where each piece lands here:

``Codex.thread_fork(thread_id, ...)``
    Branch a thread. Combined with an env snapshot this is a complete rollback
    pair; ``exec --json`` had no equivalent. (The wire call also accepts
    ``lastTurnId`` to fork at a chosen turn -- exposed as :meth:`fork_at`.)
``Thread.turn(input) -> TurnHandle``
    ``TurnHandle.stream()`` yields only this turn's notifications, so the driver
    journals events as they happen rather than reconstructing them afterwards.
``TurnHandle.interrupt()`` / ``.steer(input)``
    Real interrupt and mid-run steering.
``CodexClient(approval_handler=...)``
    ``(method, params) -> decision``. This is the policy seam: the same callback
    shape as claude-code's ``canUseTool``, so one policy serves both slots.
    **The SDK's default handler accepts everything** -- we always pass our own.
    Note the handler is a ``CodexClient`` parameter, not a ``Codex`` one, so this
    driver builds the client itself and wraps it in ``Codex``-equivalent calls
    rather than using ``Codex(config)``: that constructor makes its own client
    and there is no supported way to install a handler afterwards.
``TurnResult.usage``
    Token usage as a typed object instead of a summary line to parse.

Note the SDK spawns the ``codex`` binary itself; ``CodexConfig(codex_bin=...)``
overrides which one. The docs page for the SDK documents only ``run()`` and
``resumeThread()``; fork, steer, interrupt and the approval handler are in the
package but not on that page, so the capability table here is derived from the
installed code, not the docs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from harness.core import events as E
from harness.core.journal import JournalWriter
from harness.core.slot import McpWiring, SlotCapabilities, SlotResult, TaskSpec
from harness.normalize import codex_sdk as normalize
from harness.slots.server_base import DENY, ServerSlot


class CodexSdkSlot(ServerSlot):
    name = "codex"
    binary = "codex"
    capabilities = SlotCapabilities(
        resume=True,
        fork=True,              # Codex.thread_fork
        mcp_stdio=True,
        mcp_remote=True,
        emits_usage=True,
        deny_builtin_tools=False,
    )

    def __init__(self, policy=None) -> None:
        super().__init__(policy)
        self._client: Any = None
        self._turn: Any = None
        self.thread_id: Optional[str] = None
        self._journal: Optional[JournalWriter] = None

    # --- run ---------------------------------------------------------------
    def run(
        self,
        task: TaskSpec,
        journal: JournalWriter,
        mcp: Optional[McpWiring] = None,
    ) -> SlotResult:
        try:
            from openai_codex import CodexConfig, Sandbox, Thread
            from openai_codex.client import CodexClient
        except ImportError as exc:  # pragma: no cover - dependency is optional
            return self._fail(
                journal, "openai-codex is not installed (pip install openai-codex): %s" % exc
            )

        extra = task.extra or {}
        self._journal = journal

        journal.emit(
            E.RUN_STARTED,
            slot=self.name,
            slot_version=self.version(),
            model=task.model,
            task_prompt=task.prompt,
            cwd=task.cwd,
            transport="sdk(app-server jsonrpc)",
            config={"mcp": bool(mcp), "sandbox": extra.get("sandbox")},
        )

        config = CodexConfig(
            cwd=task.cwd,
            env=self._child_env(task),
            config_overrides=self._config_overrides(mcp, extra),
            client_name="ash-harness",
        )

        sandbox = self._sandbox(Sandbox, extra)
        client = CodexClient(
            config=config,
            approval_handler=lambda method, params: self._on_approval(method, params),
        )
        try:
            client.start()
            client.initialize()
            self._client = client
            thread_id = self._open_thread(client, task, extra, sandbox)
            self.thread_id = thread_id
            journal.emit(E.SESSION_REF, native_session_id=thread_id)

            thread = Thread(_client=client, id=thread_id)
            handle = thread.turn(task.prompt, cwd=task.cwd, model=task.model,
                                 sandbox=sandbox, effort=extra.get("effort"))
            self._turn = handle
            # One pass over the stream, journalling as it goes. `handle.run()`
            # would consume the same stream, so calling both deadlocks: the
            # second consumer waits forever for notifications the first already
            # took. Instead the SDK's own collector reads a tee'd iterator.
            result, usage = normalize.collect_and_journal(handle, journal)
            return self._finish(journal, result, usage, thread_id)
        except Exception as exc:  # noqa: BLE001 - report, never propagate
            return self._fail(journal, "%s: %s" % (type(exc).__name__, exc))
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
            self._turn = None

    # --- thread ------------------------------------------------------------
    def _open_thread(self, client, task: TaskSpec, extra: dict, sandbox) -> str:
        """Start, resume or fork; returns the thread id.

        The low-level client takes wire params as a plain dict, which is what we
        want: ``lastTurnId`` (fork at a chosen turn) has no keyword on the
        high-level ``Codex.thread_fork``.
        """
        params: Dict[str, Any] = {"cwd": task.cwd}
        if task.model:
            params["model"] = task.model
        if sandbox is not None:
            params["sandbox"] = getattr(sandbox, "value", sandbox)
        if extra.get("base_instructions"):
            params["baseInstructions"] = extra["base_instructions"]

        resume_id = extra.get("resume_session_id")
        if resume_id and extra.get("fork"):
            # Branch first so the parent thread is untouched by this run.
            if extra.get("fork_turn_id"):
                params["lastTurnId"] = extra["fork_turn_id"]
            response = client.thread_fork(resume_id, params)
            thread_id = _thread_id_of(response)
            if self._journal is not None:
                self._journal.emit(
                    "session.forked",
                    parent_session=resume_id,
                    session_id=thread_id,
                    at_turn=extra.get("fork_turn_id"),
                )
            return thread_id
        if resume_id:
            return _thread_id_of(client.thread_resume(resume_id, params))
        return _thread_id_of(client.thread_start(params))

    def _finish(self, journal, result, usage: dict, thread_id: str) -> SlotResult:
        final_text = ""
        status = "completed"
        error = None
        if result is not None:
            final_text = getattr(result, "final_response", None) or ""
            turn_status = str(getattr(result, "status", "") or "")
            if turn_status and turn_status.lower() not in ("completed", "turnstatus.completed"):
                status = "error" if "fail" in turn_status.lower() else status
            turn_error = getattr(result, "error", None)
            if turn_error is not None:
                error = str(turn_error)
                status = "error"
            typed = normalize.map_usage(getattr(result, "usage", None))
            if any(typed.values()):
                usage = typed

        journal.emit(E.RUN_FINISHED, status=status, usage=usage, error=error)
        return SlotResult(
            status=status,
            final_text=final_text,
            usage=usage,
            native_session_id=thread_id,
            error=error,
        )

    # --- policy ------------------------------------------------------------
    def _on_approval(self, method: str, params: Optional[dict]) -> dict:
        """Answer an approval request through the harness policy callback.

        The SDK's own default accepts everything, so this is the only place a
        deny can happen for codex. Mapping to the protocol's decision vocabulary
        (see CommandExecutionApprovalDecision): ``accept`` / ``decline``.
        """
        kind = _APPROVAL_KINDS.get(method, "tool")
        payload = dict(params or {})
        journal = self._journal
        if journal is None:  # pragma: no cover - run() always sets it
            return {"decision": "accept"}

        verdict, reason = self.decide(kind, dict(payload, method=method), journal)
        if verdict == DENY:
            if method == "item/permissions/requestApproval":
                # This request asks for a permission *profile*; refusing means
                # granting nothing rather than returning a decision string.
                return {"permissions": {}, "scope": "turn"}
            return {"decision": "decline"}
        if method == "item/permissions/requestApproval":
            return {"permissions": payload.get("requestedPermissions") or {}, "scope": "turn"}
        return {"decision": "accept"}

    # --- config ------------------------------------------------------------
    def _child_env(self, task: TaskSpec) -> Dict[str, str]:
        env = dict(os.environ)
        env.update(task.env or {})
        env.setdefault("NO_COLOR", "1")
        return env

    def _config_overrides(self, mcp: Optional[McpWiring], extra: dict) -> Dict[str, Any]:
        """``-c key=value`` equivalents, including the MCP server definition."""
        overrides: Dict[str, Any] = {}
        base = extra.get("config_overrides")
        if isinstance(base, dict):
            overrides.update(base)

        if mcp is not None:
            key = "mcp_servers.%s" % mcp.name
            if mcp.command:
                entry: Dict[str, Any] = {
                    "command": mcp.command[0],
                    "args": list(mcp.command[1:]),
                }
                if mcp.env:
                    entry["env"] = dict(mcp.env)
            elif mcp.url:
                entry = {"url": mcp.url}
                if mcp.headers:
                    entry["http_headers"] = dict(mcp.headers)
            else:
                entry = {}
            if entry:
                overrides[key] = entry
        return overrides

    def _sandbox(self, Sandbox, extra: dict):
        """Map ``extra["sandbox"]`` onto the SDK's preset enum.

        Default: leave it unset so codex applies its own configured policy. The
        harness's isolation boundary is the ash sandbox, not this setting.
        """
        wanted = extra.get("sandbox")
        if not wanted:
            return None
        try:
            return Sandbox(str(wanted))
        except ValueError:
            return None

    # --- extra protocol capabilities ---------------------------------------
    def interrupt(self) -> None:
        if self._turn is not None:
            try:
                self._turn.interrupt()
            except Exception:  # noqa: BLE001
                pass

    def steer(self, text: str) -> None:
        """Inject guidance into the running turn (no restart)."""
        if self._turn is not None:
            self._turn.steer(text)

    def fork_at(self, thread_id: str, last_turn_id: Optional[str] = None, **kwargs):
        """Fork a thread, optionally at a chosen turn. Returns the new id."""
        if self._client is None:
            raise RuntimeError("fork_at requires an open client (call inside run)")
        params: Dict[str, Any] = dict(kwargs)
        if last_turn_id:
            params["lastTurnId"] = last_turn_id
        return _thread_id_of(self._client.thread_fork(thread_id, params))


#: app-server approval methods -> the policy vocabulary shared across slots.
_APPROVAL_KINDS = {
    "item/commandExecution/requestApproval": "command",
    "item/fileChange/requestApproval": "patch",
    "item/permissions/requestApproval": "permission",
    "item/tool/requestUserInput": "tool",
}


def _thread_id_of(response: Any) -> str:
    """Pull the thread id out of a start/resume/fork response.

    The three responses nest it differently across protocol versions, so this
    probes rather than assuming one shape -- an unwrapped id would otherwise
    surface much later as an unusable session reference in the journal.
    """
    for path in (("thread", "id"), ("threadId",), ("thread_id",), ("id",)):
        node: Any = response
        for key in path:
            node = (node.get(key) if isinstance(node, dict) else getattr(node, key, None))
            if node is None:
                break
        if isinstance(node, str) and node:
            return node
    raise RuntimeError("no thread id in response: %r" % (response,))
