"""opencode slot driven through ``opencode serve`` (HTTP + SSE).

Verified against opencode 1.18.5, whose server publishes an OpenAPI 3.1 document
at ``GET /doc`` -- a real contract, unlike the stdout of ``opencode run``.

What the server gives us that the CLI does not:

``POST /session/{id}/fork {messageID}``
    Branch a conversation **at a chosen message**, not merely continue it. This
    is the conversation half of a rollback pair at an arbitrary step; the CLI's
    ``--fork`` could only branch from the end.
``POST /session/{id}/revert {messageID, partID}`` / ``/unrevert``
    Roll a session back in place.
``GET /permission`` + ``POST /permission/{requestID}/reply {reply}``
    The policy seam: the agent asks, we answer ``once`` / ``always`` / ``reject``.
    ``reject`` carries a message the model reads, so a denial can explain itself.
``POST /session/{id}/abort``
    Interrupt a running turn.
``GET /api/session/{id}/event`` (SSE)
    Per-session event stream, so a driver does not have to filter a global one.

Wiring notes:
- One server per slot on an ephemeral port. Sharing a server across slots would
  share the session database and the permission queue.
- ``--pure`` skips a developer's local plugins (same intent as ``claude --bare``).
- MCP still travels through the config file: ``$OPENCODE_CONFIG``. The server
  reads it at startup, so it must be written before the process starts.
- **Concurrency needs a per-lane data dir.** Sessions live in SQLite under
  ``$XDG_DATA_HOME/opencode``; several servers sharing it fail with "database is
  locked". Scope ``extra["data_home"]`` per *task*, not per attempt -- a fork
  must find the session an earlier run wrote.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from harness.core import events as E
from harness.core.journal import JournalWriter
from harness.core.slot import McpWiring, SlotCapabilities, SlotResult, TaskSpec
from harness.normalize import opencode_server as normalize
from harness.slots.server_base import (DENY, ServerProcess, ServerSlot,
                                       find_free_port)


class OpenCodeServerSlot(ServerSlot):
    name = "opencode"
    binary = "opencode"
    capabilities = SlotCapabilities(
        resume=True,
        fork=True,             # POST /session/{id}/fork {messageID} — at a step
        mcp_stdio=True,
        mcp_remote=True,
        emits_usage=True,
        deny_builtin_tools=False,
    )

    def __init__(self, policy=None) -> None:
        super().__init__(policy)
        self._config_path: Optional[str] = None
        self.base_url: Optional[str] = None
        self.session_id: Optional[str] = None

    # --- run ---------------------------------------------------------------
    def run(
        self,
        task: TaskSpec,
        journal: JournalWriter,
        mcp: Optional[McpWiring] = None,
    ) -> SlotResult:
        extra = task.extra or {}
        port = find_free_port()
        self.base_url = "http://127.0.0.1:%d" % port

        journal.emit(
            E.RUN_STARTED,
            slot=self.name,
            slot_version=self.version(),
            model=task.model,
            task_prompt=task.prompt,
            cwd=task.cwd,
            transport="http+sse",
            config={"port": port, "mcp": bool(mcp)},
        )

        command = ["opencode", "serve", "--hostname", "127.0.0.1", "--port", str(port)]
        if not extra.get("allow_plugins"):
            command.append("--pure")

        env = self.build_env(task)
        config = self._render_config(mcp, task)
        if config:
            env["OPENCODE_CONFIG"] = config
        data_home = extra.get("data_home")
        if data_home:
            os.makedirs(str(data_home), exist_ok=True)
            env["XDG_DATA_HOME"] = str(data_home)

        self._server = ServerProcess(command, cwd=task.cwd, env=env, stdin=False)
        deadline = time.time() + float(task.timeout_s)
        try:
            if not self._wait_ready(deadline):
                return self._fail(
                    journal,
                    "opencode serve did not become ready: %s" % self._server.stderr_tail,
                )
            return self._drive(task, journal, deadline)
        finally:
            self._server.terminate()
            self._cleanup_config()

    # --- HTTP --------------------------------------------------------------
    def _rpc(self, method: str, path: str, body: Optional[dict] = None,
             timeout: float = 300.0):
        url = (self.base_url or "") + path
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def _wait_ready(self, deadline: float) -> bool:
        while time.time() < deadline:
            if not self._server or not self._server.alive():
                return False
            try:
                self._rpc("GET", "/session/status", timeout=3)
                return True
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.25)
        return False

    # --- driving one turn --------------------------------------------------
    def _drive(self, task: TaskSpec, journal: JournalWriter, deadline: float) -> SlotResult:
        extra = task.extra or {}

        session_id = extra.get("resume_session_id")
        if session_id and extra.get("fork"):
            # Branch before prompting so the parent stays untouched. messageID is
            # optional: absent means "fork at the tip".
            forked = self._rpc(
                "POST", "/session/%s/fork" % session_id,
                {"messageID": extra["fork_message_id"]} if extra.get("fork_message_id") else {},
            )
            session_id = (forked or {}).get("id") or session_id
            journal.emit(
                "session.forked",
                parent_session=extra["resume_session_id"],
                session_id=session_id,
                at_message=extra.get("fork_message_id"),
            )
        elif not session_id:
            created = self._rpc("POST", "/session", {"title": extra.get("title") or "ash run"})
            session_id = (created or {}).get("id")

        if not session_id:
            return self._fail(journal, "could not create or resume an opencode session")
        self.session_id = session_id
        journal.emit(E.SESSION_REF, native_session_id=session_id)

        # Answer permission requests while the turn runs; the prompt call blocks.
        stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch_permissions, args=(session_id, journal, stop), daemon=True
        )
        watcher.start()

        body: Dict[str, object] = {"parts": [{"type": "text", "text": task.prompt}]}
        if task.model and "/" in task.model:
            provider, model_id = task.model.split("/", 1)
            body["model"] = {"providerID": provider, "modelID": model_id}
        if extra.get("agent"):
            body["agent"] = str(extra["agent"])
        if extra.get("variant"):
            body["variant"] = str(extra["variant"])

        try:
            reply = self._rpc(
                "POST", "/session/%s/message" % session_id, body,
                timeout=max(5.0, deadline - time.time()),
            )
        except urllib.error.URLError as exc:
            stop.set()
            return self._fail(journal, "prompt failed: %s" % exc)
        finally:
            stop.set()

        # The reply carries the assistant message; the durable history is the
        # authoritative record, so read it back rather than trusting one payload.
        messages = self._rpc("GET", "/session/%s/message" % session_id) or []
        usage = normalize.emit_history(messages, journal)
        final_text = normalize.final_text(messages) or normalize.reply_text(reply)

        journal.emit(E.RUN_FINISHED, status="completed", usage=usage)
        return SlotResult(
            status="completed",
            final_text=final_text,
            usage=usage,
            native_session_id=session_id,
        )

    # --- policy ------------------------------------------------------------
    def _watch_permissions(self, session_id: str, journal: JournalWriter,
                           stop: threading.Event) -> None:
        """Poll for permission requests and answer them via the policy callback.

        Polling rather than SSE on purpose: the reply endpoint is the contract we
        need and it is request/response, so a 250 ms poll is simpler than holding
        a second stream open for the same information. A request left unanswered
        blocks the agent, so the loop must keep running for the whole turn.
        """
        seen: set = set()
        while not stop.is_set():
            try:
                pending = self._rpc(
                    "GET", "/session/%s/permission" % session_id, timeout=5
                ) or []
            except Exception:  # noqa: BLE001 - server may be mid-turn
                time.sleep(0.25)
                continue
            for request in pending if isinstance(pending, list) else []:
                request_id = request.get("id") or request.get("requestID")
                if not request_id or request_id in seen:
                    continue
                seen.add(request_id)
                verdict, reason = self.decide("permission", request, journal)
                payload = {"reply": "reject" if verdict == DENY else "once"}
                if reason:
                    payload["message"] = reason
                try:
                    self._rpc(
                        "POST",
                        "/session/%s/permissions/%s" % (session_id, request_id),
                        payload, timeout=15,
                    )
                except Exception:  # noqa: BLE001 - fall back to the global route
                    try:
                        self._rpc("POST", "/permission/%s/reply" % request_id,
                                  payload, timeout=15)
                    except Exception:
                        pass
            stop.wait(0.25)

    # --- config ------------------------------------------------------------
    def _render_config(self, mcp: Optional[McpWiring], task: TaskSpec) -> Optional[str]:
        if mcp is None:
            return None
        if mcp.command:
            entry: Dict[str, object] = {
                "type": "local", "command": list(mcp.command), "enabled": True,
            }
            if mcp.env:
                entry["environment"] = dict(mcp.env)
        elif mcp.url:
            entry = {"type": "remote", "url": mcp.url, "enabled": True}
            if mcp.headers:
                entry["headers"] = dict(mcp.headers)
        else:
            return None

        payload: Dict[str, object] = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {mcp.name: entry},
        }
        base = (task.extra or {}).get("config_base")
        if isinstance(base, dict):
            merged = dict(base)
            merged.setdefault("mcp", {})
            merged["mcp"].update(payload["mcp"])  # type: ignore[index]
            merged.setdefault("$schema", payload["$schema"])
            payload = merged

        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", prefix="opencode-slot-", delete=False, encoding="utf-8"
        )
        with handle as fh:
            json.dump(payload, fh)
        self._config_path = handle.name
        return handle.name

    def _cleanup_config(self) -> None:
        if self._config_path and os.path.exists(self._config_path):
            try:
                os.unlink(self._config_path)
            except OSError:  # pragma: no cover
                pass
        self._config_path = None

    # --- extra capabilities the protocol exposes ---------------------------
    def interrupt(self) -> None:
        """Abort the running turn (the protocol's own interrupt)."""
        if self.base_url and self.session_id:
            try:
                self._rpc("POST", "/session/%s/abort" % self.session_id, {}, timeout=15)
            except Exception:  # noqa: BLE001
                pass

    def fork_at(self, session_id: str, message_id: Optional[str] = None) -> Optional[str]:
        """Branch ``session_id`` at ``message_id`` (tip when omitted)."""
        body = {"messageID": message_id} if message_id else {}
        forked = self._rpc("POST", "/session/%s/fork" % session_id, body)
        return (forked or {}).get("id")

    def revert_to(self, session_id: str, message_id: str,
                  part_id: Optional[str] = None) -> None:
        body: Dict[str, object] = {"messageID": message_id}
        if part_id:
            body["partID"] = part_id
        self._rpc("POST", "/session/%s/revert" % session_id, body)
