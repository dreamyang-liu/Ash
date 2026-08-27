"""Integration: HTTP session identity and orchestrator-provisioned binding.

Runs a real ``swebench.mcp_server --http`` against real Docker sandboxes,
because the failure this pins down was invisible to every unit test: the stdio
path (which the demo and slot tests use) binds its sandbox at startup and hides
the ``sandbox_id`` parameter, so nothing exercised the header contract.

Skipped unless Docker and the ash-runtime binary are both present.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNTIME_BIN = REPO / "runtime" / "ash-runtime"
IMAGE = "python:3.11-slim"


def _docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(
        ["docker", "info"], capture_output=True, timeout=60
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not (_docker_ready() and RUNTIME_BIN.exists()),
    reason="needs Docker and runtime/ash-runtime (cd runtime && go build -o ash-runtime .)",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """A real MCP server subprocess."""

    def __init__(self, port: int):
        self.port = port
        self.url = "http://127.0.0.1:%d/mcp" % port
        env = dict(os.environ, PYTHONPATH=str(REPO))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "swebench.mcp_server", "--http",
             "--host", "127.0.0.1", "--port", str(port),
             "--runtime-bin", str(RUNTIME_BIN)],
            cwd=str(REPO), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def wait_ready(self, timeout: float = 45.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                stderr = (self.proc.stderr.read() or b"").decode()[-800:]
                raise RuntimeError("server exited: %s" % stderr)
            try:
                self.rpc("ping", timeout=3)
                return
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.4)
        raise RuntimeError("server did not become ready")

    def rpc(self, method: str, params=None, headers=None, timeout: float = 300.0):
        body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        ).encode()
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}"), dict(response.headers)

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def server():
    srv = Server(_free_port())
    try:
        srv.wait_ready()
        yield srv
    finally:
        srv.stop()


def _text(result: dict) -> str:
    for block in result.get("content") or []:
        if block.get("type") == "text":
            return block.get("text") or ""
    return ""


def _create(server, headers):
    result, _ = server.rpc(
        "tools/call",
        {"name": "sandbox_create", "arguments": {"image": IMAGE}},
        headers,
    )
    text = _text(result["result"])
    assert text.strip().startswith("{"), "sandbox_create failed: %s" % text
    return json.loads(text)["id"]


# --- session identity ------------------------------------------------------
def test_initialize_sends_mcp_session_id_as_a_response_header(server):
    """The body field alone is invisible to a conforming client."""
    result, headers = server.rpc("initialize")
    assert headers.get("Mcp-Session-Id")
    assert headers["Mcp-Session-Id"] == result["result"]["sessionId"]


def test_echoing_mcp_session_id_preserves_visibility(server):
    """What a conforming client does automatically, once the header is sent."""
    result, headers = server.rpc("initialize")
    session = {"mcp-session-id": headers["Mcp-Session-Id"]}
    sandbox_id = _create(server, session)
    result, _ = server.rpc(
        "tools/call",
        {"name": "shell", "arguments": {"command": "echo ok", "sandbox_id": sandbox_id}},
        session,
    )
    assert "ok" in _text(result["result"])


def test_anonymous_requests_cannot_see_their_own_sandbox(server):
    """Documents why identity is required: each request is a new owner."""
    sandbox_id = _create(server, None)
    result, _ = server.rpc(
        "tools/call",
        {"name": "shell", "arguments": {"command": "echo ok", "sandbox_id": sandbox_id}},
    )
    assert "visible to you" in _text(result["result"])


def test_owner_isolation(server):
    """A second agent cannot reach the first one's sandbox."""
    sandbox_id = _create(server, {"X-Session-Owner": "agent-a"})
    result, _ = server.rpc(
        "tools/call",
        {"name": "shell", "arguments": {"command": "echo ok", "sandbox_id": sandbox_id}},
        {"X-Session-Owner": "agent-b"},
    )
    assert "visible to you" in _text(result["result"])


# --- orchestrator-provisioned binding --------------------------------------
def test_bound_session_hides_the_sandbox_id_parameter(server):
    """The model must not be able to omit it or point it elsewhere."""
    sandbox_id = _create(server, {"X-Session-Owner": "agent-bound"})
    bound = {"X-Session-Owner": "agent-bound", "X-Session-Sandbox": sandbox_id}

    result, _ = server.rpc("tools/list", headers=bound)
    tools = {t["name"]: t for t in result["result"]["tools"]}
    assert "sandbox_create" not in tools          # no lifecycle surface
    for name in ("shell", "text_editor", "grep_files", "process"):
        props = tools[name]["inputSchema"]["properties"]
        assert "sandbox_id" not in props, "%s still exposes sandbox_id" % name

    unbound, _ = server.rpc("tools/list", headers={"X-Session-Owner": "agent-unbound"})
    unbound_tools = {t["name"] for t in unbound["result"]["tools"]}
    assert "sandbox_create" in unbound_tools      # unbound sessions still get it


def test_bound_session_resolves_calls_without_a_sandbox_id(server):
    sandbox_id = _create(server, {"X-Session-Owner": "agent-bound2"})
    bound = {"X-Session-Owner": "agent-bound2", "X-Session-Sandbox": sandbox_id}
    result, _ = server.rpc(
        "tools/call", {"name": "shell", "arguments": {"command": "echo bound"}}, bound
    )
    assert "bound" in _text(result["result"])


def test_binding_is_a_default_not_a_grant(server):
    """Naming someone else's sandbox in the header gains nothing."""
    victim = _create(server, {"X-Session-Owner": "victim"})
    result, _ = server.rpc(
        "tools/call",
        {"name": "shell", "arguments": {"command": "echo pwned"}},
        {"X-Session-Owner": "attacker", "X-Session-Sandbox": victim},
    )
    assert "visible to you" in _text(result["result"])


def test_rebinding_follows_the_header(server):
    """A slot re-boarded onto a restored sandbox just changes the header."""
    owner = {"X-Session-Owner": "agent-rebind"}
    first = _create(server, owner)
    second = _create(server, owner)

    for sandbox_id, marker in ((first, "first"), (second, "second")):
        headers = dict(owner, **{"X-Session-Sandbox": sandbox_id})
        server.rpc(
            "tools/call",
            {"name": "shell", "arguments": {"command": "echo %s > /tmp/who" % marker}},
            headers,
        )
    # Each sandbox kept its own marker: the binding actually switched.
    for sandbox_id, marker in ((first, "first"), (second, "second")):
        headers = dict(owner, **{"X-Session-Sandbox": sandbox_id})
        result, _ = server.rpc(
            "tools/call", {"name": "shell", "arguments": {"command": "cat /tmp/who"}}, headers
        )
        assert marker in _text(result["result"])


# --- the provisioning helper ------------------------------------------------
def test_provision_http_end_to_end(server):
    from harness.execution.provision import provision_http

    provisioned = provision_http(server.url, image=IMAGE, agent_id="slot-provisioned")
    try:
        assert provisioned.sandbox_id
        assert provisioned.mcp.headers["X-Session-Owner"] == "slot-provisioned"
        assert provisioned.mcp.headers["X-Session-Sandbox"] == provisioned.sandbox_id

        # The wiring the slot receives resolves calls with no sandbox_id.
        result, _ = server.rpc(
            "tools/call",
            {"name": "shell", "arguments": {"command": "echo provisioned"}},
            provisioned.mcp.headers,
        )
        assert "provisioned" in _text(result["result"])
    finally:
        provisioned.destroy()


def test_provision_http_reports_an_unreachable_server():
    from harness.execution.provision import ProvisionError, provision_http

    with pytest.raises(ProvisionError):
        provision_http("http://127.0.0.1:1/mcp", image=IMAGE, agent_id="x", timeout_s=5)
