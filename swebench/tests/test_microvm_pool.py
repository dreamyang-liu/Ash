"""Tests for MicroVMPool and the optional-capability protocol on Pool.

Run by pytest with the other swebench tests. A stub HTTP server stands in for
the AgentENV API, since Firecracker needs /dev/kvm and a bare-metal host: what
is verified here is the client contract (endpoints, routing headers, capability
declarations), not the hypervisor itself.
User instruction: "pool 能加一个firecracker的backend吗" + "B吧".
"""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ash_sandbox import DockerPool, MicroVMPool, Pool  # noqa: E402


class FakeAenv(BaseHTTPRequestHandler):
    """Minimal stand-in for the AgentENV HTTP API."""

    requests: list = []  # (method, path, body, headers)
    next_id = 0

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def _record(self, method: str) -> dict:
        body = self._read_body()
        FakeAenv.requests.append((method, self.path, body, dict(self.headers)))
        return body

    def _reply(self, payload: dict):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        self._record("POST")
        if self.path == "/sandboxes":
            FakeAenv.next_id += 1
            return self._reply({"sandboxID": f"vm-{FakeAenv.next_id}"})
        if self.path.endswith("/fork"):
            FakeAenv.next_id += 1
            first = FakeAenv.next_id
            FakeAenv.next_id += 1
            return self._reply({"sandboxes": [
                {"sandboxID": f"vm-{first}"},
                {"sandboxID": f"vm-{FakeAenv.next_id}"},
            ]})
        if self.path.rstrip("/") == "":
            # A tool call proxied into a sandbox.
            return self._reply({"result": {"content": [{"type": "text", "text": "ok"}],
                                           "isError": False, "notifications": []}})
        return self._reply({})  # pause / resume

    def do_DELETE(self):
        self._record("DELETE")
        self._reply({})

    def log_message(self, *args):
        pass


@pytest.fixture
def aenv():
    FakeAenv.requests = []
    FakeAenv.next_id = 0
    server = HTTPServer(("127.0.0.1", 0), FakeAenv)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def paths_of(method: str) -> list:
    return [p for m, p, _, _ in FakeAenv.requests if m == method]


def test_capabilities_are_declared_not_assumed():
    # A harness can ask instead of checking types, and containers answer no.
    vm = MicroVMPool("http://aenv:8000")
    assert vm.supports_pause() and vm.supports_fork()

    docker = DockerPool.__new__(DockerPool)
    assert not docker.supports_pause()
    assert not docker.supports_fork()


def test_unsupported_capability_refuses_clearly():
    docker = DockerPool.__new__(DockerPool)
    with pytest.raises(NotImplementedError, match="supports_fork"):
        asyncio.run(docker.fork(None))
    with pytest.raises(NotImplementedError, match="supports_pause"):
        asyncio.run(docker.pause(None))


def test_spawn_requests_a_template_and_tracks_the_sandbox(aenv):
    pool = MicroVMPool(aenv, default_template="swe-base")

    async def scenario():
        sb = await pool.spawn()
        assert sb.sandbox_id == "vm-1"
        assert len(pool.list()) == 1
        await pool.close()

    asyncio.run(scenario())
    assert paths_of("POST")[0] == "/sandboxes"
    _, _, body, _ = FakeAenv.requests[0]
    assert body["templateID"] == "swe-base"


def test_calls_route_through_the_proxy_headers(aenv):
    pool = MicroVMPool(aenv, runtime_port=3000)

    async def scenario():
        sb = await pool.spawn()
        await sb.call("shell", command="true")
        await pool.close()

    asyncio.run(scenario())
    # The tool call must carry AgentENV's routing headers: which sandbox, and
    # which port the runtime listens on inside it.
    tool_calls = [h for m, p, _, h in FakeAenv.requests
                  if m == "POST" and p.rstrip("/") == ""]
    assert tool_calls, "no proxied tool call was made"
    headers = tool_calls[-1]
    assert headers["x-agentenv-sandbox-id"] == "vm-1"
    assert headers["x-agentenv-target-port"] == "3000"


def test_fork_returns_independent_children(aenv):
    pool = MicroVMPool(aenv)

    async def scenario():
        parent = await pool.spawn()
        children = await pool.fork(parent, count=2)
        return parent, children

    parent, children = asyncio.run(scenario())
    assert len(children) == 2
    ids = {c.sandbox_id for c in children}
    assert parent.sandbox_id not in ids, "a fork must not alias its source"
    assert len(ids) == 2
    assert paths_of("POST")[-1] == f"/sandboxes/{parent.sandbox_id}/fork"
    # Children are tracked, so destroy_all reaches them too.
    assert len(pool.list()) == 3


def test_pause_and_resume_hit_the_state_endpoints(aenv):
    pool = MicroVMPool(aenv)

    async def scenario():
        sb = await pool.spawn()
        await pool.pause(sb)
        await pool.resume(sb)
        return sb

    sb = asyncio.run(scenario())
    assert paths_of("POST")[-2:] == [
        f"/sandboxes/{sb.sandbox_id}/pause",
        f"/sandboxes/{sb.sandbox_id}/resume",
    ]


def test_destroy_removes_only_the_named_sandbox(aenv):
    pool = MicroVMPool(aenv)

    async def scenario():
        a = await pool.spawn()
        b = await pool.spawn()
        await pool.destroy(a)
        return a, b

    a, b = asyncio.run(scenario())
    assert paths_of("DELETE") == ["/sandboxes/vm-1"]
    assert [sb.sandbox_id for sb in pool.list()] == [b.sandbox_id]


def test_missing_sandbox_id_in_response_is_an_error():
    from ash_sandbox.pool import _sandbox_id

    with pytest.raises(RuntimeError, match="no sandbox id"):
        _sandbox_id({"unexpected": "shape"})


def test_microvm_pool_satisfies_the_pool_interface():
    pool = MicroVMPool("http://aenv:8000")
    assert isinstance(pool, Pool)
    for name in ("spawn", "destroy", "destroy_all", "list", "close",
                 "pause", "resume", "fork"):
        assert hasattr(pool, name)
