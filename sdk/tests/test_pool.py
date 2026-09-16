"""Tests for MicroVMPool and the optional-capability protocol on Pool.

Run by pytest from the SDK (`pytest sdk/tests`). A stub HTTP server stands in
for the AgentENV API, since Firecracker needs /dev/kvm and a bare-metal host:
what is verified here is the client contract (endpoints, routing headers,
capability declarations), not the hypervisor itself.
User instruction: "pool 能加一个firecracker的backend吗" + "B吧".
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ash_sandbox import (
    DockerPool,
    GatewayBackend,
    MicroVMPool,
    Pool,
    SandboxRouteUnavailable,
)

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

    def _reply(self, payload: dict | list):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    #: When set, the fork endpoint reports this error for every child after
    #: the first, mimicking AgentENV's per-fork failure reporting.
    fork_error: dict | None = None

    def do_POST(self):
        body = self._record("POST")
        if self.path in ("/sandboxes", "/sandboxes-cold"):
            FakeAenv.next_id += 1
            return self._reply({"sandboxID": f"vm-{FakeAenv.next_id}"})
        if self.path.endswith("/fork"):
            # AgentENV replies with one result per requested fork, each
            # carrying either `sandbox` or `error` (verified against a live
            # server; see /sandboxes/{id}/fork in its OpenAPI spec).
            results = []
            for i in range(body.get("count", 1)):
                if i > 0 and FakeAenv.fork_error:
                    results.append({"error": FakeAenv.fork_error})
                    continue
                FakeAenv.next_id += 1
                results.append({"sandbox": {"sandboxID": f"vm-{FakeAenv.next_id}"}})
            return self._reply(results)
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
    FakeAenv.fork_error = None
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


def test_failed_delete_keeps_the_sandbox_owned():
    import httpx

    pool = MicroVMPool("http://agentenv")

    async def scenario():
        await pool._client.aclose()
        pool._client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "busy"})))
        sandbox = pool.attach("owned-vm")
        with pytest.raises(httpx.HTTPStatusError):
            await pool.destroy(sandbox)
        assert sandbox.sandbox_id == "owned-vm"
        assert pool.list() == [sandbox]
        await pool._client.aclose()

    asyncio.run(scenario())

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


def test_gateway_retries_only_pre_dispatch_route_misses():
    import httpx

    requests = []

    def responder(request):
        requests.append(request)
        if len(requests) < 3:
            return httpx.Response(
                404, text="sandbox assignment not found", request=request)
        return httpx.Response(200, json={
            "result": {"content": [{"type": "text", "text": "ok"}],
                       "isError": False, "notifications": []},
        }, request=request)

    async def scenario():
        backend = GatewayBackend("http://agentenv", "vm-1",
                                 route_retry_attempts=2, route_retry_delay_s=0)
        await backend._client.aclose()
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
        try:
            result = await backend.call("shell", {"command": "true"})
            assert result.output == "ok" and not result.is_error
        finally:
            await backend.close()

    asyncio.run(scenario())
    assert len(requests) == 3


def test_gateway_exhausted_route_miss_proves_request_was_not_executed():
    import httpx

    async def scenario():
        backend = GatewayBackend("http://agentenv", "vm-missing",
                                 route_retry_attempts=1, route_retry_delay_s=0)
        await backend._client.aclose()
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(
                404, text="sandbox not found", request=request)))
        try:
            with pytest.raises(SandboxRouteUnavailable) as raised:
                await backend.call("shell", {"command": "true"})
            assert raised.value.execution_detail == {
                "kind": "sandbox_route_unavailable",
                "request_may_have_executed": False,
                "sandbox_id": "vm-missing",
                "message": "sandbox not found",
            }
        finally:
            await backend.close()

    asyncio.run(scenario())


def test_gateway_does_not_retry_runtime_404():
    import httpx

    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="application route not found", request=request)

    async def scenario():
        backend = GatewayBackend("http://agentenv", "vm-1",
                                 route_retry_attempts=3, route_retry_delay_s=0)
        await backend._client.aclose()
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await backend.call("shell", {"command": "missing"})
        finally:
            await backend.close()

    asyncio.run(scenario())
    assert calls == 1


def test_gateway_does_not_retry_ambiguous_json_404():
    import httpx

    calls = 0

    def responder(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            404, json={"error": "sandbox not found", "source": "runtime"},
            request=request)

    async def scenario():
        backend = GatewayBackend("http://agentenv", "vm-1",
                                 route_retry_attempts=3, route_retry_delay_s=0)
        await backend._client.aclose()
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await backend.call("shell", {"command": "missing"})
        finally:
            await backend.close()

    asyncio.run(scenario())
    assert calls == 1

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
    pool = MicroVMPool(aenv, sandbox_ttl=600)

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
    # Resume must carry a JSON body: AgentENV replies 415 to a bare POST, and
    # the timeout restarts the sandbox's TTL clock.
    _, _, resume_body, _ = FakeAenv.requests[-1]
    assert resume_body == {"timeout": 600}


def test_long_lease_reaches_create_cold_restore_resume_and_fork(aenv):
    pool = MicroVMPool(aenv, sandbox_ttl=11400)

    async def scenario():
        parent = await pool.spawn(image="task-template")
        await pool.spawn_from_image("registry/task:tag", resources={"cpu": 2, "memory_mb": 8192, "disk_size_mb": 65536})
        await pool.spawn(image="saved-snapshot")
        await pool.pause(parent)
        await pool.resume(parent)
        await pool.fork(parent, count=2)
        await pool.close()

    asyncio.run(scenario())
    timed = [(path, body) for method, path, body, _ in FakeAenv.requests
             if method == "POST" and "timeout" in body]
    assert len(timed) == 5
    assert all(body["timeout"] == 11400 for _, body in timed)
    assert timed[0][1]["templateID"] == "task-template"
    assert timed[1][0] == "/sandboxes-cold"
    assert timed[1][1]["diskSizeMB"] == 65536
    assert timed[2][1]["templateID"] == "saved-snapshot"
    assert timed[3][0].endswith("/resume")
    assert timed[4][0].endswith("/fork")

def test_requests_authenticate_with_the_x_api_key_header(aenv):
    # AgentENV validates X-API-KEY; an Authorization: Bearer header is not
    # checked (observed against a live server: garbage bearer tokens pass).
    pool = MicroVMPool(aenv, api_key="secret-key")

    async def scenario():
        await pool.spawn()
        await pool.close()

    asyncio.run(scenario())
    _, _, _, headers = FakeAenv.requests[0]
    assert headers.get("X-API-KEY") == "secret-key"
    assert "Authorization" not in headers

def test_spawn_sends_ttl_and_pause_policy(aenv):
    # AgentENV's default TTL is 15s, after which the VM pauses — far too short
    # for an agent turn, so spawn must always pick the TTL explicitly.
    pool = MicroVMPool(aenv, sandbox_ttl=900)

    async def scenario():
        await pool.spawn()
        await pool.close()

    asyncio.run(scenario())
    _, _, body, _ = FakeAenv.requests[0]
    assert body["timeout"] == 900
    assert body["autoPause"] is True
    assert body["autoResume"] == {"enabled": True}

def test_allow_internet_reaches_every_create_payload(aenv):
    # A no-network benchmark is only no-network if the flag is on the request
    # AgentENV sees, on both create paths; and the default must stay absent so
    # the server's own policy applies when nobody asked.
    pool = MicroVMPool(aenv, allow_internet=False)

    async def scenario():
        await pool.spawn()
        await pool.close()

    asyncio.run(scenario())
    _, _, body, _ = FakeAenv.requests[0]
    assert body["allowInternetAccess"] is False
    # POST /sandboxes on the deployed server reads the snake_case key only.
    assert body["allow_internet_access"] is False

    FakeAenv.requests.clear()
    default_pool = MicroVMPool(aenv)

    async def default_scenario():
        await default_pool.spawn()
        await default_pool.close()

    asyncio.run(default_scenario())
    _, _, body, _ = FakeAenv.requests[0]
    assert "allowInternetAccess" not in body


@pytest.mark.parametrize("allow", [True, False])
@pytest.mark.parametrize("cold_start", [True, False])
def test_explicit_network_policy_reaches_cold_and_restored_sandboxes(aenv, allow, cold_start):
    pool = MicroVMPool(aenv, allow_internet=allow)

    async def scenario():
        if cold_start:
            await pool.spawn_from_image("fixture/image")
        else:
            await pool.spawn(image="fixture-snapshot")
        await pool.close()

    asyncio.run(scenario())
    _, path, body, _ = FakeAenv.requests[0]
    assert path == ("/sandboxes-cold" if cold_start else "/sandboxes")
    assert body["allowInternetAccess"] is allow
    if not cold_start:
        assert body["allow_internet_access"] is allow

def test_spawn_refuses_what_a_template_cannot_express(aenv):
    # entrypoint and resources are container-pool concepts; a microVM template
    # bakes both in. Silently dropping them would strand a caller whose setup
    # command never ran.
    pool = MicroVMPool(aenv)

    async def spawn_with_entrypoint():
        await pool.spawn(entrypoint="pip install -e .")

    async def spawn_with_resources():
        await pool.spawn(resources={"cpu": "2"})

    with pytest.raises(ValueError, match="entrypoint"):
        asyncio.run(spawn_with_entrypoint())
    with pytest.raises(ValueError, match="resources"):
        asyncio.run(spawn_with_resources())

def test_partial_fork_failure_is_an_error_not_a_short_list(aenv):
    # A 201 from AgentENV only means the snapshot succeeded; each fork then
    # reports its own outcome. Returning 1 child when 2 were asked for would
    # let a rollout silently lose branches.
    FakeAenv.fork_error = {"code": 507, "message": "insufficient memory"}
    pool = MicroVMPool(aenv)

    async def scenario():
        parent = await pool.spawn()
        await pool.fork(parent, count=2)

    with pytest.raises(RuntimeError, match="1/2 forks failed"):
        asyncio.run(scenario())

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

def test_spawn_binds_an_identity_to_the_handle(aenv):
    # A pool-provisioned sandbox must be able to come out named. Without this
    # every agent reaches the runtime anonymously and they share one cursor
    # over the event log, silently splitting events between them.
    pool = MicroVMPool(aenv)

    async def scenario():
        sb = await pool.spawn(agent_id="reviewer")
        await sb.call("shell", command="true")
        await pool.close()

    asyncio.run(scenario())
    tool_calls = [body for m, p, body, _ in FakeAenv.requests
                  if m == "POST" and p.rstrip("/") == ""]
    assert tool_calls, "no proxied tool call was made"
    params = tool_calls[0]["params"]
    assert params["agent_id"] == "reviewer"
    # Identity is transport plumbing, not something the tool (or a model
    # reading the call) should see among the arguments.
    assert "agent_id" not in params.get("arguments", {})

def test_spawn_without_an_identity_stays_anonymous(aenv):
    # The parameter is optional: a single-agent sandbox need not name anyone.
    pool = MicroVMPool(aenv)

    async def scenario():
        sb = await pool.spawn()
        assert sb.agent_id == ""
        await pool.close()

    asyncio.run(scenario())

def test_forked_children_inherit_the_source_identity(aenv):
    # A fork is a separate VM with its own event log, so children may reuse
    # the name without competing for a cursor.
    pool = MicroVMPool(aenv)

    async def scenario():
        parent = await pool.spawn(agent_id="explorer")
        children = await pool.fork(parent, count=2)
        assert [c.agent_id for c in children] == ["explorer", "explorer"]

        # ...unless the branches should be distinguishable in traces.
        named = await pool.fork(parent, count=2, agent_ids=["branch-a", "branch-b"])
        assert [c.agent_id for c in named] == ["branch-a", "branch-b"]
        await pool.close()

    asyncio.run(scenario())

def test_pool_contract_carries_identity():
    # Python's ABC checks method names, not signatures, so an implementation
    # can silently drop a parameter. Assert the contract explicitly.
    import inspect

    from ash_sandbox import SandboxPool

    for cls in (Pool, DockerPool, MicroVMPool, SandboxPool):
        params = inspect.signature(cls.spawn).parameters
        assert "agent_id" in params, f"{cls.__name__}.spawn lost agent_id"
        assert params["agent_id"].default == "", \
            f"{cls.__name__}.spawn should make agent_id optional"
