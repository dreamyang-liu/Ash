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

from ash_sandbox import DockerPool, MicroVMPool, Pool

class FakeAenv(BaseHTTPRequestHandler):
    """Minimal stand-in for the AgentENV HTTP API."""

    requests: list = []  # (method, path, body, headers)
    next_id = 0
    next_snapshot_id = 0
    delete_status = 200

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def _record(self, method: str) -> dict:
        body = self._read_body()
        FakeAenv.requests.append((method, self.path, body, dict(self.headers)))
        return body

    def _reply(self, payload: dict | list, *, status: int = 200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    #: When set, the fork endpoint reports this error for every child after
    #: the first, mimicking AgentENV's per-fork failure reporting.
    fork_error: dict | None = None
    tool_error: dict | None = None

    def do_POST(self):
        body = self._record("POST")
        if self.path == "/sandboxes":
            FakeAenv.next_id += 1
            return self._reply({"sandboxID": f"vm-{FakeAenv.next_id}"})
        if self.path.endswith("/snapshots"):
            FakeAenv.next_snapshot_id += 1
            return self._reply(
                {"snapshotID": f"snap-{FakeAenv.next_snapshot_id}"}
            )
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
            if FakeAenv.tool_error is not None:
                return self._reply({"error": FakeAenv.tool_error})
            return self._reply({"result": {"content": [{"type": "text", "text": "ok"}],
                                           "isError": False, "notifications": []}})
        return self._reply({})  # pause / resume

    def do_DELETE(self):
        self._record("DELETE")
        self._reply({}, status=FakeAenv.delete_status)

    def log_message(self, *args):
        pass


@pytest.fixture
def aenv():
    FakeAenv.requests = []
    FakeAenv.next_id = 0
    FakeAenv.next_snapshot_id = 0
    FakeAenv.delete_status = 200
    FakeAenv.fork_error = None
    FakeAenv.tool_error = None
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

    # Docker does not advertise a full-runtime checkpoint: a container image
    # would preserve files but lose memory and background processes.
    assert vm.checkpoint_capabilities().state_scope == "full-runtime"
    assert vm.checkpoint_capabilities().multiple_restore is True
    assert vm.checkpoint_capabilities().explicit_release is True
    assert docker.checkpoint_capabilities() is None

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
    pool = MicroVMPool(aenv, runtime_port=3000, request_timeout=1234)

    async def scenario():
        sb = await pool.spawn()
        await sb.call("shell", command="true")
        backend = sb.backend
        await pool.close()
        return backend

    backend = asyncio.run(scenario())
    # The tool call must carry AgentENV's routing headers: which sandbox, and
    # which port the runtime listens on inside it.
    tool_calls = [h for m, p, _, h in FakeAenv.requests
                  if m == "POST" and p.rstrip("/") == ""]
    assert tool_calls, "no proxied tool call was made"
    headers = tool_calls[-1]
    assert headers["x-agentenv-sandbox-id"] == "vm-1"
    assert headers["x-agentenv-target-port"] == "3000"
    # Lifecycle calls and proxied runtime calls use the same deployment-level
    # timeout instead of the gateway silently reverting to 360 seconds.
    assert backend._client.timeout.read == 1234


def test_gateway_preserves_json_rpc_error_message(aenv):
    pool = MicroVMPool(aenv)
    FakeAenv.tool_error = {"code": -32000, "message": "sandbox auto-paused"}

    async def scenario():
        sb = await pool.spawn()
        result = await sb.call("shell", command="true")
        await pool.close()
        return result

    result = asyncio.run(scenario())
    assert result.is_error is True
    assert result.output == "sandbox auto-paused"

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

def test_agentenv_checkpoint_contract_creates_restores_and_releases(aenv):
    pool = MicroVMPool(aenv, sandbox_ttl=600)

    async def scenario():
        source = await pool.spawn(agent_id="parent")
        checkpoint_id = await pool.create_checkpoint(source, name="job/step 1")
        child = await pool.restore_checkpoint(checkpoint_id, agent_id="child")
        await pool.release_checkpoint(checkpoint_id)
        return source, checkpoint_id, child

    source, checkpoint_id, child = asyncio.run(scenario())
    assert checkpoint_id == "snap-1"
    assert source.sandbox_id != child.sandbox_id
    assert child.agent_id == "child"

    checkpoint_requests = [item for item in FakeAenv.requests if item[1].endswith("/snapshots")]
    assert checkpoint_requests[0][1] == f"/sandboxes/{source.sandbox_id}/snapshots"
    assert checkpoint_requests[0][2] == {"name": "job/step 1"}
    assert paths_of("DELETE")[-1] == "/templates/snap-1"


def test_agentenv_checkpoint_release_is_idempotent_when_snapshot_is_absent(aenv):
    pool = MicroVMPool(aenv)
    FakeAenv.delete_status = 404

    asyncio.run(pool.release_checkpoint("already-deleted"))

    assert paths_of("DELETE") == ["/templates/already-deleted"]


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
