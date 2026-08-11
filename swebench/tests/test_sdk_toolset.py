"""SDK-level tests for ash_sandbox.toolset + Sandbox.call_agent_tool.

Run by pytest with the other swebench tests (same sys.path pattern as
test_tool_contract.py); uses a fake Backend, no network; asyncio.run
wrappers (no pytest-asyncio in env). Part of the refactor per user
instruction: "tool 之类的可以用DSL 或者data 配置项来做" (tools as data).
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ash_sandbox import Sandbox, ToolRegistry, parse_manifest  # noqa: E402
from ash_sandbox.backends import Backend  # noqa: E402
from ash_sandbox.result import ToolResult  # noqa: E402

SHA = "c" * 64


class FakeBackend(Backend):
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    async def call(self, tool_name, args, agent_id=""):
        self.calls.append((tool_name, dict(args)))
        if tool_name in self.responses:
            return self.responses[tool_name]
        return ToolResult(output="ok", is_error=False)

    async def list_tools(self):
        return []

    async def close(self):
        pass


def make_registry(binary=None):
    reg = ToolRegistry()
    reg.register(parse_manifest({
        "name": "analyzer",
        "description": "d",
        "binary": binary or {"url": "https://example.com/a", "sha256": SHA},
        "parameters": {
            "file": {"type": "string", "required": True, "map": {"positional": 0}},
        },
        "timeout": 20,
    }))
    return reg


def test_call_agent_tool_builtin_routes():
    backend = FakeBackend()
    sb = Sandbox(backend=backend)
    r = asyncio.run(sb.call_agent_tool("shell", {"command": "echo hi"}))
    assert not r.is_error
    assert backend.calls == [("shell", {"command": "echo hi"})]
    # bash alias routes to shell via BUILTIN_ROUTES
    asyncio.run(sb.call_agent_tool("bash", {"command": "ls"}))
    assert backend.calls[-1][0] == "shell"


def test_call_agent_tool_custom_url_two_steps():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="/cache/bin\n", is_error=False),
    })
    sb = Sandbox(backend=backend, tools=make_registry())
    r = asyncio.run(sb.call_agent_tool("analyzer", {"file": "m.py"}))
    assert not r.is_error
    assert [c[0] for c in backend.calls] == ["artifact", "shell"]
    assert backend.calls[1][1]["command"] == "/cache/bin m.py"
    assert backend.calls[1][1]["timeout"] == 20


def test_call_agent_tool_custom_path_single_step():
    backend = FakeBackend()
    sb = Sandbox(backend=backend, tools=make_registry(binary={"path": "/opt/a"}))
    asyncio.run(sb.call_agent_tool("analyzer", {"file": "m.py"}))
    assert [c[0] for c in backend.calls] == ["shell"]
    assert backend.calls[0][1]["command"] == "/opt/a m.py"


def test_call_agent_tool_artifact_failure_short_circuits():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="HTTP 404", is_error=True),
    })
    sb = Sandbox(backend=backend, tools=make_registry())
    r = asyncio.run(sb.call_agent_tool("analyzer", {"file": "m.py"}))
    assert r.is_error
    assert [c[0] for c in backend.calls] == ["artifact"]


def test_per_call_registry_override():
    backend = FakeBackend()
    sb = Sandbox(backend=backend)  # default registry: no custom tools
    with pytest.raises(KeyError):
        asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}))
    # Same sandbox, per-call registry carries the tool panel.
    asyncio.run(sb.call_agent_tool(
        "analyzer", {"file": "x"},
        registry=make_registry(binary={"path": "/opt/a"}),
    ))
    assert backend.calls[-1][0] == "shell"


def test_artifact_resolved_once_then_memoised():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="/cache/bin\n", is_error=False),
    })
    sb = Sandbox(backend=backend, tools=make_registry())

    for _ in range(3):
        asyncio.run(sb.call_agent_tool("analyzer", {"file": "m.py"}))

    names = [c[0] for c in backend.calls]
    # One artifact resolution, then execution only: the redundant round-trip
    # to a cache that already holds the binary is skipped.
    assert names == ["artifact", "shell", "shell", "shell"]


def test_memo_is_per_sandbox_not_per_registry():
    # A registry is reusable across sandboxes, so a path learned in one must
    # not be assumed present in another.
    registry = make_registry()
    first = FakeBackend(responses={"artifact": ToolResult(output="/cache/a", is_error=False)})
    second = FakeBackend(responses={"artifact": ToolResult(output="/cache/b", is_error=False)})

    asyncio.run(Sandbox(backend=first, tools=registry).call_agent_tool("analyzer", {"file": "x"}))
    asyncio.run(Sandbox(backend=second, tools=registry).call_agent_tool("analyzer", {"file": "x"}))

    # The second sandbox resolves the artifact itself rather than reusing the
    # first sandbox's path.
    assert [c[0] for c in second.calls] == ["artifact", "shell"]
    assert second.calls[1][1]["command"].startswith("/cache/b")


def test_stale_memo_is_reresolved_once():
    class StalePathBackend(FakeBackend):
        """Fails the first execution after memoisation, as if /tmp was cleaned."""

        def __init__(self):
            super().__init__()
            self.shell_calls = 0

        async def call(self, tool_name, args, agent_id=""):
            self.calls.append((tool_name, dict(args)))
            if tool_name == "artifact":
                return ToolResult(output="/cache/bin", is_error=False)
            self.shell_calls += 1
            if self.shell_calls == 2:
                return ToolResult(output="sh: /cache/bin: No such file or directory",
                                  is_error=True)
            return ToolResult(output="ok", is_error=False)

    backend = StalePathBackend()
    sb = Sandbox(backend=backend, tools=make_registry())

    asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}))  # resolves + runs
    r = asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}))  # memo gone stale

    assert not r.is_error, "a stale path should be re-resolved, not surfaced as an error"
    assert [c[0] for c in backend.calls] == ["artifact", "shell", "shell", "artifact", "shell"]


def test_genuine_failure_is_not_retried():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="/cache/bin", is_error=False),
        "shell": ToolResult(output="analyzer: invalid threshold", is_error=True),
    })
    sb = Sandbox(backend=backend, tools=make_registry())

    asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}))
    backend.calls.clear()
    r = asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}))

    assert r.is_error
    # A real tool error must not be mistaken for a stale path.
    assert [c[0] for c in backend.calls] == ["shell"]


def test_prepare_tools_resolves_up_front():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="/cache/bin", is_error=False),
    })
    sb = Sandbox(backend=backend, tools=make_registry())

    paths = asyncio.run(sb.prepare_tools())
    assert paths == {"analyzer": "/cache/bin"}
    assert [c[0] for c in backend.calls] == ["artifact"]

    # The first agent call then needs no artifact round-trip at all.
    backend.calls.clear()
    asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}))
    assert [c[0] for c in backend.calls] == ["shell"]


def test_prepare_tools_fails_fast_on_bad_binary():
    backend = FakeBackend(responses={
        "artifact": ToolResult(output="download failed: HTTP 404", is_error=True),
    })
    sb = Sandbox(backend=backend, tools=make_registry())

    with pytest.raises(RuntimeError, match="cannot prepare custom tool"):
        asyncio.run(sb.prepare_tools())


def test_prepare_tools_handles_image_local_binaries():
    backend = FakeBackend()
    sb = Sandbox(backend=backend, tools=make_registry(binary={"path": "/opt/a"}))

    paths = asyncio.run(sb.prepare_tools())
    assert paths == {"analyzer": "/opt/a"}
    # Nothing to download for a binary already in the image.
    assert backend.calls == []


def test_both_pools_share_one_interface():
    from ash_sandbox import DockerPool, Pool, SandboxPool

    assert issubclass(DockerPool, Pool)
    assert issubclass(SandboxPool, Pool)
    # The shared surface a harness may rely on regardless of where sandboxes
    # come from.
    for name in ("spawn", "destroy", "destroy_all", "list", "close",
                 "__aenter__", "__aexit__"):
        assert hasattr(DockerPool, name), f"DockerPool missing {name}"
        assert hasattr(SandboxPool, name), f"SandboxPool missing {name}"


def test_a_new_pool_needs_only_the_three_abstract_steps():
    from ash_sandbox import Pool

    class FakePool(Pool):
        """What a future backend (microVM, remote host) has to supply."""

        def __init__(self):
            self._sandboxes = {}

        async def spawn(self, image=None, entrypoint=None, env=None, resources=None):
            sb = Sandbox(backend=FakeBackend())
            sb._container_id = f"sb-{len(self._sandboxes)}"
            self._sandboxes[sb._container_id] = sb
            return sb

        async def destroy(self, *sandboxes):
            for sb in sandboxes:
                self._sandboxes.pop(sb._container_id, None)

        async def destroy_all(self):
            self._sandboxes.clear()

    pool = FakePool()

    async def scenario():
        async with pool:
            a = await pool.spawn()
            await pool.spawn()
            assert len(pool.list()) == 2       # bookkeeping comes from the base
            await pool.destroy(a)
            assert len(pool.list()) == 1
        return pool.list()

    # close() (via the context manager) tears the rest down.
    assert asyncio.run(scenario()) == []


def test_incomplete_pool_cannot_be_instantiated():
    from ash_sandbox import Pool

    class Missing(Pool):
        async def spawn(self, image=None, entrypoint=None, env=None, resources=None):
            ...
        # destroy / destroy_all not implemented

    with pytest.raises(TypeError):
        Missing()


class IdentityBackend(FakeBackend):
    """Records the agent_id each call arrived with."""

    def __init__(self, responses=None):
        super().__init__(responses)
        self.identities = []

    async def call(self, tool_name, args, agent_id=""):
        self.identities.append((tool_name, agent_id))
        return await super().call(tool_name, args, agent_id)


def test_agent_id_reaches_the_backend():
    backend = IdentityBackend()
    sb = Sandbox(backend=backend)

    asyncio.run(sb.call_agent_tool("shell", {"command": "ls"}, agent_id="agent-a"))
    assert backend.identities == [("shell", "agent-a")]

    # Omitting it stays anonymous rather than inventing an identity.
    asyncio.run(sb.call_agent_tool("shell", {"command": "ls"}))
    assert backend.identities[-1] == ("shell", "")


def test_agent_id_forwarded_to_every_step_of_a_custom_tool():
    backend = IdentityBackend(responses={
        "artifact": ToolResult(output="/cache/bin", is_error=False),
    })
    sb = Sandbox(backend=backend, tools=make_registry())

    asyncio.run(sb.call_agent_tool("analyzer", {"file": "x"}, agent_id="agent-b"))
    # Download and execution are attributed to the same caller, so the
    # runtime's event origin is consistent across the pair.
    assert backend.identities == [("artifact", "agent-b"), ("shell", "agent-b")]


def test_agent_id_on_call_and_prepare_tools():
    backend = IdentityBackend(responses={
        "artifact": ToolResult(output="/cache/bin", is_error=False),
    })
    sb = Sandbox(backend=backend, tools=make_registry())

    asyncio.run(sb.prepare_tools(agent_id="preparer"))
    assert backend.identities == [("artifact", "preparer")]

    asyncio.run(sb.call("shell", agent_id="agent-c", command="ls"))
    assert backend.identities[-1] == ("shell", "agent-c")


def test_agent_id_is_not_passed_as_a_tool_argument():
    # It is transport-level addressing, not something the tool should see.
    backend = IdentityBackend()
    sb = Sandbox(backend=backend)
    asyncio.run(sb.call("shell", agent_id="agent-a", command="ls"))
    assert backend.calls[-1][1] == {"command": "ls"}


def test_call_params_places_agent_id_beside_arguments():
    from ash_sandbox.backends import call_params

    assert call_params("shell", {"command": "ls"}, "agent-a") == {
        "name": "shell",
        "arguments": {"command": "ls"},
        "agent_id": "agent-a",
    }
    # Anonymous calls send exactly what they always did.
    assert call_params("shell", {"command": "ls"}) == {
        "name": "shell",
        "arguments": {"command": "ls"},
    }


class SchemaBackend(FakeBackend):
    """Reports one builtin tool the way the runtime's tools/list does."""

    async def list_tools(self):
        return [{
            "name": "shell",
            "description": "Execute a shell command",
            "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}},
                            "required": ["command"]},
        }]


def test_tool_schemas_include_custom_tools():
    # The runtime does not know custom tools exist -- they are manifests on
    # this side -- so assembling the full panel is the SDK's job.
    sb = Sandbox(backend=SchemaBackend(), tools=make_registry(binary={"path": "/opt/a"}))
    panel = asyncio.run(sb.tool_schemas())
    names = [t["function"]["name"] for t in panel]
    assert names == ["shell", "analyzer"]


def test_tool_schemas_render_anthropic_for_both_sources():
    sb = Sandbox(backend=SchemaBackend(), tools=make_registry(binary={"path": "/opt/a"}))
    panel = asyncio.run(sb.tool_schemas(format="anthropic"))
    # Anthropic names the argument schema differently; a custom tool used to
    # be hardcoded to OpenAI's shape and silently came out wrong here.
    assert [t["name"] for t in panel] == ["shell", "analyzer"]
    for tool in panel:
        assert "input_schema" in tool, f"{tool['name']} missing input_schema"
        assert "parameters" not in tool


def test_tool_schemas_raw_keeps_runtime_shape():
    sb = Sandbox(backend=SchemaBackend(), tools=make_registry(binary={"path": "/opt/a"}))
    panel = asyncio.run(sb.tool_schemas(format="raw"))
    assert [t["name"] for t in panel] == ["shell", "analyzer"]
    for tool in panel:
        assert "inputSchema" in tool


def test_tool_schemas_accepts_per_call_registry():
    sb = Sandbox(backend=SchemaBackend())  # default registry: no custom tools
    assert len(asyncio.run(sb.tool_schemas())) == 1

    panel = asyncio.run(sb.tool_schemas(registry=make_registry(binary={"path": "/opt/a"})))
    assert len(panel) == 2


def test_unsupported_schema_format_is_rejected():
    sb = Sandbox(backend=SchemaBackend())
    with pytest.raises(ValueError, match="unsupported schema format"):
        asyncio.run(sb.tool_schemas(format="gemini"))


def test_registries_are_isolated():
    r1, r2 = make_registry(), ToolRegistry()
    assert r1.is_custom_tool("analyzer")
    assert not r2.is_custom_tool("analyzer")  # no cross-instance leakage
