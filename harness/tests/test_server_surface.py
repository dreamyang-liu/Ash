"""What tool surface the MCP proxy serves, and how a call on it routes.

The proxy used to hand-write its four tools, with no check against the runtime at
all -- the same arrangement that let the agent-side panel drift on four of seven
tools before it was compiled. ``--tools`` lets it serve a compiled panel instead;
these are the properties that made that worth doing, plus the ones that must not
change for the callers still on the literals.

No sandbox and no transport here: the surface is a pure function of a manifest.
The live paths are covered by test_mcp_http_binding.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from harness.execution.panel import load_panel
from harness.execution.server import (
    ALL_TOOLS,
    DEFAULT_SURFACE,
    EXEC_TOOLS_MULTI,
    EXEC_TOOLS_SINGLE,
    ExecSurface,
    Session,
    SessionHandler,
)


def _surface(name_or_path: str) -> ExecSurface:
    return ExecSurface(load_panel(name_or_path, format="raw"))


# --- the default: unchanged ------------------------------------------------
def test_the_default_surface_is_the_literals_object_for_object():
    """Every existing caller gets what it got before. The literals are also what
    the in-process harnesses import, so a change here would silently desynchronize
    them from this server."""
    assert DEFAULT_SURFACE.panel is None
    assert DEFAULT_SURFACE.single is EXEC_TOOLS_SINGLE
    assert DEFAULT_SURFACE.multi is EXEC_TOOLS_MULTI
    assert DEFAULT_SURFACE.all_tools() == ALL_TOOLS


def test_the_default_surface_routes_by_identity():
    """No panel means no views, so a name is already the runtime's."""
    assert DEFAULT_SURFACE.route("shell", {"command": "ls"}) == \
        ("shell", {"command": "ls"})


# --- a compiled panel ------------------------------------------------------
def test_a_panel_replaces_the_hand_written_tools():
    surface = _surface("default")
    assert surface.names == {"shell", "text_editor"}
    assert {t["name"] for t in surface.single} == {"shell", "text_editor"}


def test_the_default_panel_withholds_background_and_process():
    """A backgrounded command outlives the call that started it, and a disk-only
    checkpoint captures the filesystem and not processes -- so replaying a step
    taken while one ran diverges. `process` is absent for the same reason: with no
    way to start one there is nothing to read or kill."""
    surface = _surface("default")
    shell = next(t for t in surface.single if t["name"] == "shell")
    assert "background" not in shell["inputSchema"]["properties"]
    assert "process" not in surface.names


def test_compiled_tools_are_already_mcps_shape():
    """`format="raw"` is `{name, description, inputSchema}`; nothing converts, which
    is why the panel could move into the execution plane at all."""
    for tool in _surface("full").single:
        assert set(tool) >= {"name", "description", "inputSchema"}
        assert tool["inputSchema"]["type"] == "object"
        assert "function" not in tool, "that is the function-calling shape"


# --- sandbox_id: the part a panel knows nothing about ----------------------
def test_the_single_sandbox_surface_hides_sandbox_id():
    """A bound session serves one sandbox, so the model must not see -- and cannot
    name -- another. A panel never mentions sandbox_id at all."""
    for tool in _surface("full").single:
        assert "sandbox_id" not in tool["inputSchema"]["properties"]


def test_the_multi_sandbox_surface_requires_sandbox_id():
    """Stateless multi-sandbox mode: every exec call names its target, so there is
    no switchable "active" sandbox for concurrent calls to race on."""
    for tool in _surface("full").multi:
        schema = tool["inputSchema"]
        assert "sandbox_id" in schema["properties"]
        assert "sandbox_id" in schema["required"]


def test_injecting_sandbox_id_does_not_mutate_the_panel():
    """`single` and `multi` come off the same compiled list; sharing the dicts would
    have the multi variant's injection show up in the single one, which is the
    surface that must not have it."""
    surface = _surface("full")
    assert all("sandbox_id" not in t["inputSchema"]["properties"]
               for t in surface.single)


def test_lifecycle_tools_are_served_alongside_a_panel():
    """They are this server's own, not the runtime's, so no manifest declares
    them -- an unbound session still needs to create and list sandboxes."""
    names = [t["name"] for t in _surface("default").all_tools()]
    assert names[:3] == ["sandbox_create", "sandbox_list", "sandbox_destroy"]
    assert "shell" in names


# --- routing ---------------------------------------------------------------
def test_a_renamed_view_reaches_the_runtime_under_its_real_name(tmp_path):
    """The reason routing happens at all. An interceptor keyed on `shell`, the
    mutation notification's `_MUTATING` set and the sandbox itself all name runtime
    tools; a view called `run_tests` that arrived unrouted would be invisible to
    every one of them."""
    manifest = tmp_path / "renamed.yaml"
    manifest.write_text(
        "agent_tools:\n"
        "  - name: run_tests\n"
        "    runtime_tool: shell\n"
        "    description: Run the suite.\n"
        "    arguments:\n"
        "      target: command\n"
        "    required: [target]\n")
    surface = _surface(str(manifest))
    assert surface.names == {"run_tests"}
    assert surface.route("run_tests", {"target": "pytest -q"}) == \
        ("shell", {"command": "pytest -q"})


def test_an_argument_the_view_does_not_offer_is_refused():
    """Not dropped. A silently ignored parameter has the model believe a setting
    took effect -- the `web_fetch(max_length=…)` defect that motivated compiling."""
    with pytest.raises(ValueError) as exc:
        _surface("default").route("shell", {"command": "ls", "background": True})
    assert "background" in str(exc.value)


def test_an_unknown_tool_raises_keyerror():
    with pytest.raises(KeyError):
        _surface("default").route("no_such_tool", {})


# --- through the handler ---------------------------------------------------
@dataclass
class FakeSandbox:
    calls: List[tuple] = field(default_factory=list)

    async def call(self, name: str, **kwargs):
        self.calls.append((name, kwargs))

        @dataclass
        class R:
            output: str = "ok"
            is_error: bool = False
            error: Optional[str] = None
        return R()


@dataclass
class FakeEntry:
    id: str
    sandbox: Any
    groups: List[str] = field(default_factory=lambda: ["owner:s"])

    def visible_to(self, groups):
        return bool(set(self.groups) & set(groups))


@dataclass
class FakePool:
    entry: Any

    def get(self, sandbox_id):
        return self.entry if sandbox_id == self.entry.id else None


def _handler(surface):
    sandbox = FakeSandbox()
    entry = FakeEntry(id="sb-1", sandbox=sandbox)
    session = Session(id="s", groups=["owner:s"], bound_id="sb-1")
    return SessionHandler(session, FakePool(entry), surface=surface), sandbox


def test_the_handler_routes_before_it_calls_the_sandbox(tmp_path):
    manifest = tmp_path / "renamed.yaml"
    manifest.write_text(
        "agent_tools:\n"
        "  - name: run_tests\n"
        "    runtime_tool: shell\n"
        "    description: Run the suite.\n"
        "    arguments:\n"
        "      target: command\n"
        "    required: [target]\n")
    handler, sandbox = _handler(_surface(str(manifest)))
    asyncio.run(handler.call_tool("run_tests", {"target": "pytest -q"}))
    assert sandbox.calls == [("shell", {"command": "pytest -q"})]


def test_a_refused_argument_comes_back_as_a_tool_error_not_an_exception():
    """The caller above is an agent loop; an escaped exception kills a whole run."""
    handler, sandbox = _handler(_surface("default"))
    result = asyncio.run(handler.call_tool("shell", {"command": "ls",
                                                    "background": True}))
    assert result["isError"] is True
    assert "background" in result["text"]
    assert sandbox.calls == [], "the call must not reach the sandbox"


def test_an_unknown_tool_comes_back_as_a_tool_error():
    handler, _ = _handler(_surface("default"))
    result = asyncio.run(handler.call_tool("no_such_tool", {}))
    assert result["isError"] is True
    assert "unknown tool" in result["text"]


def test_the_mutation_hook_fires_in_multi_sandbox_mode():
    """It did not. The hook re-resolved the sandbox from `args["sandbox_id"]`, which
    the handler had already popped, so it resolved to the *bound* sandbox -- and in
    multi-sandbox mode nothing is bound, so a SandboxPool subclass hooking mutations
    was never called. Single-sandbox mode masked it, and that is the only mode that
    turns the hook on today, which is why nothing noticed."""
    sandbox = FakeSandbox()
    entry = FakeEntry(id="sb-1", sandbox=sandbox)
    seen = []

    class HookingPool(FakePool):
        async def after_mutating_call(self, entry, name, args):
            seen.append((entry.id, name))

    # No bound_id: this is the multi-sandbox case, where the call names its target.
    session = Session(id="s", groups=["owner:s"])
    handler = SessionHandler(session, HookingPool(entry), notify_mutations=True)
    asyncio.run(handler.call_tool("shell", {"command": "rm -rf x",
                                            "sandbox_id": "sb-1"}))
    assert seen == [("sb-1", "shell")]


def test_the_handler_defaults_to_the_literals_when_given_no_surface():
    session = Session(id="s", groups=["owner:s"], bound_id="sb-1")
    handler = SessionHandler(session, FakePool(FakeEntry(id="sb-1", sandbox=None)))
    assert handler.surface is DEFAULT_SURFACE


# --- checkpoints: ONE mechanism, mounted in the serving process --------------
#
# The machinery is MutationTracker (an interceptor, on the serving pipeline) plus
# Checkpointer (fired at every tool boundary). Both transports mount exactly this;
# they differ only in which process constructs it and where the records land.
# These tests drive it the way the stdio server does: tracker in a ToolPipeline,
# Checkpointer over a fake session, records appended as JSONL.

def _machinery(tmp_path, always=False):
    import json

    from harness.execution.checkpoints import Checkpointer
    from harness.execution.interceptors import MutationTracker
    from harness.execution.pipeline import CallContext, Continue, ToolPipeline
    from harness.execution.server import ToolBoundary

    class FakeSnapSession:
        def __init__(self):
            self.captures = 0

        def supports_snapshot(self):
            return True

        def snapshot(self, name=None, disk_only=True):
            self.captures += 1
            return type("Snap", (), {"id": "snap-%d" % self.captures,
                                     "rootfs_layers": None,
                                     "memory_layers": None,
                                     "chain_size_mb": None})()

    log = tmp_path / "map.jsonl"

    def append(record):
        with open(log, "a") as fh:
            fh.write(json.dumps({"step": record.turn,
                                 "snapshot_id": record.snapshot_id,
                                 "captured": bool(record.captured),
                                 "reason": record.reason}) + "\n")

    tracker = MutationTracker()
    session = FakeSnapSession()
    checkpointer = Checkpointer(session=session, tracker=tracker,
                                always=always, on_checkpoint=append)
    boundary = ToolBoundary(checkpointer.after_step)
    pipeline = ToolPipeline([tracker])

    def call(tool, args):
        """One exec call as the handler runs it: through the pipeline (feeding
        the tracker), then the boundary."""
        import asyncio

        from harness.core.result import ToolResult

        ctx = CallContext(agent_id="a", sandbox_id="sb", tool_name=tool,
                          args=dict(args))
        pipeline.execute(ctx, lambda t, a: ToolResult(success=True, output="ok"))
        asyncio.run(boundary.after_call())

    def lines():
        import json as _json

        return [_json.loads(l) for l in log.read_text().splitlines()]

    return call, lines, session


def test_a_clean_step_reuses_the_previous_snapshot(tmp_path):
    """THE regression this unification fixes, in both directions. The old stdio
    path snapshotted on `text_editor view` (a read paid for a capture); the old
    http path had an unfed tracker, so every step after the first was recorded as
    "clean" reuse of snapshot 1 -- while the agent was writing files. With the
    tracker on the serving pipeline, a write captures and a view reuses."""
    call, lines, session = _machinery(tmp_path)
    call("text_editor", {"command": "write", "path": "/a", "file_text": "x"})
    call("text_editor", {"command": "view", "path": "/a"})
    call("text_editor", {"command": "write", "path": "/b", "file_text": "y"})

    got = [(l["step"], l["snapshot_id"], l["captured"]) for l in lines()]
    assert got == [(1, "snap-1", True),
                   (2, "snap-1", False),      # view: mapped, not paid for
                   (3, "snap-2", True)]       # write: a NEW snapshot, not reuse
    assert session.captures == 2, "two writes, two captures, no more"


def test_every_step_gets_a_map_entry(tmp_path):
    """A fork at a read-only step must have something to restore from. The old
    _MUTATING pre-filter skipped grep_files entirely, so that step had no entry."""
    call, lines, _ = _machinery(tmp_path)
    call("shell", {"command": "echo hi > f"})
    call("grep_files", {"pattern": "hi"})
    assert [l["step"] for l in lines()] == [1, 2]
    assert lines()[1]["snapshot_id"] == lines()[0]["snapshot_id"]
    assert lines()[1]["captured"] is False and lines()[1]["reason"] == "clean"


def test_checkpoint_always_captures_every_step(tmp_path):
    call, lines, session = _machinery(tmp_path, always=True)
    call("text_editor", {"command": "view", "path": "/a"})
    call("text_editor", {"command": "view", "path": "/a"})
    assert session.captures == 2, "always means a distinct snapshot per step"


def test_a_failing_checkpointer_does_not_fail_the_tool_call():
    """The boundary swallows: a checkpoint is an optimisation for later analysis,
    and failing the agent's call over it is strictly worse than a gap."""
    import asyncio

    from harness.execution.server import ToolBoundary

    def explode(step):
        raise RuntimeError("backend said no")

    boundary = ToolBoundary(explode)
    asyncio.run(boundary.after_call())        # no raise
    assert boundary.step == 1, "the step still counts; the map has a gap, not a shift"
