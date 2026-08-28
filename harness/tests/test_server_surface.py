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
