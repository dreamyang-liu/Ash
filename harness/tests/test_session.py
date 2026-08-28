"""The session split: what descended into the execution plane, and what did not.

``AshSession`` used to be one class doing three jobs -- sandbox lifecycle,
snapshots, and patch extraction -- in ``swebench/sandbox.py``. The orchestrator
could not reach it from there, so it required its caller to build a session and
hand it over, which is the opposite of owning a run. The sandbox half is now
:class:`harness.execution.session.SandboxSession` and ``AshSession`` subclasses it.

These pin the properties that make the split correct rather than merely tidy: the
execution plane knows nothing about patches, the benchmark layer still gets its
baseline, and the hook that separates them runs exactly when it should.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

import pytest

from harness.execution.session import OWNER_AGENT_ID, SandboxSession

REPO = Path(__file__).resolve().parents[2]


# --- fakes -----------------------------------------------------------------
class FakeResult:
    def __init__(self, output: str = "", is_error: bool = False):
        self.output = output
        self.is_error = is_error
        self.error = None


class FakeSandbox:
    """A sandbox-shaped object. Deliberately minimal -- see the id test."""

    def __init__(self, sandbox_id: str = "sb-1", reachable: bool = True):
        self.sandbox_id = sandbox_id
        self.agent_id = "agent"
        self.base_ref = "img@sha256:abc"
        self._container_id = "c1"
        self.reachable = reachable
        self.calls: List[tuple] = []

    async def call(self, tool: str, **kwargs):
        self.calls.append((tool, kwargs))
        if not self.reachable:
            raise RuntimeError("502 Bad Gateway")
        return FakeResult(output="ok")

    async def call_agent_tool(self, tool: str, args: dict, registry=None,
                              agent_id: str = ""):
        self.calls.append((tool, args, agent_id))
        return FakeResult(output="ran %s as %s" % (tool, agent_id))


class FakePool:
    def __init__(self, sandbox=None, replacement=None):
        self.sandbox = sandbox or FakeSandbox()
        self.replacement = replacement
        self.destroyed: List[Any] = []
        self.snapshots: List[dict] = []

    def supports_snapshot(self):
        return True

    def supports_upload(self):
        return True

    def supports_cold_start(self):
        return True

    async def spawn(self, image=None, agent_id=""):
        if self.replacement is not None:
            return self.replacement
        return self.sandbox

    async def spawn_from_image(self, image, resources=None):
        return self.sandbox

    async def destroy(self, sandbox):
        self.destroyed.append(sandbox)

    async def snapshot(self, sandbox, name=None, disk_only=True):
        self.snapshots.append({"name": name, "disk_only": disk_only})
        return type("Snap", (), {"id": "snap-1"})()


def _session(pool: FakePool, cls=SandboxSession, **kwargs) -> SandboxSession:
    """A session with its pool already in place, skipping create()."""
    session = cls(quiet=True, **kwargs)
    session._pool = pool
    session._sandbox = pool.sandbox
    return session


# --- what the execution plane must NOT know --------------------------------
def _imported_modules(path: Path) -> set:
    """Every module name this file imports, including deferred ones.

    AST rather than a substring search: the first version of this test grepped the
    file for "swebench" and tripped over the module's own docstring explaining
    where the code moved from. Prose about a layer is not a dependency on it --
    and an import inside a function is, which is what a naive `^import` regex
    would have missed.
    """
    import ast

    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _code_identifiers(path: Path) -> set:
    """Names and attributes appearing in code, ignoring comments and strings."""
    import ast

    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
    return found


def test_the_session_module_knows_nothing_about_patches_or_benchmarks():
    """"What counts as the answer" is the eval layer's question. A session runs
    tools and takes snapshots; the moment it also decides what to extract, every
    benchmark's notion of an answer starts leaking downward."""
    path = REPO / "harness" / "execution" / "session.py"
    identifiers = _code_identifiers(path)
    for forbidden in ("get_patch", "extract_patch", "UNTRACKED_LIST",
                      "_base_commit", "WORKDIR"):
        assert forbidden not in identifiers, \
            "%s belongs to the layer that knows what the answer is" % forbidden


def test_the_harness_package_does_not_import_the_benchmark_layer():
    """The direction of the dependency is the whole point of the layering. Swept
    over the WHOLE package (demo_fork.py used to slip through because only
    execution/ was checked), with one documented exception: extract.py lazily
    imports swebench's patch extractor inside the convenience helper that exists
    precisely to hand a benchmark's own extractor back to it."""
    allowed = {REPO / "harness" / "extract.py"}
    for path in sorted((REPO / "harness").rglob("*.py")):
        if "tests" in path.parts or path in allowed:
            continue
        offenders = {m for m in _imported_modules(path)
                     if m == "swebench" or m.startswith("swebench.")}
        assert not offenders, "%s imports %s" % (path, ", ".join(offenders))


# --- the subclass relationship ---------------------------------------------
def test_ash_session_is_a_sandbox_session():
    from swebench.sandbox import AshSession

    assert issubclass(AshSession, SandboxSession)


def test_the_harness_agent_id_is_one_constant():
    """Two spellings of the owner's identity would put bookkeeping into an
    agent's event stream, and consume its cursor over the log."""
    from swebench.sandbox import HARNESS_AGENT_ID

    assert HARNESS_AGENT_ID is OWNER_AGENT_ID


def test_the_subclass_adds_base_commit_to_the_environment():
    """The image identifies the bits; the commit identifies the code inside them,
    and only the benchmark layer cares."""
    from swebench.sandbox import AshSession

    session = _session(FakePool(), cls=AshSession)
    session._base_commit = "abc123"
    env = session.environment()
    assert env["base_commit"] == "abc123"
    # ...and the generic half is still there.
    assert env["sandbox_id"] == "sb-1"
    assert env["base_ref"] == "img@sha256:abc"
    assert "base_commit" not in SandboxSession.environment(session)


# --- the hook that separates them ------------------------------------------
def test_after_create_runs_on_create(pooled):
    seen = []

    class Probing(SandboxSession):
        async def _after_create(self, sandbox):
            seen.append(sandbox)

    pool = pooled(FakePool())
    session = Probing(quiet=True)
    assert session.create("img") is True
    assert seen == [pool.sandbox], "the baseline probe must run once, on create"


def test_after_create_does_not_run_on_swap_sandbox():
    """The asymmetry is load-bearing, not an oversight. A subclass records a
    *baseline* in this hook -- which files the image itself left untracked -- and
    re-probing after a re-board would file the agent's own new files under that
    baseline and silently drop them from the answer."""
    seen = []

    class Probing(SandboxSession):
        async def _after_create(self, sandbox):
            seen.append(sandbox)

    replacement = FakeSandbox(sandbox_id="sb-2")
    pool = FakePool(replacement=replacement)
    session = _session(pool, cls=Probing)
    assert session.swap_sandbox("snap-x") is True
    assert session._sandbox is replacement
    assert seen == [], "a re-board must not re-probe the baseline"


def test_a_failing_create_reports_false_rather_than_raising(pooled):
    """A run reports; it does not explode in its caller's face."""
    class Broken(SandboxSession):
        async def _after_create(self, sandbox):
            raise RuntimeError("probe blew up")

    pooled(FakePool())
    assert Broken(quiet=True).create("img") is False


# --- the bug this refactor introduced, and its fix -------------------------
def test_a_progress_line_cannot_break_the_operation_it_reports():
    """Reporting used to sit behind `if not self.quiet`, so a quiet caller never
    computed the sandbox id. Moving reporting into a helper made the argument eager
    -- and a successful re-board started raising AttributeError from a *logging*
    line on any sandbox object that could not name itself."""
    class Nameless:
        agent_id = "agent"
        _container_id = "c"

        async def call(self, tool, **kwargs):
            return FakeResult(output="ok")

    replacement = Nameless()
    pool = FakePool(replacement=replacement)
    for quiet in (True, False):
        session = SandboxSession(quiet=quiet)
        session._pool = pool
        session._sandbox = FakeSandbox()
        pool.destroyed.clear()
        assert session.swap_sandbox("snap-x") is True, "quiet=%s" % quiet
        assert session.sandbox_id == "unknown"


# --- the checkpoint contract -----------------------------------------------
def test_the_three_methods_checkpointing_calls_are_present():
    """harness.checkpointing duck-types on exactly these. A session missing one
    cannot record the environment half of a rollback pair."""
    for name in ("snapshot", "squash_snapshot", "swap_sandbox"):
        assert callable(getattr(SandboxSession, name, None)), name


def test_snapshot_defaults_to_disk_only():
    """A rollout replays by restoring the disk and re-feeding the transcript, so
    paying for a memory image every step buys nothing."""
    pool = FakePool()
    session = _session(pool)
    assert session.snapshot().id == "snap-1"
    assert pool.snapshots == [{"name": None, "disk_only": True}]


def test_snapshot_returns_none_rather_than_failing_the_episode():
    """A checkpoint is an optimisation for later analysis. An episode that dies
    because a snapshot did is strictly worse than one with a gap in its history."""
    class Failing(FakePool):
        async def snapshot(self, sandbox, name=None, disk_only=True):
            raise RuntimeError("backend said no")

    assert _session(Failing()).snapshot() is None


def test_squash_returns_the_input_when_the_backend_cannot():
    """A deep chain still works; it just makes its children's checkpoints more
    expensive. Returning None here would look like a lost checkpoint."""
    class NoSquash(FakePool):
        def supports_snapshot(self):
            return False

    snap = object()
    assert _session(NoSquash()).squash_snapshot(snap) is snap


def test_an_unreachable_reboard_target_is_not_adopted():
    """A disk-only snapshot cold-boots, so its runtime is only back if the
    template declares a startup command. Adopting an unreachable replacement turns
    every later tool call into a transport error; keeping the old sandbox costs
    only a deeper layer chain."""
    original = FakeSandbox(sandbox_id="sb-1")
    replacement = FakeSandbox(sandbox_id="sb-2", reachable=False)
    pool = FakePool(sandbox=original, replacement=replacement)
    session = _session(pool)
    assert session.swap_sandbox("snap-x") is False
    assert session._sandbox is original
    assert pool.destroyed == [replacement], "the useless replacement is cleaned up"


# --- the tool seam ---------------------------------------------------------
def test_the_executor_binds_one_agents_identity():
    """The identity belongs to the channel, not to each call, so a call site
    cannot forget it and an agent cannot act as another."""
    pool = FakePool()
    session = _session(pool)
    result = session.executor_for("agent-7")("shell", {"command": "ls"})
    assert result.success
    assert "as agent-7" in result.output


def test_the_owners_own_channel_is_not_an_agents():
    pool = FakePool()
    session = _session(pool)
    assert "as %s" % OWNER_AGENT_ID in session.execute("shell", {"command": "ls"}).output


def test_an_agent_id_in_the_arguments_is_dropped():
    """A model that emits one would otherwise collide with the bound identity --
    a TypeError, surfaced to the model as a baffling tool failure."""
    pool = FakePool()
    session = _session(pool)
    session.executor_for("real")("shell", {"command": "ls", "agent_id": "spoofed"})
    tool, args, agent_id = pool.sandbox.calls[-1]
    assert agent_id == "real"
    assert "agent_id" not in args


def test_a_call_with_no_sandbox_is_an_error_not_a_crash():
    session = SandboxSession(quiet=True)
    result = session.execute("shell", {"command": "ls"})
    assert not result.success
    assert "No active sandbox" in (result.error or "")


def test_sandbox_id_is_unknown_before_spawn():
    assert SandboxSession(quiet=True).sandbox_id == "unknown"


# --- helper ----------------------------------------------------------------
@pytest.fixture
def pooled(monkeypatch):
    """Make ``create()`` reach a FakePool, exercising the real code path.

    Patches the two module-level functions ``create`` uses rather than reaching
    into the session, so what runs is the actual spawn logic -- template
    resolution off, cold start on.
    """
    import harness.execution.session as module

    def install(pool: FakePool):
        monkeypatch.setattr(module, "build_pool",
                            lambda backend, runtime_bin=None: pool)
        monkeypatch.setattr(module, "builder_from_backend", lambda backend: None)
        return pool

    return install
