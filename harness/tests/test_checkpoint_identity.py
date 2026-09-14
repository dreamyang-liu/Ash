"""Exact pairing across the real hook, journal, handler, bridge and selector."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import warnings

import pytest

from ash_sandbox.result import ToolResult
from harness.checkpointing import SnapshotBridge
from harness.core.checkpoint_identity import CALL_IDENTITY_KEY
from harness.core.journal import JournalWriter, read_journal
from harness.execution.interceptors import MutationTracker
from harness.execution.pipeline import ToolPipeline
from harness.execution.server import Session, SessionHandler, ToolBoundary
from harness.execution.session import SandboxSession
from harness.rollback import branch_checkpoints, load_checkpoints
from harness.slots.claude_code import ClaudeCodeSlot


class MemorySession:
    def __init__(self):
        self.state = []
        self.snapshots = {}
        self.fail = False

    def supports_snapshot(self):
        return True

    def snapshot(self, **kwargs):
        if self.fail:
            return None
        sid = "snap-%d" % (len(self.snapshots) + 1)
        self.snapshots[sid] = list(self.state)
        return SimpleNamespace(id=sid, rootfs_layers=None, memory_layers=None,
                               chain_size_mb=None)

    async def call(self, name, **args):
        assert CALL_IDENTITY_KEY not in args, "private identity leaked to runtime"
        if name == "shell":
            if args["command"] == "transport-error":
                raise TimeoutError("synthetic uncertain transport")
            self.state.append(args["command"])
        return ToolResult("ok", args.get("command") == "command-error")


@pytest.fixture
def exact(tmp_path):
    journal = JournalWriter(tmp_path / "calls.jsonl")
    memory = MemorySession()
    tracker = MutationTracker()
    bridge = SnapshotBridge.install(journal, memory, tracker=tracker, exact_mode=True)
    journal.emit("session.ref", native_session_id="native")
    entry = SimpleNamespace(id="fake", sandbox=memory, visible_to=lambda _: True)
    handler = SessionHandler(
        Session(id="test", groups=["owner:test"], bound_id="fake"),
        SimpleNamespace(get=lambda _: entry), pipeline=ToolPipeline([tracker]),
        boundary=ToolBoundary(bridge.on_tool_boundary, validate_call=bridge.validate_call,
                              on_unavailable=bridge.record_unavailable))
    slot = ClaudeCodeSlot()
    slot._journal = journal
    slot._checkpoint_server = "ash"
    yield SimpleNamespace(journal=journal, memory=memory, bridge=bridge,
                          handler=handler, slot=slot, path=journal.path)
    journal.close()


async def approve(env, call_id, command="write", tool="shell", arguments=None):
    verdict = await env.slot._pre_tool_use(
        {"tool_name": "mcp__ash__" + tool,
         "tool_input": arguments if arguments is not None else {"command": command}}, call_id)
    return verdict["hookSpecificOutput"]["updatedInput"]


async def execute(env, call_id, args, tool="shell"):
    result = await env.handler.call_tool(tool, args)
    env.journal.emit("tool.finished", call_id=call_id,
                     status="error" if result["isError"] else "ok", output=result["text"])
    return result


def test_missing_request_preserves_step_and_rejects_late_delivery(exact):
    async def scenario():
        a = await approve(exact, "c1", "A")
        await execute(exact, "c1", a)
        lost = await approve(exact, "c2", "lost")
        exact.journal.emit("tool.finished", call_id="c2", status="error", output="timeout")
        c = await approve(exact, "c3", "B")
        await execute(exact, "c3", c)
        late = await exact.handler.call_tool("shell", lost)
        assert late["isError"] and "already finished" in late["text"]
        exact.bridge.finalize_calls()
        points = branch_checkpoints(exact.path)
        assert set(points) == {1, 3}
        assert points[3].call_id == "c3"
        assert exact.memory.snapshots[points[3].snapshot_id] == ["A", "B"]
        assert next(c for c in load_checkpoints(exact.path) if c.step == 2).reason == "not_executed"
    asyncio.run(scenario())


def test_hook_and_native_stream_share_one_start_record(exact):
    first = exact.journal.emit("tool.started", call_id="c1", name="mcp__ash__shell",
                               args={"command": "A"})
    args = asyncio.run(approve(exact, "c1", "A"))
    duplicate = exact.journal.emit("tool.started", call_id="c1", name="mcp__ash__shell", args=args)
    assert duplicate == first
    assert args[CALL_IDENTITY_KEY] == {"step": 1, "call_id": "c1"}
    assert len(exact.journal.tool_calls()) == 1
    assert CALL_IDENTITY_KEY not in first["args"]


def test_read_only_reuses_snapshot_without_losing_identity(exact):
    async def scenario():
        await execute(exact, "c1", await approve(exact, "c1", "A"))
        await execute(exact, "c2", await approve(exact, "c2", tool="grep_files",
                      arguments={"pattern": "A", "path": "/app"}), "grep_files")
        points = branch_checkpoints(exact.path)
        assert points[1].snapshot_id == points[2].snapshot_id
        assert points[2].reason == "clean" and points[2].call_id == "c2"
    asyncio.run(scenario())


def test_capture_failure_excluded_but_next_step_keeps_its_number(exact):
    async def scenario():
        exact.memory.fail = True
        await execute(exact, "c1", await approve(exact, "c1", "A"))
        exact.memory.fail = False
        await execute(exact, "c2", await approve(exact, "c2", "B"))
        points = branch_checkpoints(exact.path)
        assert set(points) == {2}
        assert exact.memory.snapshots[points[2].snapshot_id] == ["A", "B"]
    asyncio.run(scenario())


def test_failed_final_capture_cannot_fork_but_grades_last_saved_state(exact, monkeypatch):
    from harness.rollback import fork_plan
    from swebench import fork_eval

    graded = []

    def grade(snapshot_id, instance, config):
        graded.append(exact.memory.snapshots[snapshot_id])
        return fork_eval.Grade()

    monkeypatch.setattr(fork_eval, "backend_for", lambda *args: None)

    async def scenario():
        await execute(exact, "c1", await approve(exact, "c1", "A"))
        exact.memory.fail = True
        await execute(exact, "c2", await approve(exact, "c2", "B"))
        with pytest.raises(ValueError, match="no exact"):
            fork_plan(exact.path, 2)
        result = fork_eval.grade_attempt(SimpleNamespace(journal_path=exact.path), {}, None,
                                        SimpleNamespace(grade=grade))
        assert result.error is None
        assert graded == [["A"]]
        assert result.grading_snapshot["capture_step"] == 1
        assert result.grading_snapshot["later_checkpoint_issues"][0]["reason"] == "failed"
    asyncio.run(scenario())


def test_stdio_identity_and_failure_records_survive_tailing(exact, tmp_path, monkeypatch):
    from harness.execution.server import _stdio_checkpoints
    from harness.orchestrator.run import CheckpointTail

    async def scenario():
        monkeypatch.setattr("harness.execution.server.AttachedSandboxSession",
                            lambda *a: exact.memory)
        log = tmp_path / "stdio.jsonl"
        args = SimpleNamespace(checkpoint_log=log, checkpoint_always=False,
                               checkpoint_call_identity=True)
        exact.handler.boundary = _stdio_checkpoints(
            args, exact.handler.pipeline.interceptors[0], None, None)
        await execute(exact, "c1", await approve(exact, "c1", "A"))
        exact.memory.fail = True
        await execute(exact, "c2", await approve(exact, "c2", "B"))
        CheckpointTail(log, exact.bridge).drain()
        points = branch_checkpoints(exact.path)
        assert set(points) == {1}
        assert points[1].call_id == "c1"
        assert load_checkpoints(exact.path)[-1].reason == "failed"
        assert load_checkpoints(exact.path)[-1].call_id == "c2"
    asyncio.run(scenario())


def test_out_of_order_execution_is_not_a_false_prefix(exact):
    async def scenario():
        a = await approve(exact, "c1", "A")
        b = await approve(exact, "c2", "B")
        await execute(exact, "c2", b)
        await execute(exact, "c1", a)
        await execute(exact, "c3", await approve(exact, "c3", "C"))
        assert set(branch_checkpoints(exact.path)) == {3}
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["missing", "wrong-step", "wrong-args", "duplicate"])
def test_untrusted_or_repeated_identity_never_executes(exact, kind):
    async def scenario():
        args = await approve(exact, "c1", "A")
        if kind == "missing":
            args.pop(CALL_IDENTITY_KEY)
        elif kind == "wrong-step":
            args[CALL_IDENTITY_KEY]["step"] = 9
        elif kind == "wrong-args":
            args["command"] = "different"
        else:
            result = await exact.handler.call_tool("shell", args)
            assert not result["isError"]
        before = list(exact.memory.state)
        result = await exact.handler.call_tool("shell", args)
        assert result["isError"]
        assert exact.memory.state == before
    asyncio.run(scenario())


def test_transport_uncertainty_blocks_further_mutations_and_captures(exact):
    async def scenario():
        await execute(exact, "c1", await approve(exact, "c1", "transport-error"))
        blocked = await execute(exact, "c2", await approve(exact, "c2", "A"))
        assert blocked["isError"]
        assert exact.memory.state == [] and exact.memory.snapshots == {}
        assert branch_checkpoints(exact.path) == {}
    asyncio.run(scenario())


def test_command_failure_is_not_a_transport_failure(exact):
    async def scenario():
        await execute(exact, "c1", await approve(exact, "c1", "command-error"))
        await execute(exact, "c2", await approve(exact, "c2", "A"))
        assert set(branch_checkpoints(exact.path)) == {1, 2}
    asyncio.run(scenario())


def test_cancel_during_capture_keeps_gate_until_capture_settles(exact):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        original = exact.memory.snapshot

        def capture(**kwargs):
            entered.set()
            assert release.wait(2)
            return original(**kwargs)

        exact.memory.snapshot = capture
        a = await approve(exact, "c1", "A")
        first = asyncio.create_task(exact.handler.call_tool("shell", a))
        assert await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        b = await approve(exact, "c2", "B")
        second = asyncio.create_task(exact.handler.call_tool("shell", b))
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert exact.memory.state == ["A"]
        finally:
            release.set()
            outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 2)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        points = branch_checkpoints(exact.path)
        assert set(points) == {2}
        assert exact.memory.snapshots[points[2].snapshot_id] == ["A", "B"]
        assert load_checkpoints(exact.path)[1].reason == "cancelled"
    asyncio.run(scenario())


def test_orchestrator_enables_identity_on_actual_slot_entry(tmp_path, monkeypatch):
    from harness.core.slot import AgentSlot, SlotCapabilities, SlotResult
    from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec

    memory = MemorySession()
    server = SimpleNamespace(boundary=None)
    owned = OwnedSandbox(session=memory, server=server, sandbox_id="fake")

    class Slot(AgentSlot):
        capabilities = SlotCapabilities()

        def run(self, task, journal, mcp=None):
            assert task.extra["checkpoint_identity"] is True
            assert server.boundary.require_identity
            assert server.boundary.validate_call is not None
            journal.emit("session.ref", native_session_id="s")
            journal.emit("tool.started", call_id="lost", name="mcp__ash__shell", args={})
            return SlotResult(status="completed")

    monkeypatch.setattr("harness.slots.load_slot", lambda _: Slot)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *a: (owned, None))
    monkeypatch.setattr(Orchestrator, "_teardown", lambda *a: None)
    out = Orchestrator(out_dir=tmp_path).run(RunSpec(
        prompt="fake", slot="claude-code", journal_path=tmp_path / "run.jsonl"))
    assert out.ok, out.error
    points = load_checkpoints(out.journal_path)
    assert len(points) == 1 and points[0].reason == "not_executed"


def test_shutdown_timeout_does_not_destroy_a_still_active_sandbox():
    from harness.orchestrator.run import Orchestrator, RunSpec

    destroyed = []

    def still_active():
        raise TimeoutError("capture in progress")

    owned = SimpleNamespace(stop_server=still_active, destroy=lambda: destroyed.append(True))
    assert Orchestrator()._teardown(RunSpec(prompt="fake"), None, owned, None) is False
    assert destroyed == []


def test_closed_bridge_rejects_late_snapshot_publication(exact):
    args = asyncio.run(approve(exact, "c1", "A"))
    exact.bridge.finalize_calls()
    exact.bridge.close()
    before = list(read_journal(exact.path))
    exact.bridge.on_tool_boundary(1, call_id="c1")
    exact.bridge.record_pair(1, "late-snapshot", call_id="c1")
    assert list(read_journal(exact.path)) == before
    assert exact.memory.snapshots == {}


def test_drain_waits_for_execution_and_rejects_new_calls():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []
        boundary = ToolBoundary(lambda step: seen.append(("snapshot", step)))

        async def operation(step):
            entered.set()
            await release.wait()
            seen.append(("execute", step))
            return {"isError": False}

        first = asyncio.create_task(boundary.run_call(operation))
        await asyncio.wait_for(entered.wait(), 2)
        drain = asyncio.create_task(boundary.drain())
        await asyncio.sleep(0)
        assert not drain.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, drain), 2)
        assert (await boundary.run_call(operation))["isError"]
        assert seen == [("execute", 1), ("snapshot", 1)]
    asyncio.run(scenario())


def test_session_snapshot_workers_do_not_reenter_the_same_loop():
    entered, release = threading.Event(), threading.Event()

    class Pool:
        def supports_snapshot(self):
            return True

        async def snapshot(self, sandbox, **kwargs):
            entered.set()
            assert await asyncio.to_thread(release.wait, 2)
            return "snapshot"

    session = SandboxSession()
    session._pool, session._sandbox = Pool(), object()
    with warnings.catch_warnings(record=True) as caught, ThreadPoolExecutor(2) as pool:
        warnings.simplefilter("always", RuntimeWarning)
        first = pool.submit(session.snapshot)
        assert entered.wait(2)
        second = pool.submit(session.snapshot)
        try:
            assert not second.done()
        finally:
            release.set()
        assert first.result(2) == second.result(2) == "snapshot"
    session._drive(session._loop.shutdown_default_executor())
    session._loop.close()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
