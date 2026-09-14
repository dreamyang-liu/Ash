"""A quarantined sandbox must stop inference, including a silent SDK stream."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from harness.core.control import RunAborted, RunControl
from harness.core.journal import read_journal
from harness.execution.pipeline import ToolPipeline
from harness.execution.interceptors import MutationTracker, default_pipeline
from harness.execution.server import Session, SessionHandler
from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec
from harness.slots.claude_code import _with_timeout
from harness.tests.test_checkpoint_identity import MemorySession


def test_abort_wakes_a_silent_stream_and_closes_it_in_its_owner_task():
    async def scenario():
        control = RunControl()
        entered = asyncio.Event()
        owners = []

        async def stream():
            owners.append(asyncio.current_task())
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                owners.append(asyncio.current_task())

        async def consume():
            async for _ in _with_timeout(stream(), 10800, control):
                pytest.fail("silent stream unexpectedly yielded")

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.to_thread(control.request_stop, "execution_uncertain at step 1")
        with pytest.raises(RunAborted, match="execution_uncertain"):
            await asyncio.wait_for(task, 2)
        assert len(owners) == 2 and owners[0] is owners[1]
    asyncio.run(scenario())


def test_normal_stream_is_not_cancelled_during_close():
    async def scenario():
        closed = []

        async def stream():
            try:
                yield "one"
                yield "two"
            finally:
                await asyncio.sleep(0)
                closed.append(True)

        assert [item async for item in _with_timeout(stream(), 2)] == ["one", "two"]
        assert closed == [True]
    asyncio.run(scenario())


def test_deadline_still_stops_a_silent_stream():
    async def scenario():
        closed = []

        async def stream():
            try:
                await asyncio.Event().wait()
                yield "never"
            finally:
                closed.append(True)

        with pytest.raises(asyncio.TimeoutError):
            async for _ in _with_timeout(stream(), .02):
                pass
        assert closed == [True]
    asyncio.run(scenario())


@pytest.mark.parametrize("failure_kind,running,exit_code", [
    ("transport_exception", False, None),
    ("runtime_timeout", False, None),
    ("runtime_timeout", True, 137),
    ("runtime_still_running", True, None),
])
def test_real_orchestrator_uncertainty_event_stops_the_sdk_driver(
        tmp_path, monkeypatch, failure_kind, running, exit_code):
    sdk = pytest.importorskip("claude_agent_sdk")
    memory = MemorySession()
    if failure_kind != "transport_exception":
        from ash_sandbox.result import ToolResult

        async def uncertain_call(name, **args):
            return ToolResult("not settled", True,
                              timed_out=failure_kind == "runtime_timeout",
                              running=running, exit_code=exit_code)
        memory.call = uncertain_call
    tracker = MutationTracker()
    server = SimpleNamespace(boundary=None)
    owned = OwnedSandbox(session=memory, server=server, tracker=tracker, sandbox_id="fake")
    entry = SimpleNamespace(id="fake", sandbox=memory, visible_to=lambda _: True)
    attempts, closed = [], []

    async def query(prompt, options):
        try:
            yield sdk.SystemMessage(subtype="init", data={"session_id": "native"})
            args = {"command": "transport-error"}
            yield sdk.AssistantMessage(content=[sdk.ToolUseBlock(
                id="c1", name="mcp__ash__shell", input=args)], model="fake")
            hook = options.hooks["PreToolUse"][0].hooks[0]
            approved = await hook({"tool_name": "mcp__ash__shell", "tool_input": args}, "c1", None)
            handler = SessionHandler(
                Session(id="test", groups=["owner:test"], bound_id="fake"),
                SimpleNamespace(get=lambda _: entry), pipeline=ToolPipeline([tracker]),
                boundary=server.boundary)
            attempts.append("c1")
            await handler.call_tool("shell", approved["hookSpecificOutput"]["updatedInput"])
            # Even when no next model message arrives, the driver must exit now,
            # not wait for the 10800-second rollout cap or keep requesting tools.
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    monkeypatch.setattr(sdk, "query", query)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *a: (owned, None))
    monkeypatch.setattr(Orchestrator, "_teardown", lambda *a: True)
    outcome = Orchestrator(out_dir=tmp_path).run(RunSpec(
        prompt="fake", slot="claude-code", journal_path=tmp_path / "parent.jsonl",
        timeout_s=3))
    assert outcome.status == "error" and "execution_uncertain" in outcome.error
    assert attempts == ["c1"] and closed == [True]
    events = read_journal(outcome.journal_path)
    assert any(e.get("type") == "agent.error" and e.get("reason") == "execution_uncertain" for e in events)
    assert len([e for e in events if e.get("type") == "tool.started"]) == 1
    failure = next(e for e in events if e.get("type") == "checkpoint.captured"
                   and e.get("reason") == "execution_uncertain")
    assert failure["execution_detail"]["kind"] == failure_kind
    if failure_kind == "transport_exception":
        assert failure["execution_detail"]["exception"] == "TimeoutError"


@pytest.mark.parametrize("continue_after_timeout", [False, True])
def test_settled_timeout_reaches_driver_and_has_gradable_exact_state(
        tmp_path, monkeypatch, continue_after_timeout):
    from ash_sandbox.result import ToolResult
    from harness.rollback import branch_checkpoints
    from swebench import fork_eval

    sdk = pytest.importorskip("claude_agent_sdk")
    memory = MemorySession()

    async def call(name, **args):
        memory.state.append(args["command"])
        if args["command"] == "partial-write":
            return ToolResult.from_response({"content": [{"type": "text", "text": json.dumps({
                "stdout": "partial output", "stderr": "partial stderr", "exit_code": 137,
                "timed_out": True, "running": False,
            })}]})
        return ToolResult("recovered", False)

    memory.call = call
    tracker = MutationTracker()
    server = SimpleNamespace(boundary=None)
    owned = OwnedSandbox(session=memory, server=server, tracker=tracker, sandbox_id="fake")
    entry = SimpleNamespace(id="fake", sandbox=memory, visible_to=lambda _: True)
    received = []
    commands = ["partial-write", "recover"] if continue_after_timeout else ["partial-write"]

    async def query(prompt, options):
        yield sdk.SystemMessage(subtype="init", data={"session_id": "native"})
        handler = SessionHandler(
            Session(id="test", groups=["owner:test"], bound_id="fake"),
            SimpleNamespace(get=lambda _: entry), pipeline=default_pipeline(extra=[tracker]),
            boundary=server.boundary)
        for index, command in enumerate(commands, 1):
            call_id = f"call-{index}"
            args = {"command": command}
            yield sdk.AssistantMessage(content=[sdk.ToolUseBlock(
                id=call_id, name="mcp__ash__shell", input=args)], model="fake",
                message_id=f"turn-{index}")
            hook = options.hooks["PreToolUse"][0].hooks[0]
            approved = await hook({"tool_name": "mcp__ash__shell", "tool_input": args}, call_id, None)
            result = await handler.call_tool("shell", approved["hookSpecificOutput"]["updatedInput"])
            received.append(result["text"])
            assert list(memory.snapshots.values())[-1] == commands[:index]
            yield sdk.UserMessage(content=[sdk.ToolResultBlock(
                tool_use_id=call_id, content=result["text"], is_error=result["isError"])])
        yield sdk.ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                                is_error=False, num_turns=len(commands), session_id="native", result="done")

    monkeypatch.setattr(sdk, "query", query)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: (owned, None))
    monkeypatch.setattr(Orchestrator, "_teardown", lambda *args: True)
    outcome = Orchestrator(out_dir=tmp_path).run(RunSpec(
        prompt="fake", slot="claude-code", journal_path=tmp_path / "parent.jsonl", timeout_s=3))
    assert outcome.status == "completed", outcome.error
    assert "[timed out]" in received[0]
    assert "partial output" in received[0] and "partial stderr" in received[0]
    assert len(received) == len(commands)
    points = branch_checkpoints(outcome.journal_path)
    assert set(points) == set(range(1, len(commands) + 1))
    assert memory.snapshots[points[1].snapshot_id] == ["partial-write"]
    assert points[1].call_id == "call-1" and points[1].reason == "captured"
    assert not any(event.get("reason") == "execution_uncertain"
                   for event in read_journal(outcome.journal_path))
    graded = []

    def grade(snapshot_id, instance, config):
        graded.append(snapshot_id)
        assert memory.snapshots[snapshot_id] == commands
        return fork_eval.Grade()

    monkeypatch.setattr(fork_eval, "backend_for", lambda *args: None)
    result = fork_eval.grade_attempt(outcome, {}, None, SimpleNamespace(grade=grade))
    assert result.error is None
    assert graded == [points[len(commands)].snapshot_id]
