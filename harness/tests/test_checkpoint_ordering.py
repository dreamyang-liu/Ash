"""Execution and capture must form one cancellation-safe sandbox boundary."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from ash_sandbox.result import ToolResult
from harness.execution.interceptors import MutationTracker, default_pipeline
from harness.execution.pipeline import ToolPipeline
from harness.execution.server import Session, SessionHandler, ToolBoundary


def make_handler(sandbox, capture):
    entry = SimpleNamespace(id="fake", sandbox=sandbox, visible_to=lambda _: True)
    boundary = ToolBoundary(capture)
    return SessionHandler(
        Session(id="test", groups=["owner:test"], bound_id="fake"),
        SimpleNamespace(get=lambda _: entry),
        pipeline=ToolPipeline([MutationTracker()]), boundary=boundary,
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_same_sandbox_execution_and_capture_cannot_overlap(cancel):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        state, snapshots = [], []

        class Sandbox:
            async def call(self, name, **args):
                command = args["command"]
                if command == "first":
                    entered.set()
                    await release.wait()
                state.append(command)
                return ToolResult(command, False)

        handler = make_handler(Sandbox(), lambda n: snapshots.append((n, list(state))))
        first = asyncio.create_task(handler.call_tool("shell", {"command": "first"}))
        await asyncio.wait_for(entered.wait(), 2)
        if cancel:
            first.cancel()
        second = asyncio.create_task(handler.call_tool("shell", {"command": "second"}))
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert "second" not in state, "next call entered before previous call settled"
        finally:
            release.set()
            outcomes = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 2)
        if cancel:
            assert isinstance(outcomes[0], asyncio.CancelledError)
        assert snapshots == [(1, ["first"]), (2, ["first", "second"])]

    asyncio.run(scenario())


@pytest.mark.parametrize("with_pipeline", [False, True])
def test_settled_timeout_capture_finishes_before_next_command(with_pipeline):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        state, snapshots = [], []

        class Sandbox:
            async def call(self, name, **args):
                state.append(args["command"])
                if args["command"] == "partial-write":
                    return ToolResult.from_response({"content": [{"type": "text", "text": json.dumps({
                        "stdout": "wrote partial output", "stderr": "", "exit_code": 137,
                        "timed_out": True, "running": False,
                    })}]})
                return ToolResult("recovered", False)

        def capture(step):
            if step == 1:
                entered.set()
                assert release.wait(2)
            snapshots.append((step, list(state)))

        handler = make_handler(Sandbox(), capture)
        handler.pipeline = default_pipeline() if with_pipeline else None
        first = asyncio.create_task(handler.call_tool("shell", {"command": "partial-write"}))
        assert await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(handler.call_tool("shell", {"command": "recover"}))
        try:
            await asyncio.sleep(.02)
            assert state == ["partial-write"]
            assert not first.done()
        finally:
            release.set()
        timeout_result, recovered = await asyncio.wait_for(asyncio.gather(first, second), 2)
        if with_pipeline:
            assert "[timed out]" in timeout_result["text"]
            assert "wrote partial output" in timeout_result["text"]
        else:
            assert json.loads(timeout_result["text"])["timed_out"] is True
        assert recovered["text"] == "recovered"
        assert snapshots == [(1, ["partial-write"]), (2, ["partial-write", "recover"])]

    asyncio.run(scenario())


@pytest.mark.parametrize("with_pipeline", [False, True])
@pytest.mark.parametrize("running,exit_code", [(True, 137), (False, None)])
def test_unsettled_timeout_still_blocks_execution_and_capture(with_pipeline, running, exit_code):
    async def scenario():
        attempts, snapshots = [], []

        class Sandbox:
            async def call(self, name, **args):
                attempts.append(args["command"])
                return ToolResult("uncertain", False, timed_out=True,
                                  running=running, exit_code=exit_code)

        handler = make_handler(Sandbox(), lambda step: snapshots.append(step))
        handler.pipeline = default_pipeline() if with_pipeline else None
        await handler.call_tool("shell", {"command": "first"})
        blocked = await handler.call_tool("shell", {"command": "second"})
        assert blocked["isError"]
        assert attempts == ["first"]
        assert snapshots == []

    asyncio.run(scenario())
