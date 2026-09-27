import json
import threading

import httpx
import pytest

from harness.core.control import RunAborted, RunControl
from harness.core.events import Usage
from harness.core.journal import JournalWriter, read_journal
from harness.gateway.routing import ModelRoute, RoutingTable
from harness.gateway.server import GatewayServer, _absorb_usage
from harness.rollout import RolloutControls
from harness.tests.test_gateway import upstream
from runstore.message_export import export_tools


def test_chat_sampling_tools_auth_and_limits_reach_the_upstream(upstream, tmp_path):
    import time

    tool = {"type": "function", "function": {
        "name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}
    with JournalWriter(tmp_path / "journal.jsonl") as journal:
        control = RunControl()
        contract = {"message_export": True, "model_endpoint": upstream.base_url, "max_turns": 1,
                    "deadline_at": time.time() + 30, "model": "policy",
                    "sampling_params": {"temperature": .7, "top_p": .9, "top_k": 20,
                                        "stop": ["END"], "max_new_tokens": 42}}
        controls = RolloutControls(contract, journal, control)
        table = RoutingTable({"default": ModelRoute(base_url=upstream.base_url, api_key="upstream-only")})
        with GatewayServer(table, journal=journal, request_policy=controls) as gateway:
            token = table.mint("mini")
            headers = {"Authorization": "Bearer " + token.token}
            payload = {"model": "alias", "messages": [{"role": "user", "content": "task"}], "tools": [tool]}
            response = httpx.post(gateway.base_url + "/v1/chat/completions", json=payload, headers=headers)
            assert response.status_code == 200
            sent = upstream.requests[0]
            assert sent["headers"]["Authorization"] == "Bearer upstream-only"
            assert sent["body"]["model"] == "policy"
            assert sent["body"]["top_k"] == 20 and sent["body"]["max_tokens"] == 42
            assert sent["body"]["stop"] == ["END"] and sent["body"]["temperature"] == .7
            assert sent["body"]["tools"] == [tool]
            refused = httpx.post(gateway.base_url + "/v1/chat/completions", json=payload, headers=headers)
            assert refused.status_code == 400 and len(upstream.requests) == 1
            assert control.stop_reason == "max_turns_reached"
    assert export_tools(read_journal(tmp_path / "journal.jsonl")) == [tool]


def test_chat_usage_preserves_cache_and_reasoning():
    usage = Usage()
    _absorb_usage({"prompt_tokens": 11, "completion_tokens": 7,
                   "prompt_tokens_details": {"cached_tokens": 3},
                   "completion_tokens_details": {"reasoning_tokens": 5}}, usage)
    assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens,
            usage.reasoning_output_tokens) == (11, 7, 3, 5)


def test_stop_cancels_silent_http_transport(monkeypatch):
    import asyncio
    from harness.core.http import post

    started, closed = threading.Event(), threading.Event()
    control = RunControl()

    async def silent(*args, **kwargs):
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(httpx.AsyncClient, "post", silent)
    stopper = threading.Thread(target=lambda: (started.wait(2), control.request_stop("cutoff")))
    stopper.start()
    try:
        with pytest.raises(RunAborted, match="cutoff"):
            post("http://example.invalid", timeout_s=30, control=control)
        assert closed.is_set()
    finally:
        stopper.join(2)


def test_mcp_preserves_real_exit_code_without_polluting_text_content():
    from harness.execution.server import _tool_response

    wire = _tool_response("call", {"type": "text", "text": "output", "isError": True,
                                 "_command_outcome": {"exit_code": 42, "stdout": "output"},
                                 "_execution_uncertain": {"kind": "private"}})
    assert wire["result"]["content"] == [{"type": "text", "text": "output"}]
    assert wire["result"]["structuredContent"]["command_outcome"]["exit_code"] == 42
    assert wire["result"]["isError"] is True


def test_plain_success_metadata_is_specific_to_foreground_shell():
    from ash_sandbox.result import ToolResult
    from harness.execution.server import _runtime_result

    sdk = ToolResult("stdout\n", False)
    command = _runtime_result("shell", {}, sdk)
    assert command.outcome.exit_code == 0 and command.outcome.stdout == "stdout\n"
    assert _runtime_result("shell", {"background": True}, sdk).outcome is None
    assert _runtime_result("text_editor", {}, sdk).outcome is None
    assert _runtime_result("shell", {}, ToolResult("rejected", True)).outcome is None


def test_gateway_stop_closes_the_real_upstream_connection():
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from fastapi import FastAPI, Request
    from rl_driver.tests.test_http_postgres import serving

    app = FastAPI()
    entered, disconnected = threading.Event(), threading.Event()
    control = RunControl()

    @app.post("/v1/chat/completions")
    async def silent(request: Request):
        entered.set()
        for _ in range(200):
            if await request.is_disconnected():
                disconnected.set()
                return {}
            await asyncio.sleep(.01)
        return {}

    class Policy:
        def __init__(self):
            self.control = control
        def prepare_model_request(self, payload, shape):
            return payload

    with serving(app) as url:
        table = RoutingTable({"default": ModelRoute(base_url=url)})
        with GatewayServer(table, request_policy=Policy(), require_token=False) as gateway:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(httpx.post, gateway.base_url + "/v1/chat/completions",
                                         json={"model": "fixture", "messages": []}, timeout=5)
                assert entered.wait(2)
                control.request_stop("rollout deadline", stop_reason="timeout")
                assert future.result(timeout=3).status_code == 502
                assert disconnected.wait(2), "upstream generation connection survived rollout stop"
