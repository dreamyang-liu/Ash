"""Disconnects and bounded admission must not leave orphaned model calls."""
import asyncio
import json

import pytest

from deepswe.branching.bridge import create_app


class Request:
    headers = {"x-api-key": "branchbench:task/bpo/branch"}

    def __init__(self, disconnected=False):
        self.disconnected = disconnected

    async def json(self):
        return {"messages": [{"role": "user", "content": "task"}], "stream": False}

    async def is_disconnected(self):
        return self.disconnected


def setup_bridge(monkeypatch, tmp_path, **options):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    app = create_app("model", tmp_path, concurrency=1, disconnect_poll=.001, **options)
    return next(route.endpoint for route in app.routes if route.path == "/v1/messages")


def blocking_client(monkeypatch):
    import httpx
    state = {"started": 0, "cancelled": 0}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            state["started"] += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                state["cancelled"] += 1
                raise

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    return state


def audits(tmp_path):
    return [json.loads(path.read_text()) for path in (tmp_path / "provider-responses").glob("*.json")]


def test_disconnect_cancels_upstream_and_releases_slot(monkeypatch, tmp_path):
    endpoint = setup_bridge(monkeypatch, tmp_path)
    state = blocking_client(monkeypatch)

    async def run():
        for _ in range(2):
            response = await asyncio.wait_for(endpoint(Request(True)), 1)
            assert response.status_code == 499
        assert state == {"started": 2, "cancelled": 2}

    asyncio.run(run())
    for record in audits(tmp_path):
        assert record["validation_error"] == "CancelledError"
        assert record["finished_stage"] == "upstream"
        assert record["usage_unknown"] is True
        assert "private-test-key" not in json.dumps(record)


@pytest.mark.parametrize("disconnect", [True, False])
def test_queued_disconnect_or_deadline_never_calls_upstream(monkeypatch, tmp_path, disconnect):
    endpoint = setup_bridge(monkeypatch, tmp_path, queue_timeout=.02)
    state = blocking_client(monkeypatch)

    async def run():
        first = asyncio.create_task(endpoint(Request()))
        try:
            for _ in range(100):
                if state["started"]:
                    break
                await asyncio.sleep(.001)
            assert state["started"] == 1
            response = await asyncio.wait_for(endpoint(Request(disconnect)), 1)
            assert response.status_code == (499 if disconnect else 502)
            assert state["started"] == 1
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)

    asyncio.run(run())
    queued = next(row for row in audits(tmp_path) if row["finished_stage"] == "queued")
    assert queued["usage_unknown"] is False
    assert queued["validation_error"] == ("CancelledError" if disconnect else "TimeoutError")
    assert state["cancelled"] == 1


def test_total_upstream_deadline_cancels_and_releases_slot(monkeypatch, tmp_path):
    endpoint = setup_bridge(monkeypatch, tmp_path, timeout=.02)
    state = blocking_client(monkeypatch)

    async def run():
        for _ in range(2):
            response = await asyncio.wait_for(endpoint(Request()), 1)
            assert response.status_code == 502
        assert state == {"started": 2, "cancelled": 2}

    asyncio.run(run())
    assert all(row["validation_error"] == "TimeoutError" for row in audits(tmp_path))


@pytest.mark.parametrize("stream", [False, True])
def test_real_http_disconnect_cancels_model_call(monkeypatch, tmp_path, stream):
    import httpx
    import socket
    import uvicorn

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    real_client = httpx.AsyncClient
    state = blocking_client(monkeypatch)
    app = create_app("model", tmp_path, disconnect_poll=.01)

    async def run():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        request = None
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(.01)
            assert server.started
            async with real_client(trust_env=False) as client:
                request = asyncio.create_task(client.post(
                    "http://127.0.0.1:%d/v1/messages" % port,
                    headers=Request.headers,
                    json={"messages": [{"role": "user", "content": "task"}], "stream": stream}))
                for _ in range(200):
                    if state["started"]:
                        break
                    await asyncio.sleep(.01)
                assert state["started"] == 1
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
                for _ in range(200):
                    if state["cancelled"]:
                        break
                    await asyncio.sleep(.01)
                assert state["cancelled"] == 1
        finally:
            if request:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            server.should_exit = True
            await asyncio.wait_for(serving, 5)
            sock.close()

    asyncio.run(run())
    assert audits(tmp_path)[0]["validation_error"] == "CancelledError"


@pytest.mark.parametrize("statuses, expected", [([429, 502, 200], 200), ([503, 503, 503], 502), ([403], 502)])
def test_transient_retries_are_bounded_and_audited(monkeypatch, tmp_path, statuses, expected):
    import httpx

    endpoint = setup_bridge(monkeypatch, tmp_path, retry_delay=0)
    seen = []

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            status = statuses[len(seen)]
            seen.append(status)
            return httpx.Response(status, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                                                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    response = asyncio.run(endpoint(Request()))
    assert (200 if isinstance(response, dict) else response.status_code) == expected
    assert seen == statuses
    assert [a["status"] for a in audits(tmp_path)[0]["attempts"]] == statuses
