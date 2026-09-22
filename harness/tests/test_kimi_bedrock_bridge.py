import asyncio
import io
import json

import httpx
import pytest

pytest.importorskip("botocore")
pytest.importorskip("fastapi")
from botocore.exceptions import ClientError, ReadTimeoutError

from scripts.kimi_bedrock_bridge import MODEL, create_app, main


class Client:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        value = {
            "model": "kimi-k3",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Inspecting.", "reasoning_content": "native reasoning",
                "tool_calls": [{"id": "next", "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}],
            }}],
        }
        return {"body": io.BytesIO(json.dumps(value).encode())}


class ResponseError(Exception):
    def __init__(self, response):
        super().__init__("fixture transport error")
        self.response = response


def request(app, payload, token="fixture"):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/chat/completions", json=payload,
                                     headers={"Authorization": "Bearer " + token})
    return asyncio.run(run())


@pytest.mark.parametrize("error,status,code", [
    (ReadTimeoutError(endpoint_url="https://fixture"), 504, "ReadTimeoutError"),
    (ResponseError(None), 502, "ResponseError"),
    (ResponseError({"Error": None, "ResponseMetadata": None}), 502, "ResponseError"),
    (ClientError({"Error": {"Code": "ServiceUnavailableException"},
                  "ResponseMetadata": {"HTTPStatusCode": 503}}, "InvokeModel"), 503, "ServiceUnavailableException"),
    (ClientError({"Error": {"Code": "ThrottlingException"},
                  "ResponseMetadata": {"HTTPStatusCode": 400}}, "InvokeModel"), 429, "ThrottlingException"),
    (ResponseError({"ResponseMetadata": {"HTTPStatusCode": None}}), 502, "ResponseError"),
])
def test_transport_failures_return_and_record_original_error(tmp_path, error, status, code):
    client = Client(error)
    app = create_app(client=client, token="fixture", records=tmp_path)
    response = request(app, {"model": MODEL, "messages": [{"role": "user", "content": "fixture"}]})
    assert response.status_code == status
    assert response.json()["error"]["type"] == code
    assert len(client.calls) == 1
    rows = [json.loads(line) for line in (tmp_path / "bridge-events.jsonl").read_text().splitlines()]
    assert rows[-1]["type"] == "chat.error" and rows[-1]["status"] == status
    assert rows[-1]["error_type"] == code and rows[-1]["elapsed_s"] >= 0
    assert not list(tmp_path.glob("*.response.json"))


def test_native_history_and_injected_content_survive_without_converse(tmp_path):
    client = Client()
    app = create_app(client=client, token="fixture", records=tmp_path)
    messages = [
        {"role": "user", "content": "Inspect the repository."},
        {"role": "assistant", "content": "Checking.", "reasoning_content": "retained reasoning",
         "tool_calls": [{"id": "existing", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]},
        {"role": "tool", "tool_call_id": "existing", "content": "/app"},
        {"role": "assistant", "content": "I will inspect the source before editing.",
         "tool_calls": [{"id": "injected", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command":"git status"}'}}]},
        {"role": "tool", "tool_call_id": "injected", "content": "clean"},
    ]
    response = request(app, {"model": MODEL, "messages": messages})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["reasoning_content"] == "native reasoning"
    native = json.loads(client.calls[0]["body"])
    assert client.calls[0]["modelId"] == MODEL and native["messages"] == messages
    assert native["reasoning_effort"] == "max" and native["max_tokens"] == 64000


@pytest.mark.parametrize("change,token,status", [
    ({}, "wrong", 401), ({"model": "other-model"}, "fixture", 400),
    ({"reasoning_effort": "high"}, "fixture", 400), ({"max_tokens": 64001}, "fixture", 400),
])
def test_invalid_request_does_not_reach_provider(tmp_path, change, token, status):
    client = Client()
    app = create_app(client=client, token="fixture", records=tmp_path)
    response = request(app, {"model": MODEL, "messages": [], **change}, token)
    assert response.status_code == status and not client.calls


def test_timeout_argument_checked_before_loading_credentials(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--token-file", str(tmp_path / "missing"), "--records", str(tmp_path), "--read-timeout", "0"])
    assert error.value.code == 2


def test_cli_passes_explicit_profile_timeout_and_server_settings(tmp_path, monkeypatch):
    boto3 = pytest.importorskip("boto3")
    uvicorn = pytest.importorskip("uvicorn")
    seen = {}

    class Session:
        def __init__(self, **kwargs):
            seen["session"] = kwargs

        def client(self, service, config):
            seen.update(service=service, config=config)
            return Client()

    monkeypatch.setattr(boto3, "Session", Session)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: seen.update(server=kwargs))
    token = tmp_path / "token"
    token.write_text("fixture\n")
    main(["--token-file", str(token), "--records", str(tmp_path / "records"),
          "--profile", "fixture-profile", "--region", "us-west-2",
          "--read-timeout", "480", "--port", "18241"])
    assert seen["session"] == {"profile_name": "fixture-profile", "region_name": "us-west-2"}
    assert seen["service"] == "bedrock-runtime"
    assert seen["config"].read_timeout == 480
    assert seen["config"].retries == {"total_max_attempts": 1}
    assert seen["server"] == {"host": "127.0.0.1", "port": 18241}
