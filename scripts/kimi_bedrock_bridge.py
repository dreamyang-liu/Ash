"""Serve mini's chat protocol using native Kimi K3 InvokeModel payloads."""

import argparse
import asyncio
import hmac
import json
from pathlib import Path
import time
from uuid import uuid4

from botocore.exceptions import ConnectTimeoutError, ReadTimeoutError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MODEL = "us.moonshotai.kimi-k3"


def error_details(error):
    """Transport errors may have response=None rather than an AWS error body."""
    raw = getattr(error, "response", None)
    details = raw if isinstance(raw, dict) else {}
    raw_info = details.get("Error")
    info = raw_info if isinstance(raw_info, dict) else {}
    raw_metadata = details.get("ResponseMetadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    code = info.get("Code") or type(error).__name__
    status = metadata.get("HTTPStatusCode", 502)
    if type(status) is not int or not 400 <= status <= 599:
        status = 502
    if isinstance(error, (ReadTimeoutError, ConnectTimeoutError, TimeoutError)):
        status = 504
    elif code in {"ThrottlingException", "TooManyRequestsException"}:
        status = 429
    return str(code), status


def native_payload(payload):
    if payload.get("model") != MODEL:
        raise HTTPException(400, "Only the pinned Kimi K3 model is allowed")
    if payload.get("stream"):
        raise HTTPException(400, "This bridge requires complete non-streaming replies")
    limit = payload.get("max_tokens", 64000)
    if type(limit) is not int or not 1 <= limit <= 64000:
        raise HTTPException(400, "max_tokens must be an integer in 1..64000")
    if payload.get("reasoning_effort", "max") != "max":
        raise HTTPException(400, "This Kimi K3 bridge requires max reasoning effort")
    allowed = {
        "model", "messages", "tools", "tool_choice", "max_tokens", "stream",
        "reasoning_effort", "temperature", "top_p", "top_k", "stop", "parallel_tool_calls",
    }
    if set(payload) - allowed:
        raise HTTPException(400, "Unsupported request fields")
    native = {k: v for k, v in payload.items() if k not in {"model", "stream"}}
    # Avoid a Converse conversion: retain reasoning_content, ordinary injected
    # assistant content, tool calls and their correlation IDs.
    native["messages"] = [
        {k: v for k, v in message.items() if v is not None}
        for message in payload["messages"]
    ]
    native.update(max_tokens=limit, reasoning_effort="max")
    return native


def create_app(*, client, token, records, read_timeout_seconds=360):
    """Inject the SDK client so importing this module never reads credentials."""
    if not token:
        raise ValueError("A non-empty bridge token is required")
    records = Path(records)
    records.mkdir(parents=True, exist_ok=True)
    app = FastAPI()

    def log(**row):
        with (records / "bridge-events.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": time.time(), **row}) + "\n")

    def invoke(native):
        response = client.invoke_model(
            modelId=MODEL, body=json.dumps(native),
            contentType="application/json", accept="application/json",
        )
        return json.loads(response["body"].read())

    @app.get("/health")
    async def health():
        return {
            "status": "ok", "model": MODEL, "reasoning_effort": "max", "max_tokens": 64000,
            "upstream_api": "InvokeModel", "read_timeout_seconds": read_timeout_seconds,
            "automatic_retries": 0,
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(401, "Invalid bridge token")
        native = native_payload(await request.json())
        identifier = uuid4().hex
        started = time.monotonic()
        (records / (identifier + ".request.json")).write_text(json.dumps(native, indent=2) + "\n")
        log(type="bedrock.wire", request_id=identifier, model=MODEL,
            effort="max", max_tokens=native["max_tokens"])
        try:
            value = await asyncio.to_thread(invoke, native)
            if len(value.get("choices", [])) != 1:
                raise ValueError("Expected exactly one assistant choice")
            value["model"] = MODEL
            response = JSONResponse(value)
            (records / (identifier + ".response.json")).write_text(json.dumps(value, indent=2) + "\n")
            log(type="chat.done", request_id=identifier, status=200,
                elapsed_s=time.monotonic() - started)
            return response
        except Exception as error:
            code, status = error_details(error)
            log(type="chat.error", request_id=identifier, status=status, error_type=code,
                error=str(error)[:1500], elapsed_s=time.monotonic() - started)
            return JSONResponse(
                {"error": {"type": code, "message": str(error)[:1500]}}, status_code=status,
            )

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default=None, help="Optional AWS profile; otherwise use the SDK credential chain")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18239)
    parser.add_argument("--read-timeout", type=int, default=360)
    args = parser.parse_args(argv)
    if args.read_timeout < 1:
        parser.error("--read-timeout must be positive")
    import boto3
    from botocore.config import Config
    import uvicorn

    client = boto3.Session(profile_name=args.profile, region_name=args.region).client(
        "bedrock-runtime",
        config=Config(read_timeout=args.read_timeout, connect_timeout=10,
                      max_pool_connections=64, retries={"total_max_attempts": 1}),
    )
    app = create_app(client=client, token=args.token_file.read_text().strip(),
                     records=args.records, read_timeout_seconds=args.read_timeout)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
