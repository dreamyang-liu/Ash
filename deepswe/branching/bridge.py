"""Local Claude Code Messages -> OpenAI chat bridge with native request audits.

Run one instance per model/cohort. Its upstream credentials stay in environment
variables; the SDK key is an experiment owner label, never an upstream key.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import httpx

from .storage import save


def text_content(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if any(block.get("type") != "text" for block in content):
        raise ValueError("This coding benchmark supports only text tool results")
    return "\n".join(block["text"] for block in content)


def to_chat(body: dict, model: str, effort: str, extra: dict | None = None) -> dict:
    messages = []
    system = text_content(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for message in body["messages"]:
        role = message["role"]
        blocks = message["content"]
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if role == "system":
            messages.append({"role": "system", "content": text_content(blocks)})
        elif role == "assistant":
            item = {"role": "assistant", "content": None}
            text, thinking, calls = [], [], []
            for block in blocks:
                kind = block["type"]
                if kind == "text":
                    text.append(block["text"])
                elif kind == "thinking":
                    thinking.append(block["thinking"])
                elif kind == "tool_use":
                    calls.append({"id": block["id"], "type": "function", "function": {
                        "name": block["name"], "arguments": json.dumps(block["input"])}})
                else:
                    raise ValueError("Unsupported assistant block: " + kind)
            item["content"] = "\n".join(text) or None
            if thinking:
                item["reasoning_content"] = "\n".join(thinking)
            if calls:
                item["tool_calls"] = calls
            # The SDK may split thinking, speech, and a tool call into adjacent
            # assistant messages. Preserve them in a single provider turn.
            if messages and messages[-1]["role"] == "assistant":
                previous = messages[-1]
                for key in ("content", "reasoning_content"):
                    value = "\n".join(x for x in (previous.get(key), item.get(key)) if x)
                    if value:
                        previous[key] = value
                if calls:
                    previous.setdefault("tool_calls", []).extend(calls)
            else:
                messages.append(item)
        elif role == "user":
            pending = []
            for block in blocks:
                if block["type"] == "text":
                    pending.append(block["text"])
                elif block["type"] == "tool_result":
                    if pending:
                        messages.append({"role": "user", "content": "\n".join(pending)})
                        pending = []
                    messages.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                     "content": text_content(block.get("content"))})
                else:
                    raise ValueError("Unsupported user block: " + block["type"])
            if pending:
                messages.append({"role": "user", "content": "\n".join(pending)})
        else:
            raise ValueError("Unsupported message role: " + role)
    payload = {"model": model, "messages": messages, "reasoning_effort": effort,
               "temperature": 1., "top_p": .95, "stream": False,
               "max_tokens": min(body.get("max_tokens", 32768), 32768)}
    if body.get("tools"):
        payload["tools"] = [{"type": "function", "function": {
            "name": tool["name"], "description": tool.get("description", ""),
            "parameters": tool["input_schema"]}} for tool in body["tools"]]
    choice = body.get("tool_choice") or {}
    if choice.get("type") in ("auto", "none"):
        payload["tool_choice"] = choice["type"]
    elif choice.get("type") == "any":
        payload["tool_choice"] = "required"
    elif choice.get("type") == "tool":
        payload["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if extra:
        protected = {"messages", "model", "stream", "tools", "reasoning_effort"}
        if protected.intersection(extra):
            raise ValueError("Provider extras override a protected request field")
        payload.update(extra)
    return payload


def from_chat(data: dict, model: str) -> dict:
    if data.get("error") or not data.get("choices"):
        raise ValueError("Missing valid provider completion")
    choice = data["choices"][0]
    finish = choice.get("finish_reason")
    if finish not in ("stop", "tool_calls", "length"):
        raise ValueError("Unexpected provider finish reason")
    message = choice["message"]
    content = []
    if message.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": message["reasoning_content"],
                        "signature": ""})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        arguments = json.loads(call["function"]["arguments"])
        if not isinstance(arguments, dict) or not call.get("id"):
            raise ValueError("Invalid provider tool call")
        content.append({"type": "tool_use", "id": call["id"],
                        "name": call["function"]["name"], "input": arguments})
    if finish != "length" and not any(b["type"] in ("text", "tool_use") for b in content):
        raise ValueError("Completed response contains neither text nor a tool call")
    reason = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
    if finish == "length":
        reason = "max_tokens"
    usage = data.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"id": data.get("id", "msg_" + uuid4().hex), "type": "message",
            "role": "assistant", "model": model, "content": content,
            "stop_reason": reason, "stop_sequence": None,
            "usage": {"input_tokens": max(0, usage.get("prompt_tokens", 0) - cached),
                      "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0,
                      "output_tokens": usage.get("completion_tokens", 0)}}


def event(kind: str, **body) -> str:
    return "event: " + kind + "\ndata: " + json.dumps(dict(type=kind, **body)) + "\n\n"


def message_events(message: dict):
    yield event("message_start", message=dict(message, content=[], stop_reason=None))
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        if kind == "text":
            start, delta = {"type": kind, "text": ""}, {"type": "text_delta", "text": block["text"]}
        elif kind == "thinking":
            start = {"type": kind, "thinking": "", "signature": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        else:
            start = dict(block, input={})
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        yield event("content_block_start", index=index, content_block=start)
        yield event("content_block_delta", index=index, delta=delta)
        if kind == "thinking":
            yield event("content_block_delta", index=index,
                        delta={"type": "signature_delta", "signature": ""})
        yield event("content_block_stop", index=index)
    yield event("message_delta", delta={"stop_reason": message["stop_reason"], "stop_sequence": None},
                usage={"output_tokens": message["usage"]["output_tokens"]})
    yield event("message_stop")


def create_app(model: str, audit_root: Path, *, effort: str = "high",
               extra: dict | None = None, concurrency: int = 2, timeout: float = 1800.,
               queue_timeout: float = 30., disconnect_poll: float = 1.,
               max_attempts: int = 3, retry_delay: float = 2.):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    if (concurrency < 1 or max_attempts < 1 or retry_delay < 0
            or min(timeout, queue_timeout, disconnect_poll) <= 0):
        raise ValueError("Bridge concurrency and deadlines must be positive")
    app = FastAPI()
    bridge_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    semaphore = asyncio.Semaphore(concurrency)
    audit_root.mkdir(parents=True, exist_ok=True)
    base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
    api_key = os.environ["OPENAI_API_KEY"]

    @app.get("/health")
    async def health():
        return {"model": model, "effort": effort, "audit_root": str(audit_root.resolve()),
                "concurrency": concurrency, "queue_timeout": queue_timeout,
                "upstream_timeout": timeout, "cancel_on_disconnect": True,
                "max_upstream_attempts": max_attempts,
                "bridge_sha256": bridge_sha256}

    @app.post("/v1/messages/count_tokens")
    async def count(request: Request):
        # Claude uses this only for its context guard, not benchmark accounting.
        return {"input_tokens": max(1, len(json.dumps(await request.json())) // 3)}

    async def complete(body: dict, owner: str, *, openai_wire: bool = False) -> dict:
        payload = dict(body) if openai_wire else to_chat(body, model, effort, extra)
        if openai_wire:
            payload.update(model=model, reasoning_effort=effort, stream=False)
            payload.update(extra or {})
        request_id = uuid4().hex
        started = time.time()
        record = {"request_id": request_id, "owner": owner, "time": started,
                  "model": model, "status": None}
        audit_path = audit_root / "provider-responses" / (request_id + ".json")
        audit = {"owner": owner, "request": payload, "started_at": started,
                 "client_stream": bool(body.get("stream")), "stage": "queued", "attempts": []}
        save(audit_path, audit)
        acquired = False
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=queue_timeout)
            acquired = True
            record["queue_seconds"] = time.time() - started
            audit.update(stage="upstream", acquired_at=time.time())
            save(audit_path, audit)
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(timeout=timeout) as client:
                    for number in range(1, max_attempts + 1):
                        attempt = {"number": number, "started_at": time.time()}
                        audit["attempts"].append(attempt)
                        save(audit_path, audit)
                        try:
                            response = await client.post(base_url + "/chat/completions", json=payload,
                                                        headers={"Authorization": "Bearer " + api_key})
                            attempt["status"] = response.status_code
                            if response.status_code not in (429, 502, 503, 504) or number == max_attempts:
                                break
                        except httpx.TransportError as exc:
                            attempt.update(error_type=type(exc).__name__, usage_unknown=True)
                            if number == max_attempts:
                                raise
                        finally:
                            attempt["seconds"] = time.time() - attempt["started_at"]
                            save(audit_path, audit)
                        await asyncio.sleep(retry_delay * 2 ** (number - 1))
            record["status"] = response.status_code
            if response.status_code != 200:
                raise RuntimeError("Upstream HTTP %s" % response.status_code)
            data = response.json()
            audit["native_response"] = data
            save(audit_path, audit)
            record["usage"] = data.get("usage")
            return data if openai_wire else from_chat(data, model)
        except asyncio.CancelledError:
            record["validation_error"] = "CancelledError"
            raise
        except Exception as exc:
            record["validation_error"] = type(exc).__name__
            raise
        finally:
            if acquired:
                semaphore.release()
            else:
                record["queue_seconds"] = time.time() - started
            record["seconds"] = time.time() - started
            record["usage_unknown"] = (acquired and not bool(record.get("usage"))) or any(
                attempt.get("usage_unknown", False) for attempt in audit["attempts"])
            record["upstream_attempts"] = len(audit["attempts"])
            audit.update({key: record[key] for key in (
                "status", "seconds", "queue_seconds", "validation_error", "usage_unknown") if key in record})
            audit["finished_stage"] = audit["stage"]
            audit["stage"] = "finished"
            save(audit_path, audit)
            # One uvicorn process owns this file; no await inside the append.
            with (audit_root / "actor-usage.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        owner = request.headers.get("x-api-key", "")
        if not owner.startswith("branchbench:"):
            return JSONResponse({"error": {"type": "authentication_error",
                                           "message": "Expected branchbench owner label"}}, status_code=401)
        if not body.get("stream"):
            task = asyncio.create_task(complete(body, owner))
            try:
                while not task.done():
                    await asyncio.wait({task}, timeout=disconnect_poll)
                    if not task.done() and await request.is_disconnected():
                        return JSONResponse({"error": {"type": "api_error", "message": "Client disconnected"}},
                                            status_code=499)
                return await task
            except Exception as exc:
                return JSONResponse({"error": {"type": "api_error", "message": type(exc).__name__}},
                                    status_code=502)
            finally:
                # Non-streaming ASGI handlers survive client disconnects unless
                # cancelled explicitly, including queued compaction retries.
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        async def stream():
            task = asyncio.create_task(complete(body, owner))
            try:
                while not task.done():
                    yield event("ping")
                    await asyncio.wait({task}, timeout=10)
                message = await task
                for item in message_events(message):
                    yield item
            except Exception as exc:
                yield event("error", error={"type": "api_error", "message": type(exc).__name__})
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        authorization = request.headers.get("authorization", "")
        owner = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        if not owner.startswith("branchbench:"):
            return JSONResponse({"error": {"type": "authentication_error",
                                           "message": "Expected branchbench owner label"}}, status_code=401)
        if body.get("stream"):
            return JSONResponse({"error": {"type": "invalid_request_error",
                                           "message": "mini bridge requires stream=false"}}, status_code=400)
        task = asyncio.create_task(complete(body, owner, openai_wire=True))
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=disconnect_poll)
                if not task.done() and await request.is_disconnected():
                    return JSONResponse({"error": {"type": "api_error",
                                                   "message": "Client disconnected"}}, status_code=499)
            return await task
        except Exception as exc:
            return JSONResponse({"error": {"type": "api_error", "message": type(exc).__name__}},
                                status_code=502)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # FastAPI resolves postponed annotation names against module globals.
    return app


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18187)
    parser.add_argument("--effort", default="high")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--queue-timeout", type=float, default=30.)
    parser.add_argument("--upstream-timeout", type=float, default=1800.)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--extra-json", default="{}")
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.audit_root, effort=args.effort,
                           extra=json.loads(args.extra_json), concurrency=args.concurrency,
                           queue_timeout=args.queue_timeout, timeout=args.upstream_timeout,
                           max_attempts=args.max_attempts),
                host="127.0.0.1", port=args.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
