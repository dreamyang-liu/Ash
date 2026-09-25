"""OpenAI-compatible probability scoring for BPO branch selection."""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import time

import httpx

from .policies import entropy_record
from .storage import fingerprint, load, rows, save


class ProbabilityUnavailable(RuntimeError):
    """The endpoint cannot support probability-based BPO for this model."""


class ChatClient:
    def __init__(self, *, timeout: float = 300., base_url: str | None = None,
                 api_key: str | None = None):
        self.base_url = (base_url or os.environ["OPENAI_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.timeout = timeout

    def complete(self, payload: dict, audit: Path) -> dict:
        started = time.time()
        record = {"request": payload, "request_sha256": fingerprint(payload),
                  "started_at": started}
        save(audit, record)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.base_url + "/chat/completions",
                                       headers={"Authorization": "Bearer " + self.api_key},
                                       json=payload)
            record["http_status"] = response.status_code
            # Never persist headers, API keys, or full exception/request objects.
            if response.status_code != 200:
                if payload.get("logprobs") and response.status_code in (400, 422):
                    raise ProbabilityUnavailable(
                        "Provider rejected logprobs request (HTTP %s); no entropy fallback"
                        % response.status_code)
                raise RuntimeError("Model endpoint returned HTTP %s" % response.status_code)
            data = response.json()
            record["response"] = data
            if data.get("error") or not data.get("choices"):
                raise ValueError("Provider returned no valid completion")
            return data
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["seconds"] = time.time() - started
            save(audit, record)


def read_provider(root: Path, request_id: str) -> dict:
    if Path(request_id).name != request_id or "/" in request_id or "\\" in request_id:
        raise ValueError("Invalid request id")
    path = root / "provider-responses" / (request_id + ".json")
    if path.exists():
        return load(path)
    with gzip.open(str(path) + ".gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def candidates_from_audit(root: Path, owner: str, checkpoints: dict,
                          model: str) -> list[dict]:
    """Map the last tool result of the actual next request to its snapshot.

    Only successful requests of THIS root are used. Token positions count all
    preceding original completion tokens, including reasoning. Missing accounting
    or mixed model names are errors, not opportunities to guess token positions.
    """
    calls = {p.call_id: step for step, p in checkpoints.items()}
    usage = [r for r in rows(root / "actor-usage.jsonl")
             if r.get("owner") == owner and r.get("status") == 200
             and not r.get("validation_error")]
    usage.sort(key=lambda r: r["time"])
    position = 0
    candidates = {}
    for row in usage:
        raw = read_provider(root, row["request_id"])
        request = raw["request"]
        if raw.get("owner") != owner or request["model"] != model:
            raise ValueError("Provider trace owner/model does not match root rollout")
        tools = [m for m in request["messages"] if m["role"] == "tool"]
        step = calls.get(tools[-1].get("tool_call_id")) if tools else None
        if step is not None and step not in candidates:
            candidates[step] = {"step": step, "token_position": position,
                                "snapshot_id": checkpoints[step].snapshot_id,
                                "request_id": row["request_id"], "request": request,
                                "native_response": raw["native_response"]}
        count = raw["native_response"].get("usage", {}).get("completion_tokens")
        if type(count) is not int or count < 0:
            raise ValueError("Original completion-token accounting is missing")
        position += count
    if not candidates:
        raise ValueError("No original provider request maps to an exact root checkpoint")
    return list(candidates.values())


def score_candidate(candidate: dict, client: ChatClient, directory: Path,
                    *, top_k: int = 5) -> dict:
    original = candidate["request"]
    identity = fingerprint({"request": original, "snapshot": candidate["snapshot_id"],
                            "top_k": top_k, "version": 1})
    cache = directory / ("step-%s.json" % candidate["step"])
    if cache.exists():
        result = load(cache)
        if result["cache_identity"] != identity:
            raise ValueError("Entropy cache differs from current root/configuration")
        return result
    native = candidate["native_response"]
    token_list = (native["choices"][0].get("logprobs") or {}).get("content") or []
    source = "original-response"
    usage = None
    if not token_list:
        source = "historical-prefix-rescore"
        request = deepcopy(original)
        request.pop("max_tokens", None)
        request.pop("stream_options", None)
        request.update(logprobs=True, top_logprobs=top_k,
                       max_completion_tokens=1, stream=False)
        result = client.complete(request, directory / ("step-%s.request.json" % candidate["step"]))
        token_list = (result["choices"][0].get("logprobs") or {}).get("content") or []
        usage = result.get("usage")
    if not token_list:
        raise ProbabilityUnavailable("Endpoint omitted requested logprobs; BPO cannot run")
    result = {key: candidate[key] for key in ("step", "token_position", "snapshot_id", "request_id")}
    result.update(entropy_record(token_list[0]), source=source, cache_identity=identity,
                  request_sha256=fingerprint(original), scoring_usage=usage,
                  entropy_kind="first-reported-content-token-top-k-plus-tail-lower-bound",
                  first_token=token_list[0], observed_logprob_tokens=len(token_list))
    save(cache, result)
    return result
