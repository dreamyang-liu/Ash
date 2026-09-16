"""Shared tool-free Bedrock review requests; no agent or sandbox execution."""

import json
import os
import re
import urllib.request


MANTLE = "https://bedrock-mantle.%s.api.aws/openai/v1/responses"
CONVERSE = "https://bedrock-runtime.%s.amazonaws.com/model/%s/converse"
ANALYST_MAX_TOKENS = 32_000


def ask_analyst(model: str, prompt: str, region: str = "us-west-2",
                timeout: float = 300.0) -> str:
    """One analyst call. The endpoint follows from the model name.

    ``openai.*`` is Mantle's catalogue and speaks Responses; everything else is
    asked through Converse, which is how Bedrock serves Anthropic's models. The
    alternative -- one protocol plus a translator -- buys nothing here: this is a
    single request with no tools and no streaming.
    """
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not key:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK is required for the analyst")
    headers = {"Authorization": "Bearer %s" % key,
               "Content-Type": "application/json"}

    if model.startswith("openai."):
        body = json.dumps({"model": model, "input": prompt,
                           "max_output_tokens": ANALYST_MAX_TOKENS}).encode()
        request = urllib.request.Request(MANTLE % region, data=body,
                                        headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        chunks = []
        for item in payload.get("output") or []:
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text"):
                    chunks.append(part.get("text") or "")
        return "\n".join(chunks)

    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": ANALYST_MAX_TOKENS},
    }).encode()
    request = urllib.request.Request(CONVERSE % (region, model), data=body,
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    message = (payload.get("output") or {}).get("message") or {}
    return "\n".join(part.get("text") or ""
                     for part in (message.get("content") or []))


def extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("analyst returned no JSON: %r" % text[:200])
        candidate = text[start:end + 1]
    # strict=False: long hints arrive with literal newlines inside strings --
    # invalid JSON, harmless intent.
    return json.loads(candidate, strict=False)
