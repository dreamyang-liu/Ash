"""Model-neutral Chat Completions transport for bounded benchmark reviews."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from harness.rollout import endpoint
from swebench import structured_review


class ReviewTransport:
    def __init__(self, directory: str | Path, *, model: str, model_endpoint: str,
                 api_key_env: str, max_output_tokens: int,
                 timeout_s: float = 420, reasoning_effort: str | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.model_endpoint = endpoint(model_endpoint)
        self.api_key_env = api_key_env
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.sequence = 0
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("review output budget must be positive")
        if timeout_s <= 0:
            raise ValueError("review timeout must be positive")

    def __call__(self, model: str, prompt: str) -> str:
        if model != self.model:
            raise ValueError("Review model differs from the frozen evaluation model")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"Review key environment variable {self.api_key_env} is unset")
        kind, adapted, extra = structured_review.request_format(prompt)
        if extra:
            raise ValueError("Fenced review must not introduce model parameters")
        payload = {"model": model, "messages": [{"role": "user", "content": adapted}],
                   "max_tokens": self.max_output_tokens, "stream": False}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        self.sequence += 1
        receipt = self.directory / f"review-{self.sequence:03d}.json"
        request = {"kind": kind, "model": model, "prompt": adapted,
                   "max_tokens": self.max_output_tokens,
                   "reasoning_effort": self.reasoning_effort}
        receipt.write_text(json.dumps({"request": request}, indent=2))
        try:
            response = httpx.post(
                self.model_endpoint + "/v1/chat/completions", json=payload,
                headers={"Authorization": "Bearer " + key},
                timeout=httpx.Timeout(self.timeout_s, connect=10),
            )
            response.raise_for_status()
            value = response.json()
            text = structured_review.response_text(kind, value)
        except Exception as error:
            receipt.write_text(json.dumps({"request": request,
                                           "error": f"{type(error).__name__}: {error}"}, indent=2))
            raise
        receipt.write_text(json.dumps({"request": request, "response": value,
                                       "text": text}, indent=2))
        return text
