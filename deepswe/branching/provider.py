"""Audited OpenAI-compatible Shepherd meta-agent calls."""

from __future__ import annotations

import os
from pathlib import Path
import time

import httpx

from .storage import fingerprint, save


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
