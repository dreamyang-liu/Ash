"""Small HTTP client; completed results remain durable and repeatably readable."""

from __future__ import annotations

import time

import httpx

from runstore.specs import JobSpec


class Client:
    def __init__(self, url: str, token: str) -> None:
        self.http = httpx.Client(base_url=url.rstrip("/"),
                                 headers={"Authorization": "Bearer " + token}, timeout=30)

    def close(self) -> None:
        self.http.close()

    def submit(self, job: JobSpec, idempotency_key: str) -> str:
        response = self.http.post("/v1/jobs", json=job.validate(),
                                  headers={"Idempotency-Key": idempotency_key})
        response.raise_for_status()
        return response.json()["id"]

    def get(self, job_id: str) -> dict:
        response = self.http.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def get_result(self, job_id: str) -> dict:
        response = self.http.get(f"/v1/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()

    def events(self, job_id: str, *, attempt_id: str | None = None, after: int = 0) -> list[dict]:
        params = {"after": after}
        if attempt_id:
            params["attempt_id"] = attempt_id
        response = self.http.get(f"/v1/jobs/{job_id}/events", params=params)
        response.raise_for_status()
        return response.json()

    def recovery_points(self, job_id: str) -> list[dict]:
        response = self.http.get(f"/v1/jobs/{job_id}/recovery-points")
        response.raise_for_status()
        return response.json()

    def branch(self, job_id: str, point_id: str, *, idempotency_key: str,
               context: dict | None = None, **overrides) -> str:
        body = {"point_id": point_id, "overrides": overrides}
        if context is not None:
            body["context"] = context
        response = self.http.post(f"/v1/jobs/{job_id}/branch",
                                  json=body,
                                  headers={"Idempotency-Key": idempotency_key})
        response.raise_for_status()
        return response.json()["id"]

    def query_prefix(self, scope: dict, calls: list[dict]) -> dict:
        response = self.http.post("/v1/prefix/query", json={"scope": scope, "calls": calls})
        response.raise_for_status()
        return response.json()

    def wait(self, job_id: str, timeout_s: float = 3600, interval_s: float = 1) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            result = self.get_result(job_id)
            if result["ready"] or result["state"] == "quarantined":
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("Client wait timed out; job remains queued/running")
            time.sleep(min(interval_s, max(0, deadline - time.monotonic())))
