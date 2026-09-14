"""Miles wire client and a separate execution-debugging client."""

import time

import httpx

from rl_driver.backend import segment


class Client:
    """Submit the unchanged RolloutGroupRequest and read ash-rollout-v2 results."""

    def __init__(self, url: str, token: str | None = None):
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=30,
                                 headers={"Authorization": "Bearer " + token} if token else {})

    def close(self):
        self.http.close()

    def submit(self, request: dict) -> str:
        response = self.http.post("/rollout-groups", json=request)
        response.raise_for_status()
        return response.json()["rollout_job_id"]

    def get(self, group_id: str) -> dict:
        response = self.http.get(f"/rollout-groups/{segment(group_id)}")
        response.raise_for_status()
        return response.json()

    def wait(self, group_id: str, *, timeout_s: float = 3600, interval_s: float = 1) -> dict:
        if timeout_s <= 0 or interval_s <= 0:
            raise ValueError("Wait timeout and interval must be positive")
        deadline = time.monotonic() + timeout_s
        while True:
            result = self.get(group_id)
            if result["status"] not in {"queued", "running"}:
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Stopped waiting; rollout execution is unchanged")
            time.sleep(min(interval_s, remaining))

    def execution(self, group_id: str) -> dict:
        response = self.http.get(f"/miles-executions/{segment(group_id)}")
        response.raise_for_status()
        return response.json()

    def release(self, group_id: str) -> dict:
        response = self.http.delete(f"/rollout-groups/{segment(group_id)}")
        response.raise_for_status()
        return response.json()


class ExecutionClient:
    def __init__(self, url: str, token: str):
        self.http = httpx.Client(base_url=url.rstrip("/"),
                                 headers={"Authorization": "Bearer " + token}, timeout=30)

    def close(self):
        self.http.close()

    def submit(self, request: dict) -> str:
        response = self.http.post("/execution-groups", json=request)
        response.raise_for_status()
        return response.json()["rollout_job_id"]

    def get(self, group_id: str) -> dict:
        response = self.http.get(f"/execution-groups/{segment(group_id)}")
        response.raise_for_status()
        return response.json()

    def wait(self, group_id: str, *, timeout_s: float = 3600, interval_s: float = 1) -> dict:
        if timeout_s <= 0 or interval_s <= 0:
            raise ValueError("Wait timeout and interval must be positive")
        deadline = time.monotonic() + timeout_s
        while True:
            result = self.get(group_id)
            if result["ready"] or result["status"] == "quarantined":
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("Stopped waiting; group execution is unchanged")
            time.sleep(min(interval_s, deadline - time.monotonic()))

    def events(self, group_id: str, sample_id: str, *, after: int = 0) -> list[dict]:
        response = self.http.get(f"/execution-groups/{segment(group_id)}/samples/{segment(sample_id)}/events",
                                 params={"after": after})
        response.raise_for_status()
        return response.json()

    def recovery_points(self, group_id: str, sample_id: str) -> list[dict]:
        response = self.http.get(f"/execution-groups/{segment(group_id)}/samples/{segment(sample_id)}/recovery-points")
        response.raise_for_status()
        return response.json()

    def release(self, group_id: str) -> dict:
        response = self.http.delete(f"/execution-groups/{segment(group_id)}")
        response.raise_for_status()
        return response.json()

    def reconcile_submission(self, group_id: str, sample_id: str, *, phase: str, job_id: str) -> dict:
        response = self.http.post(
            f"/execution-groups/{segment(group_id)}/samples/{segment(sample_id)}/reconcile-submission",
            json={"phase": phase, "job_id": job_id})
        response.raise_for_status()
        return response.json()
