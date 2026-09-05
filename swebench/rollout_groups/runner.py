"""Strategy-neutral rollout-group orchestration.

The runner owns lifecycle, cancellation and resource cleanup. Branch choice,
checkpoint cadence and model prompting are injected through protocols.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .protocol import (
    GeneratedSpan,
    RolloutGroupRequest,
    RolloutGroupResult,
    RolloutSubmission,
    Trajectory,
)


class ModelClient(Protocol):
    """Generate from the endpoint supplied by Miles.

    A strategy decides how many calls to make and how to associate responses
    with branches.  The client only standardizes the transport-level result.
    """

    def generate(self, *, endpoint: str, prompt_token_ids: list[int],
                 sampling_params: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]: ...


class EnvironmentProvider(Protocol):
    """Environment lifecycle used by a branch strategy."""

    def spawn(self, request: RolloutGroupRequest) -> Any: ...
    def snapshot(self, sandbox: Any, *, name: str) -> str: ...
    def restore(self, snapshot_id: str) -> Any: ...
    def fork(self, sandbox: Any, *, count: int) -> list[Any]: ...
    def destroy(self, sandbox: Any) -> None: ...


class RolloutStrategy(Protocol):
    """Algorithm plug-in; no HTTP or job-state responsibilities."""

    def run(self, request: RolloutGroupRequest, context: "RolloutContext") -> RolloutGroupResult: ...


@dataclass
class RolloutContext:
    cancel_event: threading.Event
    model_client: ModelClient | None
    environment_provider: EnvironmentProvider | None
    checkpoint_store: Any | None
    job_id: str
    deadline: float | None = None

    def check_cancelled(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.cancel_event.set()
        if self.cancel_event.is_set():
            raise RolloutCancelled("rollout job was cancelled")

    @property
    def remaining_wall_time_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())


class RolloutCancelled(Exception):
    pass


class EndpointModelClient:
    """Small SGLang-compatible client for strategy implementations.

    It intentionally does not know about OpenAI or LiteLLM.  Ash can call a
    Miles Session Server or raw SGLang endpoint by supplying the corresponding
    JSON body in ``request``; the response is returned unchanged for the
    strategy's trajectory exporter to normalize.
    """

    def __init__(self, *, timeout_seconds: float = 120.0):
        self.timeout_seconds = timeout_seconds

    def generate(self, *, endpoint: str, prompt_token_ids: list[int],
                 sampling_params: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        # SGLang's native endpoint expects generation controls under one
        # ``sampling_params`` object.  A Session Server adapter can instead
        # replace this client while keeping the strategy and HTTP contract.
        body = dict(request)
        body["input_ids"] = prompt_token_ids
        body["sampling_params"] = sampling_params
        data = json.dumps(body).encode("utf-8")
        url = endpoint.rstrip("/") + "/generate"
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("model endpoint returned a non-object response")
        return payload


class GroupRolloutService:
    """In-process asynchronous service backing the three HTTP endpoints."""

    def __init__(self, strategy_factory, *, model_client: ModelClient | None = None,
                 environment_provider: EnvironmentProvider | None = None,
                 checkpoint_store: Any | None = None):
        self.strategy_factory = strategy_factory
        self.model_client = model_client
        self.environment_provider = environment_provider
        self.checkpoint_store = checkpoint_store
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def submit(self, request: RolloutGroupRequest) -> RolloutSubmission:
        canonical = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock:
            existing = self._jobs.get(request.rollout_job_id)
            if existing is not None:
                if existing["request_json"] != canonical:
                    raise ValueError("rollout_job_id already exists with a different request")
                return RolloutSubmission(request.rollout_job_id, existing["result"].status)
            result = RolloutGroupResult(
                rollout_job_id=request.rollout_job_id,
                prompt_group_id=request.prompt_group_id,
                status="queued",
                max_samples=request.max_samples,
            )
            record = {
                "request": request,
                "request_json": canonical,
                "result": result,
                "cancel_event": threading.Event(),
                "thread": None,
            }
            self._jobs[request.rollout_job_id] = record
            worker = threading.Thread(target=self._run, args=(request.rollout_job_id,),
                                      name=f"ash-rollout-{request.rollout_job_id}", daemon=True)
            record["thread"] = worker
            worker.start()
            return RolloutSubmission(request.rollout_job_id, "queued")

    def get(self, job_id: str) -> RolloutGroupResult:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record["result"]

    def cancel(self, job_id: str) -> RolloutGroupResult:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            result: RolloutGroupResult = record["result"]
            if result.status in {"completed", "early_stopped", "failed", "cancelled"}:
                return result
            record["cancel_event"].set()
            record["result"] = RolloutGroupResult(
                rollout_job_id=result.rollout_job_id,
                prompt_group_id=result.prompt_group_id,
                status="cancelled",
                max_samples=result.max_samples,
                stop_reason="cancelled by Miles",
            )
            return record["result"]

    def _set_result(self, job_id: str, result: RolloutGroupResult) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            current = self._jobs[job_id]["result"]
            if current.status == "cancelled":
                return
            self._jobs[job_id]["result"] = result

    def _run(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            request: RolloutGroupRequest = record["request"]
            cancel_event: threading.Event = record["cancel_event"]
            if record["result"].status == "cancelled":
                return
            self._jobs[job_id]["result"] = RolloutGroupResult(
                rollout_job_id=job_id,
                prompt_group_id=request.prompt_group_id,
                status="running",
                max_samples=request.max_samples,
            )
        context = RolloutContext(
            cancel_event,
            self.model_client,
            self.environment_provider,
            self.checkpoint_store,
            job_id,
            deadline=time.monotonic() + request.budgets.max_wall_time_seconds,
        )
        timeout_timer = threading.Timer(
            request.budgets.max_wall_time_seconds,
            cancel_event.set,
        )
        timeout_timer.daemon = True
        timeout_timer.start()
        started = time.monotonic()
        try:
            strategy = self.strategy_factory(request, context)
            result = strategy.run(request, context)
            if not isinstance(result, RolloutGroupResult):
                raise TypeError("rollout strategy must return RolloutGroupResult")
            if result.status in {"queued", "running"}:
                raise ValueError("rollout strategy must return a terminal result")
            if result.rollout_job_id != request.rollout_job_id or result.prompt_group_id != request.prompt_group_id:
                raise ValueError("strategy result does not match request identity")
            if result.max_samples != request.max_samples:
                raise ValueError("strategy result max_samples does not match request")
            if result.actual_samples < request.minimum_returned_samples:
                raise ValueError(
                    f"strategy returned {result.actual_samples} samples, below the requested "
                    f"minimum {request.minimum_returned_samples}"
                )
            requested_slots = {slot.sample_slot_id for slot in request.sample_slots}
            returned_slots = {trajectory.sample_slot_id for trajectory in result.trajectories}
            if not returned_slots <= requested_slots:
                raise ValueError("strategy returned a sample_slot_id not allocated by Miles")
            context.check_cancelled()
            self._set_result(job_id, result)
        except RolloutCancelled as exc:
            self._set_result(job_id, RolloutGroupResult(
                rollout_job_id=job_id, prompt_group_id=request.prompt_group_id,
                status="cancelled", max_samples=request.max_samples,
                stop_reason=str(exc),
            ))
        except Exception as exc:  # noqa: BLE001 - failure is part of the wire contract
            self._set_result(job_id, RolloutGroupResult(
                rollout_job_id=job_id, prompt_group_id=request.prompt_group_id,
                status="failed", max_samples=request.max_samples,
                stop_reason=f"{type(exc).__name__}: {exc}",
                consumed_budget={"elapsed_seconds": round(time.monotonic() - started, 3)},
            ))
        finally:
            timeout_timer.cancel()
            close = getattr(strategy if "strategy" in locals() else None, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
