"""Strategy-neutral rollout-group orchestration.

The runner owns lifecycle, cancellation and resource cleanup. Branch choice,
checkpoint cadence and model prompting are injected through protocols.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from .protocol import (
    PROTOCOL_VERSION,
    GeneratedSpan,
    RolloutGroupRequest,
    RolloutGroupResult,
    RolloutProgress,
    RolloutDeletion,
    RolloutSubmission,
    Trajectory,
)


logger = logging.getLogger(__name__)


class ModelClient(Protocol):
    """Generate from the endpoint supplied by Miles.

    A strategy decides how many calls to make and how to associate responses
    with branches.  The client only standardizes the transport-level result.
    """

    def generate(self, *, endpoint: str, prompt_token_ids: list[int],
                 sampling_params: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EnvironmentCheckpoint:
    """Opaque environment state returned to an Ash rollout strategy."""

    checkpoint_id: str
    owner_job_id: str
    source_sandbox_id: str
    backend: str
    state_scope: Literal["full-runtime", "filesystem-only"]
    multiple_restore: bool
    explicit_release: bool

    def __post_init__(self) -> None:
        for name in ("checkpoint_id", "owner_job_id", "source_sandbox_id", "backend"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.state_scope not in {"full-runtime", "filesystem-only"}:
            raise ValueError(
                "state_scope must be 'full-runtime' or 'filesystem-only'"
            )
        for name in ("multiple_restore", "explicit_release"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")


class EnvironmentProvider(Protocol):
    """Environment lifecycle used by a branch strategy."""

    def spawn(self, request: RolloutGroupRequest) -> Any: ...
    def create_checkpoint(
        self, sandbox: Any, *, owner_job_id: str, name: str | None = None
    ) -> EnvironmentCheckpoint: ...
    def restore_checkpoint(
        self, checkpoint: EnvironmentCheckpoint, *, agent_id: str = ""
    ) -> Any: ...
    def release_checkpoint(self, checkpoint: EnvironmentCheckpoint) -> bool: ...
    def destroy(self, sandbox: Any) -> None: ...


@dataclass(frozen=True)
class TaskEvaluation:
    """Benchmark result produced while the final sandbox is still alive."""

    reward: float | dict[str, Any]
    metadata: dict[str, Any]


class TaskAdapter(Protocol):
    """Server-side task setup and grading kept outside the public request."""

    def validate_request(self, request: RolloutGroupRequest) -> None: ...
    def prepare(
        self,
        request: RolloutGroupRequest,
        sandbox: Any,
        context: "RolloutContext | None" = None,
    ) -> Any: ...
    def evaluate(
        self,
        request: RolloutGroupRequest,
        sandbox: Any,
        trajectory: Trajectory,
        prepared: Any,
        context: "RolloutContext | None" = None,
    ) -> TaskEvaluation: ...


class RolloutStrategy(Protocol):
    """Algorithm plug-in; no HTTP or job-state responsibilities."""

    def run(self, request: RolloutGroupRequest, context: "RolloutContext") -> RolloutGroupResult: ...


@dataclass
class RolloutContext:
    cancel_event: threading.Event
    model_client: ModelClient | None
    environment_provider: EnvironmentProvider | None
    job_id: str
    deadline: float | None = None
    task_adapter: TaskAdapter | None = None
    progress_callback: Any | None = None

    def update_progress(
        self,
        phase: str,
        *,
        model_calls: int | None = None,
        tool_calls: int | None = None,
        completed_samples: int | None = None,
        active_sample_slot_id: str | None = None,
    ) -> None:
        """Publish bounded operational state without exposing trajectory text."""
        if self.progress_callback is not None:
            self.progress_callback(
                phase=phase,
                model_calls=model_calls,
                tool_calls=tool_calls,
                completed_samples=completed_samples,
                active_sample_slot_id=active_sample_slot_id,
            )

    def check_cancelled(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.cancel_event.set()
            raise RolloutCancelled("rollout wall-time budget exhausted")
        if self.cancel_event.is_set():
            raise RolloutCancelled("rollout job was cancelled")

    @property
    def remaining_wall_time_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def prepare_task(self, request: RolloutGroupRequest, sandbox: Any) -> Any:
        if self.task_adapter is None:
            return None
        self.check_cancelled()
        self.update_progress("preparing_task")
        return self.task_adapter.prepare(request, sandbox, self)

    def evaluate_trajectory(
        self,
        request: RolloutGroupRequest,
        sandbox: Any,
        trajectory: Trajectory,
        prepared: Any,
    ) -> Trajectory:
        if self.task_adapter is None:
            return trajectory
        self.check_cancelled()
        self.update_progress("evaluating")
        evaluation = self.task_adapter.evaluate(
            request, sandbox, trajectory, prepared, self
        )
        self.check_cancelled()
        return replace(
            trajectory,
            reward=evaluation.reward,
            metadata={**trajectory.metadata, **evaluation.metadata},
        )


class RolloutCancelled(Exception):
    pass


class EnvironmentInUse(Exception):
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

    def __init__(
        self,
        strategy_factory,
        *,
        model_client: ModelClient | None = None,
        environment_provider: EnvironmentProvider | None = None,
        task_adapter: TaskAdapter | None = None,
        profile_writer: Any | None = None,
        result_ttl_seconds: float = 300.0,
    ):
        if result_ttl_seconds <= 0:
            raise ValueError("result_ttl_seconds must be greater than zero")
        self.strategy_factory = strategy_factory
        self.model_client = model_client
        self.environment_provider = environment_provider
        self.task_adapter = task_adapter
        self.profile_writer = profile_writer
        self.result_ttl_seconds = result_ttl_seconds
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        # Workers remain tracked after DELETE removes their public job record.
        # A storage supervisor can therefore wait for cancellation cleanup to
        # finish before replacing an AgentENV node.
        self._workers: set[threading.Thread] = set()

    def submit(self, request: RolloutGroupRequest) -> RolloutSubmission:
        # Environment selection is optional for providers such as the
        # sequential strategy. Providers that resolve logical environment
        # references can expose this hook to reject a request before enqueue.
        validate_environment = getattr(
            self.environment_provider, "validate_request", None
        )
        if callable(validate_environment):
            validate_environment(request)
        if self.task_adapter is not None:
            self.task_adapter.validate_request(request)
        canonical = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._prune_terminal_jobs_locked()
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
                "terminal_at": None,
                "started_at": None,
                "progress": {
                    "phase": "queued",
                    "model_calls": 0,
                    "tool_calls": 0,
                    "completed_samples": 0,
                    "active_sample_slot_id": None,
                    "updated_at_unix_seconds": time.time(),
                },
            }
            self._jobs[request.rollout_job_id] = record
            worker = threading.Thread(target=self._run, args=(request.rollout_job_id,),
                                      name=f"ash-rollout-{request.rollout_job_id}", daemon=True)
            record["thread"] = worker
            self._workers.add(worker)
            worker.start()
            return RolloutSubmission(request.rollout_job_id, "queued")

    def activity(self) -> dict[str, Any]:
        """Return aggregate lifecycle state without exposing task contents."""
        with self._lock:
            self._workers = {worker for worker in self._workers if worker.is_alive()}
            counts: dict[str, int] = {}
            for record in self._jobs.values():
                status = record["result"].status
                counts[status] = counts.get(status, 0) + 1
            return {
                "protocol_version": PROTOCOL_VERSION,
                "status": "ready",
                "active_workers": len(self._workers),
                "retained_jobs": len(self._jobs),
                "jobs_by_status": counts,
            }

    def list_environments(self) -> dict[str, Any]:
        """Describe the static environment refs accepted by this deployment.

        Dynamic OCI images are policy-validated when submitted and are not
        persisted in the static catalog merely because their prepared snapshot
        is cached by the environment backend.
        """
        list_environments = getattr(
            self.environment_provider, "list_environments", None
        )
        environments = list_environments() if callable(list_environments) else []
        return {
            "protocol_version": PROTOCOL_VERSION,
            "environments": environments,
        }

    def release_environment(self, environment_ref) -> dict[str, Any]:
        """Release one idle, dynamically prepared rollout environment.

        The lock makes the active-reference check and backend release atomic
        with respect to job submission. Terminal jobs remain references until
        the consumer deletes them, proving that their results were collected.
        """
        release = getattr(self.environment_provider, "release_environment", None)
        if not callable(release):
            raise ValueError("environment provider does not support cache release")
        with self._lock:
            blockers = [
                job_id
                for job_id, record in self._jobs.items()
                if record["request"].environment_ref == environment_ref
            ]
            if blockers:
                raise EnvironmentInUse(
                    "environment is still referenced by rollout jobs: "
                    + ", ".join(sorted(blockers))
                )
            released = release(environment_ref)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "environment_ref": environment_ref.to_dict(),
            "status": "released" if released else "not_cached",
        }

    def get(self, job_id: str) -> RolloutGroupResult:
        with self._lock:
            self._prune_terminal_jobs_locked()
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            result = record["result"]
            if result.status not in {"queued", "running"}:
                return result
            progress = dict(record["progress"])
            started_at = record["started_at"]
            elapsed = 0.0 if started_at is None else max(0.0, time.monotonic() - started_at)
            request: RolloutGroupRequest = record["request"]
            remaining = max(0.0, request.budgets.max_wall_time_seconds - elapsed)
            return replace(
                result,
                progress=RolloutProgress(
                    **progress,
                    elapsed_seconds=round(elapsed, 3),
                    remaining_wall_time_seconds=round(remaining, 3),
                ),
            )

    def delete(self, job_id: str) -> RolloutDeletion:
        """Cancel unfinished work and release the in-memory job record.

        The returned snapshot lets the HTTP layer acknowledge what was
        deleted.  Removing terminal results is important because they contain
        complete trajectories and would otherwise accumulate for the lifetime
        of the rollout service.
        """
        with self._lock:
            self._prune_terminal_jobs_locked()
            record = self._jobs.pop(job_id, None)
            if record is None:
                raise KeyError(job_id)
            result: RolloutGroupResult = record["result"]
            if result.status in {"completed", "early_stopped", "failed", "cancelled"}:
                return RolloutDeletion(job_id, result.status)
            record["cancel_event"].set()
            return RolloutDeletion(job_id, "cancelled")

    def cancel(self, job_id: str) -> RolloutDeletion:
        """Compatibility alias for callers using the original method name."""
        return self.delete(job_id)

    def _set_result(self, job_id: str, result: RolloutGroupResult) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            current = self._jobs[job_id]["result"]
            if current.status == "cancelled":
                return
            self._jobs[job_id]["result"] = result
            if result.status in {"completed", "early_stopped", "failed", "cancelled"}:
                self._jobs[job_id]["terminal_at"] = time.monotonic()

    def _update_progress(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record["result"].status not in {"queued", "running"}:
                return
            progress = record["progress"]
            for name, value in changes.items():
                if value is not None:
                    if name in {"model_calls", "tool_calls", "completed_samples"}:
                        progress[name] = max(int(progress.get(name, 0)), int(value))
                    else:
                        progress[name] = value
            progress["updated_at_unix_seconds"] = time.time()

    def _prune_terminal_jobs_locked(self) -> None:
        cutoff = time.monotonic() - self.result_ttl_seconds
        expired = [
            job_id
            for job_id, record in self._jobs.items()
            if record["terminal_at"] is not None and record["terminal_at"] <= cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _consumed_budget_snapshot(
        self, job_id: str, *, started: float
    ) -> dict[str, int | float]:
        """Keep measured work when a rollout terminates outside a strategy."""
        with self._lock:
            record = self._jobs.get(job_id)
            progress = {} if record is None else record["progress"]
            return {
                "model_calls": int(progress.get("model_calls", 0)),
                "tool_calls": int(progress.get("tool_calls", 0)),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

    def _run(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            request: RolloutGroupRequest = record["request"]
            cancel_event: threading.Event = record["cancel_event"]
            if record["result"].status == "cancelled":
                return
            started = time.monotonic()
            record["started_at"] = started
            record["progress"].update(
                phase="starting",
                updated_at_unix_seconds=time.time(),
            )
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
            job_id,
            deadline=time.monotonic() + request.budgets.max_wall_time_seconds,
            task_adapter=self.task_adapter,
            progress_callback=lambda **changes: self._update_progress(job_id, **changes),
        )
        timeout_timer = threading.Timer(
            request.budgets.max_wall_time_seconds,
            cancel_event.set,
        )
        timeout_timer.daemon = True
        timeout_timer.start()
        try:
            strategy = self.strategy_factory(request, context)
            context.update_progress("running_strategy")
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
            elapsed_seconds = round(time.monotonic() - started, 3)
            if self.profile_writer is not None:
                self.profile_writer.write(
                    request, result, elapsed_seconds=elapsed_seconds
                )
            self._set_result(job_id, result)
        except RolloutCancelled as exc:
            self._set_result(job_id, RolloutGroupResult(
                rollout_job_id=job_id, prompt_group_id=request.prompt_group_id,
                status="cancelled", max_samples=request.max_samples,
                stop_reason=str(exc),
                consumed_budget=self._consumed_budget_snapshot(
                    job_id, started=started
                ),
            ))
        except Exception as exc:  # noqa: BLE001 - failure is part of the wire contract
            logger.exception("rollout job %s failed", job_id)
            self._set_result(job_id, RolloutGroupResult(
                rollout_job_id=job_id, prompt_group_id=request.prompt_group_id,
                status="failed", max_samples=request.max_samples,
                stop_reason=f"{type(exc).__name__}: {exc}",
                consumed_budget=self._consumed_budget_snapshot(
                    job_id, started=started
                ),
            ))
        finally:
            timeout_timer.cancel()
            close = getattr(strategy if "strategy" in locals() else None, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            with self._lock:
                self._workers.discard(threading.current_thread())
