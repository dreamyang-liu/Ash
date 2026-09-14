"""Shared Miles SessionTree execution support for rollout strategies.

Branch policies decide how a rollout tree grows.  This module owns the stable
mechanics they should not need to reimplement: Miles session HTTP calls, agent
execution with tool budgets, and conversion of SessionTree records into the
token-aligned trajectory contract.
"""

from __future__ import annotations

import asyncio
import copy
import json
import inspect
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..agent import AshAgent
from ..models import AgentConfig, ToolResult
from .protocol import GeneratedSpan, RolloutGroupRequest, Trajectory
from .runner import RolloutCancelled, RolloutContext


class MilesSessionClient:
    """Synchronous client for the Miles v2 session lifecycle."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 120.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def create(self) -> str:
        payload = self._request("POST", "/sessions", {})
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Miles session server returned no session_id")
        return session_id

    def require_capabilities(self, *required: str) -> dict[str, Any]:
        """Fail before rollout when an endpoint lacks required semantics."""
        payload = self._request("GET", "/health", None)
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, list) or any(
            not isinstance(item, str) for item in capabilities
        ):
            raise RuntimeError(
                "Miles session server does not report capabilities; deploy a "
                "compatible version to every configured endpoint"
            )
        missing = sorted(set(required) - set(capabilities))
        if missing:
            raise RuntimeError(
                "Miles session server is missing required capabilities: "
                + ", ".join(missing)
            )
        return payload

    def get(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{session_id}", None)

    def collect_samples(self, session_id: str) -> bytes:
        """Ask Miles to materialize and validate all SessionTree leaves."""
        request = urllib.request.Request(
            self.endpoint + f"/sessions/{session_id}/samples",
            data=b"{}",
            headers={"Accept": "application/octet-stream", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read()

    def delete(self, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}", None)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.endpoint + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read()
            status = getattr(response, "status", None)
        # Miles returns 204 No Content for a successful session deletion.  The
        # lifecycle client still uses one request helper for JSON endpoints, so
        # an empty successful body must be treated as an empty object rather
        # than parsed as JSON.
        if not raw.strip():
            if method == "DELETE" or status in {204, 205, 304}:
                return {}
            raise ValueError("Miles session server returned an empty response")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Miles session server returned a non-object response")
        return value


class SessionAgentStrategySupport:
    """Reusable agent/session execution mechanics for tree-growth policies."""

    def __init__(
        self,
        *,
        agent_config: AgentConfig | None = None,
        session_server_endpoint: str | None = None,
        allow_text_prompt: bool = True,
    ) -> None:
        self.agent_config = agent_config or AgentConfig(model="openai/local")
        self.session_server_endpoint = session_server_endpoint.rstrip("/") if session_server_endpoint else None
        self.allow_text_prompt = allow_text_prompt

    def session_endpoint(self, request: RolloutGroupRequest) -> str:
        endpoint = request.session_server_endpoint or self.session_server_endpoint or request.model_endpoint
        if "/sessions" in endpoint:
            raise ValueError("session_server_endpoint must be the Miles server base URL, not a session URL")
        return endpoint

    def initial_messages(self, request: RolloutGroupRequest) -> list[dict[str, Any]]:
        if isinstance(request.prompt, list):
            return [dict(message) for message in request.prompt]
        if not self.allow_text_prompt:
            raise ValueError("text prompts are disabled for the agent-loop strategy")
        return [{"role": "user", "content": request.prompt}]

    def _run_agent(
        self,
        *,
        request: RolloutGroupRequest,
        context: RolloutContext,
        client: MilesSessionClient,
        session_id: str,
        slot,
        sandbox,
        initial_messages: list[dict[str, Any]],
        agent_id: str,
        max_model_calls: int | None,
        max_tool_calls: int | None,
        model_calls_offset: int = 0,
        tool_calls_offset: int = 0,
        on_turn_end=None,
    ) -> tuple[str, int, int, list[dict[str, Any]], float]:
        config = self._agent_config(
            request,
            client,
            session_id,
            max_model_calls,
            remaining_wall_time_seconds=context.remaining_wall_time_seconds,
        )
        executor, executed_tool_calls, tool_seconds = self._tool_executor(
            sandbox,
            agent_id=agent_id,
            context=context,
            max_tool_calls=max_tool_calls,
            tool_calls_offset=tool_calls_offset,
        )
        agent = AshAgent(
            config,
            executor=executor,
            agent_id=agent_id,
            sandbox_id=str(getattr(sandbox, "sandbox_id", "unknown")),
        )
        agent.on_step = lambda step, kind, _summary: context.update_progress(
            "tool_execution" if kind != "error" else "agent_error",
            model_calls=model_calls_offset + step,
            tool_calls=tool_calls_offset + executed_tool_calls(),
            active_sample_slot_id=slot.sample_slot_id,
        )
        agent.stream = False
        latest_messages: list[dict[str, Any]] = []

        def report_model_generation(_agent, _conversation) -> None:
            context.update_progress(
                "model_generation",
                # The hook runs immediately before issuing the request, while
                # CostTracker increments only after a response is received.
                # Count this admitted in-flight call so cancellation does not
                # make a long first generation look like zero model work.
                model_calls=model_calls_offset + agent.cost.api_calls + 1,
                tool_calls=tool_calls_offset + executed_tool_calls(),
                active_sample_slot_id=slot.sample_slot_id,
            )

        before_query_hooks = getattr(agent, "before_query_hooks", None)
        if isinstance(before_query_hooks, list):
            before_query_hooks.insert(0, report_model_generation)

        # AshAgent's model call is synchronous, so checking cancel_event only
        # between tool calls is insufficient.  A small watcher translates job
        # cancellation into deletion of the active Miles session.  Miles then
        # cancels its upstream HTTP request, which SGLang treats as a client
        # disconnect and aborts at the scheduler.
        cancel_watch_done = threading.Event()

        def propagate_cancellation() -> None:
            wait = getattr(context.cancel_event, "wait", None)
            if not callable(wait):
                return
            while not cancel_watch_done.is_set():
                if wait(timeout=0.1):
                    try:
                        client.delete(session_id)
                    except Exception:
                        # Cleanup is best-effort here.  The strategy's normal
                        # finally block remains the authoritative cleanup path.
                        pass
                    return

        cancel_watcher = threading.Thread(
            target=propagate_cancellation,
            name=f"ash-cancel-{session_id}",
            daemon=True,
        )
        cancel_watcher.start()

        def capture_messages(step: int, messages: list[dict[str, Any]]) -> None:
            latest_messages[:] = copy.deepcopy(messages)
            context.update_progress(
                "model_generation",
                model_calls=model_calls_offset + step,
                tool_calls=tool_calls_offset + executed_tool_calls(),
                active_sample_slot_id=slot.sample_slot_id,
            )
            if on_turn_end is not None:
                on_turn_end(step, messages)

        agent.on_turn_end = capture_messages
        try:
            context.update_progress(
                "model_generation",
                model_calls=model_calls_offset,
                tool_calls=tool_calls_offset,
                active_sample_slot_id=slot.sample_slot_id,
            )
            status = agent.run(
                task="",
                instance_id=request.task_id,
                initial_messages=initial_messages,
            )
        finally:
            cancel_watch_done.set()
            cancel_watcher.join(timeout=0.2)
        context.check_cancelled()
        if status == "error":
            detail = agent.last_model_error or "unknown model request failure"
            raise RuntimeError(f"AshAgent model request failed: {detail}")
        if not latest_messages:
            errors = [
                str(message.get("content") or "")
                for message in agent.trajectory.messages
                if message.get("role") == "error"
            ]
            detail = errors[-1] if errors else "no recorded model error"
            raise RuntimeError(
                "AshAgent stopped before its first model-visible response "
                f"(status={status}): {detail}"
            )
        if _tool_call_count(latest_messages) < _tool_call_count(initial_messages):
            raise RuntimeError("agent history lost tool results from its checkpoint prefix")
        return (
            status,
            agent.cost.api_calls,
            executed_tool_calls(),
            latest_messages,
            tool_seconds(),
        )

    def _agent_config(
        self,
        request: RolloutGroupRequest,
        client: MilesSessionClient,
        session_id: str,
        max_model_calls: int | None,
        remaining_wall_time_seconds: float | None = None,
    ) -> AgentConfig:
        sampling = dict(request.sampling_params)
        model = request.model or sampling.pop("model", None) or self.agent_config.model
        if "/" not in str(model):
            model = f"openai/{model}"
        max_tokens = sampling.get("max_tokens", sampling.get("max_new_tokens", self.agent_config.max_tokens))
        temperature = sampling.get("temperature", self.agent_config.temperature)
        raw_extra_body = sampling.get("extra_body")
        if raw_extra_body is not None and not isinstance(raw_extra_body, dict):
            raise ValueError("sampling_params.extra_body must be an object")
        extra_body = dict(raw_extra_body or {})
        for name in (
            "seed",
            "top_p",
            "top_k",
            "stop",
            "stop_token_ids",
            "skip_special_tokens",
            "no_stop_trim",
            "spaces_between_special_tokens",
            "chat_template_kwargs",
        ):
            if name in sampling and name not in extra_body:
                extra_body[name] = sampling[name]
        request_timeout = self.agent_config.request_timeout
        if remaining_wall_time_seconds is not None:
            request_timeout = min(
                request_timeout or remaining_wall_time_seconds,
                remaining_wall_time_seconds,
            )
        return replace(
            self.agent_config,
            model=str(model),
            api_base=f"{client.endpoint}/sessions/{session_id}/v1",
            api_key=self.agent_config.api_key or "EMPTY",
            max_tokens=int(max_tokens),
            temperature=None if temperature is None else float(temperature),
            prompt_cache=False,
            extra_body=extra_body or None,
            request_timeout=request_timeout,
            # AshAgent's standalone default is finite.  A null rollout budget
            # delegates termination to wall time, cancellation and the model
            # context limit instead of accidentally retaining that default.
            step_limit=(
                sys.maxsize
                if max_model_calls is None
                else min(self.agent_config.step_limit, max_model_calls)
            ),
            cost_limit=(
                sys.float_info.max
                if max_model_calls is None
                else self.agent_config.cost_limit
            ),
        )

    @staticmethod
    def _tool_executor(
        sandbox,
        *,
        agent_id: str,
        context: RolloutContext,
        max_tool_calls: int | None,
        tool_calls_offset: int = 0,
    ) -> tuple[
        Callable[[str, dict[str, Any]], ToolResult],
        Callable[[], int],
        Callable[[], float],
    ]:
        call = getattr(sandbox, "call_agent_tool", None) or getattr(sandbox, "call", None)
        if call is None:
            session_executor = getattr(sandbox, "executor_for", None)
            if session_executor is not None:
                call = session_executor(agent_id)
        if call is None:
            raise TypeError("environment sandbox must expose call_agent_tool or call")
        calls = 0
        elapsed_seconds = 0.0

        def execute(name: str, args: dict[str, Any]) -> ToolResult:
            nonlocal calls, elapsed_seconds
            context.check_cancelled()
            if max_tool_calls is not None and calls >= max_tool_calls:
                raise RolloutCancelled("rollout tool-call budget exhausted")
            calls += 1
            context.update_progress(
                "tool_execution",
                tool_calls=tool_calls_offset + calls,
            )
            call_args = dict(args)
            remaining = context.remaining_wall_time_seconds
            if remaining is not None:
                if remaining <= 0:
                    context.check_cancelled()
                if name == "shell":
                    requested = call_args.get("timeout")
                    # Give the runtime a short interval to serialize the
                    # command's timeout outcome before the outer transport is
                    # cancelled at the rollout deadline.
                    remaining_timeout = max(1, int(max(1.0, remaining - 5.0)))
                    call_args["timeout"] = (
                        remaining_timeout
                        if requested is None
                        else min(int(requested), remaining_timeout)
                    )
            started = time.monotonic()
            try:
                try:
                    parameters = inspect.signature(call).parameters.values()
                except (TypeError, ValueError):
                    accepts_transport_timeout = False
                else:
                    accepts_transport_timeout = any(
                        parameter.name == "timeout"
                        or parameter.kind
                        in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
                        for parameter in parameters
                    )
                result = _run_async(
                    call(name, call_args, remaining)
                    if remaining is not None and accepts_transport_timeout
                    else call(name, call_args)
                )
            finally:
                elapsed_seconds += time.monotonic() - started
            if isinstance(result, ToolResult):
                return result
            return ToolResult.from_sdk(result)

        return execute, lambda: calls, lambda: elapsed_seconds


def session_model_seconds(state: dict[str, Any]) -> float:
    """Sum request-to-commit latency for model calls on one session path."""
    elapsed = 0.0
    for record in state.get("records") or []:
        timestamp = record.get("timestamp")
        request_timestamp = record.get("request_timestamp")
        if isinstance(timestamp, (int, float)) and isinstance(
            request_timestamp, (int, float)
        ):
            elapsed += max(0.0, float(timestamp) - float(request_timestamp))
    return elapsed


def branch_input_length(state: dict[str, Any], checkpoint_messages: list[dict[str, Any]]) -> int:
    """Return the first child request's actual tokenized input length."""
    for record in state.get("records") or []:
        request = record.get("request") or {}
        if request.get("messages") == checkpoint_messages:
            input_ids = request.get("input_ids") or []
            if input_ids:
                return len(input_ids)
    raise ValueError("Miles session did not expose the child request at the checkpoint message boundary")


def response_input_length(state: dict[str, Any], response_id: str) -> int:
    """Return the model-visible input length for a recorded generation."""
    tree = (state.get("metadata") or {}).get("tree") or {}
    for node in tree.get("nodes") or []:
        if node.get("response_id") == response_id:
            span = node.get("completion_span") or []
            if len(span) == 2:
                return int(span[0])
    raise ValueError(
        f"Miles SessionTree has no node for response_id {response_id!r}"
    )


def trajectory_from_session(
    request: RolloutGroupRequest,
    sample_slot_id: str,
    state: dict[str, Any],
    status: str,
    *,
    branch_id: str,
    messages: list[dict[str, Any]] | None = None,
    prompt_token_alignment: str = "request_exact",
    parent_branch_id: str | None = None,
    branch_point_token_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    records = state.get("records")
    session_metadata = state.get("metadata") or {}
    token_ids = session_metadata.get("accumulated_token_ids")
    if not isinstance(records, list) or not records or not isinstance(token_ids, list) or not token_ids:
        raise ValueError("Miles session did not return records and accumulated_token_ids")

    spans: list[GeneratedSpan] = []
    final_messages: list[dict[str, Any]] | None = None
    for record in records:
        req = record.get("request") or {}
        response_envelope = record.get("response") or {}
        choices = response_envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Miles session record contains no response choices")
        response = choices[0]
        info = response.get("meta_info") or {}
        pairs = info.get("output_token_logprobs") or []
        output_ids = [int(pair[1]) for pair in pairs]
        input_ids = [int(token) for token in req.get("input_ids", [])]
        if not output_ids or not input_ids:
            raise ValueError("Miles session record lacks token-level generation data")
        message = response.get("message") or {}
        final_messages = [dict(item) for item in req.get("messages", [])] + [dict(message)]
        logs = [float(pair[0]) for pair in pairs] if request.return_rollout_logprobs else None
        spans.append(
            GeneratedSpan(
                response_id=str(response_envelope.get("id") or f"{request.rollout_job_id}:{len(spans)}"),
                start=len(input_ids),
                end=len(input_ids) + len(output_ids),
                input_token_ids=tuple(input_ids),
                output_token_ids=tuple(output_ids),
                weight_version=str(info.get("weight_version") or request.expected_weight_version or "unknown"),
                finish_reason=str(response.get("finish_reason") or "stop"),
                output_token_log_probs=None if logs is None else tuple(logs),
            )
        )

    visible_messages = copy.deepcopy(messages if messages is not None else final_messages)
    if visible_messages is None:
        raise ValueError("Miles session returned no final messages")
    response_text = ""
    for message in reversed(visible_messages):
        if message.get("role") == "assistant":
            response_text = str(message.get("content") or "")
            break
    prompt_length = len(records[0].get("request", {}).get("input_ids", []))
    return Trajectory(
        sample_slot_id=sample_slot_id,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        branch_point_token_count=branch_point_token_count,
        messages=visible_messages,
        token_ids=[int(token) for token in token_ids],
        prompt_length=prompt_length,
        generated_spans=spans,
        response_text=response_text,
        status=trajectory_status(status),
        prompt_token_alignment=prompt_token_alignment,
        metadata=dict(metadata or {}),
    )


def trajectory_status(agent_status: str) -> str:
    if agent_status == "completed":
        return "completed"
    if agent_status in {"step_limit", "cost_limit", "length_limit"}:
        return "truncated"
    return "failed"


def _tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in messages
        if message.get("role") == "assistant"
    )


def _run_async(awaitable):
    if not asyncio.iscoroutine(awaitable) and not isinstance(awaitable, asyncio.Future):
        return awaitable
    return asyncio.run(awaitable)
