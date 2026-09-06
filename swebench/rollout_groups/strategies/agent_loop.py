"""Ash agent-loop strategy backed by a Miles v2 session server.

This adapter is intentionally small: Ash owns the agent/tool loop and the
environment, while Miles owns tokenization, session-tree records and the
training sample.  A future branch policy can replace this strategy without
changing the rollout-groups protocol.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from dataclasses import replace
from typing import Any

from ..protocol import GeneratedSpan, RolloutGroupRequest, RolloutGroupResult, Trajectory
from ..runner import RolloutCancelled, RolloutContext
from ...agent import AshAgent
from ...models import AgentConfig, ToolResult


class MilesSessionClient:
    """Small synchronous client for the Miles session lifecycle.

    The strategy runs in a worker thread, so stdlib HTTP keeps this adapter
    independent of Ash's optional async dependencies.
    """

    def __init__(self, endpoint: str, *, timeout_seconds: float = 120.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def create(self) -> str:
        payload = self._request("POST", "/sessions", {})
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Miles session server returned no session_id")
        return session_id

    def get(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{session_id}", None)

    def delete(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/sessions/{session_id}", None)
        except Exception:
            # The trajectory has already been read; cleanup must not hide the
            # rollout result or the original agent error.
            pass

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.endpoint + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Miles session server returned a non-object response")
        return value


class MilesSessionAgentRolloutStrategy:
    """Run one AshAgent per allocated slot through Miles v2 sessions."""

    def __init__(self, *, agent_config: AgentConfig | None = None, allow_text_prompt: bool = True):
        # Miles exposes an OpenAI-compatible local endpoint.  Do not inherit
        # AshAgent's Anthropic-oriented standalone default unless the caller
        # explicitly supplies an AgentConfig.
        self.agent_config = agent_config or AgentConfig(model="openai/local")
        self.allow_text_prompt = allow_text_prompt

    def run(self, request: RolloutGroupRequest, context: RolloutContext) -> RolloutGroupResult:
        if context.environment_provider is None:
            raise RuntimeError("agent-loop rollout requires an environment provider")
        endpoint = request.session_server_endpoint or request.model_endpoint
        if "/sessions" in endpoint:
            raise ValueError("session_server_endpoint must be the Miles server base URL, not a session URL")
        trajectories: list[Trajectory] = []
        model_calls = 0
        tool_calls = 0
        for slot in request.sample_slots[: request.max_samples]:
            context.check_cancelled()
            sandbox = context.environment_provider.spawn(request)
            session_client = MilesSessionClient(endpoint, timeout_seconds=context.remaining_wall_time_seconds or 120.0)
            session_id = session_client.create()
            try:
                config = self._config(request, session_client, session_id)
                executor = self._executor(sandbox, context=context, max_tool_calls=request.budgets.max_tool_calls)
                agent = AshAgent(
                    config,
                    executor=executor,
                    agent_id=f"{request.rollout_job_id}:{slot.sample_slot_id}",
                    sandbox_id=str(getattr(sandbox, "sandbox_id", "unknown")),
                )
                agent.stream = False
                status = agent.run(
                    task="",
                    instance_id=slot.sample_slot_id,
                    initial_messages=_initial_messages(request.prompt, allow_text=self.allow_text_prompt),
                )
                model_calls += agent.cost.api_calls
                tool_calls += _tool_call_count(agent.trajectory.messages)
                state = session_client.get(session_id)
                trajectories.append(_trajectory_from_session(request, slot.sample_slot_id, state, status))
            finally:
                session_client.delete(session_id)
                context.environment_provider.destroy(sandbox)

        return RolloutGroupResult(
            rollout_job_id=request.rollout_job_id,
            prompt_group_id=request.prompt_group_id,
            status="completed",
            max_samples=request.max_samples,
            trajectories=trajectories,
            consumed_budget={"model_calls": model_calls, "tool_calls": tool_calls},
        )

    def _config(self, request: RolloutGroupRequest, client: MilesSessionClient, session_id: str) -> AgentConfig:
        sampling = dict(request.sampling_params)
        model = request.model or sampling.pop("model", None) or self.agent_config.model
        # LiteLLM needs a provider prefix for an OpenAI-compatible local
        # endpoint; preserve an explicitly qualified provider unchanged.
        if "/" not in str(model):
            model = f"openai/{model}"
        max_tokens = sampling.get("max_tokens", sampling.get("max_new_tokens", self.agent_config.max_tokens))
        temperature = sampling.get("temperature", self.agent_config.temperature)
        return replace(
            self.agent_config,
            model=str(model),
            api_base=f"{client.endpoint}/sessions/{session_id}/v1",
            api_key=self.agent_config.api_key or "EMPTY",
            max_tokens=int(max_tokens),
            temperature=None if temperature is None else float(temperature),
            prompt_cache=False,
            step_limit=min(self.agent_config.step_limit, request.budgets.max_model_calls),
        )

    @staticmethod
    def _executor(sandbox, *, context: RolloutContext, max_tool_calls: int):
        call = getattr(sandbox, "call_agent_tool", None)
        if call is None:
            call = getattr(sandbox, "call", None)
        if call is None:
            raise TypeError("environment sandbox must expose call_agent_tool or call")

        calls = 0

        def execute(name: str, args: dict[str, Any]) -> ToolResult:
            nonlocal calls
            context.check_cancelled()
            if calls >= max_tool_calls:
                raise RolloutCancelled("rollout tool-call budget exhausted")
            calls += 1
            result = _run_async(call(name, args))
            if isinstance(result, ToolResult):
                return result
            return ToolResult.from_sdk(result)

        return execute


def _initial_messages(prompt: str | list[dict[str, Any]], *, allow_text: bool) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    if not allow_text:
        raise ValueError("text prompts are disabled for the agent-loop strategy")
    return [{"role": "user", "content": prompt}]


def _trajectory_from_session(
    request: RolloutGroupRequest, sample_slot_id: str, state: dict[str, Any], status: str
) -> Trajectory:
    records = state.get("records")
    metadata = state.get("metadata") or {}
    token_ids = metadata.get("accumulated_token_ids")
    if not isinstance(records, list) or not records or not isinstance(token_ids, list) or not token_ids:
        raise ValueError("Miles session did not return records and accumulated_token_ids")

    spans: list[GeneratedSpan] = []
    final_messages: list[dict[str, Any]] | None = None
    final_text = ""
    for record in records:
        req = record.get("request") or {}
        response_envelope = record.get("response") or {}
        response = response_envelope.get("choices", [{}])[0]
        info = response.get("meta_info") or {}
        pairs = info.get("output_token_logprobs") or []
        output_ids = [int(pair[1]) for pair in pairs]
        input_ids = [int(token) for token in req.get("input_ids", [])]
        if not output_ids or not input_ids:
            raise ValueError("Miles session record lacks token-level generation data")
        message = response.get("message") or {}
        final_messages = [dict(item) for item in req.get("messages", [])] + [dict(message)]
        final_text = str(message.get("content") or "")
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

    prompt_length = len(records[0].get("request", {}).get("input_ids", []))
    if final_messages is None:
        raise ValueError("Miles session returned no final messages")
    return Trajectory(
        sample_slot_id=sample_slot_id,
        branch_id=f"{request.rollout_job_id}:{sample_slot_id}",
        messages=final_messages,
        token_ids=[int(token) for token in token_ids],
        prompt_length=prompt_length,
        generated_spans=spans,
        response_text=final_text,
        status="completed" if status == "completed" else "failed",
        metadata={"strategy": "miles-session-agent-loop", "session": metadata.get("tree", {})},
    )


def _tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(len(message.get("tool_calls") or []) for message in messages if message.get("role") == "assistant")


def _run_async(awaitable):
    if not asyncio.iscoroutine(awaitable) and not isinstance(awaitable, asyncio.Future):
        return awaitable
    return asyncio.run(awaitable)
