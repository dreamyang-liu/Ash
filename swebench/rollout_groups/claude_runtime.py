"""Claude Agent SDK execution over one Ash sandbox and one Miles session."""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ash_sandbox.panel import compile_panel, load_declaration, parse_agent_tool

from ..agent.tools import DEFAULT_PANEL, RUNTIME_SCHEMA, resolve_panel
from ..models import ToolResult
from .protocol import RolloutGroupRequest
from .runner import EnvironmentCheckpoint, RolloutCancelled, RolloutContext
from .session_runtime import MilesSessionClient


_SYSTEM_PROMPT = """\
You are a software engineering agent working in an isolated sandbox.
Use only the Ash MCP tools exposed to you. The repository is in /testbed.
Solve the user's task, verify the result when practical, and stop naturally
when the work is complete. Do not ask the user to execute commands for you.
"""


@dataclass
class ClaudeRunResult:
    status: str
    session_id: str
    messages: list[dict[str, Any]]
    model_calls: int
    tool_calls: int
    tool_seconds: float
    checkpoint: EnvironmentCheckpoint | None = None
    checkpoint_tool_use_id: str | None = None
    checkpoint_message_uuid: str | None = None
    synthetic_message_uuids: list[str] = field(default_factory=list)
    model_response_ids: list[str] = field(default_factory=list)


class ClaudeAgentRuntime:
    """Drive the official Claude harness without importing ``AshAgent``."""

    tool_names = ("shell", "text_editor", "grep_files", "process")

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str = _SYSTEM_PROMPT,
        tool_panel: str = DEFAULT_PANEL,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.tool_panel = tool_panel

    def run_parent(
        self,
        *,
        request: RolloutGroupRequest,
        context: RolloutContext,
        session_client: MilesSessionClient,
        miles_session_id: str,
        sandbox: Any,
        agent_id: str,
        max_model_calls: int | None,
        max_tool_calls: int | None,
        capture_checkpoint: bool,
        config_dir: str | Path,
        on_checkpoint: Callable[[EnvironmentCheckpoint], None] | None = None,
    ) -> ClaudeRunResult:
        return asyncio.run(
            self._run(
                request=request,
                context=context,
                session_client=session_client,
                miles_session_id=miles_session_id,
                sandbox=sandbox,
                agent_id=agent_id,
                max_model_calls=max_model_calls,
                max_tool_calls=max_tool_calls,
                capture_checkpoint=capture_checkpoint,
                config_dir=config_dir,
                on_checkpoint=on_checkpoint,
            )
        )

    def run_child(
        self,
        *,
        request: RolloutGroupRequest,
        context: RolloutContext,
        session_client: MilesSessionClient,
        miles_session_id: str,
        sandbox: Any,
        agent_id: str,
        max_model_calls: int | None,
        max_tool_calls: int | None,
        parent_claude_session_id: str,
        checkpoint_message_uuid: str,
        config_dir: str | Path,
    ) -> ClaudeRunResult:
        return asyncio.run(
            self._run(
                request=request,
                context=context,
                session_client=session_client,
                miles_session_id=miles_session_id,
                sandbox=sandbox,
                agent_id=agent_id,
                max_model_calls=max_model_calls,
                max_tool_calls=max_tool_calls,
                capture_checkpoint=False,
                config_dir=config_dir,
                resume=parent_claude_session_id,
                resume_session_at=checkpoint_message_uuid,
            )
        )

    async def _run(
        self,
        *,
        request: RolloutGroupRequest,
        context: RolloutContext,
        session_client: MilesSessionClient,
        miles_session_id: str,
        sandbox: Any,
        agent_id: str,
        max_model_calls: int | None,
        max_tool_calls: int | None,
        capture_checkpoint: bool,
        config_dir: str | Path,
        on_checkpoint: Callable[[EnvironmentCheckpoint], None] | None = None,
        resume: str | None = None,
        resume_session_at: str | None = None,
    ) -> ClaudeRunResult:
        sdk = _load_claude_sdk()
        panel = _build_claude_panel(self.tool_panel)
        schemas = {
            item["function"]["name"]: item["function"]
            for item in panel.schema
            if item["function"]["name"] in self.tool_names
        }
        if set(schemas) != set(self.tool_names):
            missing = sorted(set(self.tool_names) - set(schemas))
            raise RuntimeError(f"Claude Ash MCP panel is missing tools: {missing}")

        call = getattr(sandbox, "call_agent_tool", None) or getattr(sandbox, "call", None)
        if call is None:
            executor_for = getattr(sandbox, "executor_for", None)
            if executor_for is not None:
                call = executor_for(agent_id)
        if call is None:
            raise TypeError("environment sandbox must expose call_agent_tool or call")

        tool_calls = 0
        tool_seconds = 0.0
        checkpoint: EnvironmentCheckpoint | None = None
        checkpoint_tool_use_id: str | None = None
        tool_lock = asyncio.Lock()

        async def execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
            nonlocal tool_calls, tool_seconds
            # Serialize the offered tools. The stream loop creates a checkpoint
            # only after every result from one assistant tool batch is visible,
            # so no pending call can mutate the environment behind that point.
            async with tool_lock:
                context.check_cancelled()
                if max_tool_calls is not None and tool_calls >= max_tool_calls:
                    raise RolloutCancelled("rollout tool-call budget exhausted")
                tool_calls += 1
                context.update_progress("tool_execution", tool_calls=tool_calls)
                started = time.monotonic()
                try:
                    result = await _call_tool(call, name, panel, args)
                finally:
                    tool_seconds += time.monotonic() - started
            if not isinstance(result, ToolResult):
                result = ToolResult.from_sdk(result)
            text = result.output or (f"Error: {result.error}" if result.error else "")
            return {
                "content": [{"type": "text", "text": text}],
                # Miles preserves this error bit in Claude history while the
                # local SGLang conversion treats its text as ordinary tool output.
                "is_error": not result.success,
            }

        sdk_tools = []
        for name in self.tool_names:
            schema = schemas[name]

            async def handler(args, tool_name=name):
                return await execute(tool_name, args)

            sdk_tools.append(
                sdk.tool(name, schema.get("description", ""), schema["parameters"])(handler)
            )

        config_dir = str(config_dir)
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        try:
            options = sdk.ClaudeAgentOptions(
                model=request.model or self.model,
                system_prompt=self.system_prompt,
                tools=[],
                mcp_servers={
                    "ash-sandbox": sdk.create_sdk_mcp_server(
                        "ash-sandbox", tools=sdk_tools
                    )
                },
                strict_mcp_config=True,
                allowed_tools=[
                    f"mcp__ash-sandbox__{name}" for name in self.tool_names
                ],
                setting_sources=[],
                permission_mode="dontAsk",
                max_turns=max_model_calls,
                thinking={"type": "disabled"},
                effort=_claude_effort(request.sampling_params),
                cwd=Path(config_dir),
                env={
                    "ANTHROPIC_API_KEY": "EMPTY",
                    "ANTHROPIC_BASE_URL": (
                        f"{session_client.endpoint}/sessions/{miles_session_id}"
                    ),
                    "CLAUDE_CONFIG_DIR": config_dir,
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY": "1",
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(
                        request.sampling_params.get(
                            "max_tokens",
                            request.sampling_params.get("max_new_tokens", 32000),
                        )
                    ),
                },
                resume=resume,
                resume_session_at=resume_session_at,
                fork_session=resume is not None,
            )
            prompt = "" if resume is not None else _prompt_text(request.prompt)
            messages = []
            result_message = None
            model_message_ids: set[str] = set()
            model_response_ids: list[str] = []
            synthetic_message_uuids: list[str] = []
            checkpoint_message_uuid = None
            tool_result_uuids: dict[str, str] = {}
            pending_tool_use_ids: set[str] = set()

            stream = sdk.query(prompt=prompt, options=options)
            cancel_watch_done = threading.Event()

            def propagate_cancellation() -> None:
                wait = getattr(context.cancel_event, "wait", None)
                if not callable(wait):
                    return
                while not cancel_watch_done.is_set():
                    if wait(timeout=0.1):
                        try:
                            session_client.delete(miles_session_id)
                        except Exception:
                            pass
                        return

            cancel_watcher = threading.Thread(
                target=propagate_cancellation,
                name=f"ash-claude-cancel-{miles_session_id}",
                daemon=True,
            )
            cancel_watcher.start()
            try:
                async with asyncio.timeout(context.remaining_wall_time_seconds):
                    async for message in stream:
                        context.check_cancelled()
                        messages.append(_message_dict(message))
                        if isinstance(message, sdk.AssistantMessage):
                            model_message_id = (
                                message.message_id
                                or message.uuid
                                or str(len(messages))
                            )
                            if model_message_id not in model_message_ids:
                                model_message_ids.add(model_message_id)
                                model_response_ids.append(model_message_id)
                            pending_tool_use_ids.update(_tool_use_ids(message))
                            context.update_progress(
                                "model_generation", model_calls=len(model_message_ids)
                            )
                        elif isinstance(message, sdk.UserMessage):
                            if resume is not None and _is_empty_user_message(message):
                                if message.uuid:
                                    synthetic_message_uuids.append(message.uuid)
                            completed_tool_ids = _tool_result_ids(message)
                            for tool_use_id in completed_tool_ids:
                                if message.uuid:
                                    tool_result_uuids[tool_use_id] = message.uuid
                                pending_tool_use_ids.discard(tool_use_id)
                            if (
                                capture_checkpoint
                                and checkpoint is None
                                and completed_tool_ids
                                and not pending_tool_use_ids
                            ):
                                checkpoint_tool_use_id = completed_tool_ids[-1]
                                checkpoint = await asyncio.to_thread(
                                    context.environment_provider.create_checkpoint,
                                    sandbox,
                                    owner_job_id=request.rollout_job_id,
                                    name=(
                                        f"{request.rollout_job_id}-claude-tool-"
                                        f"{tool_calls}-{uuid.uuid4().hex[:12]}"
                                    ),
                                )
                                if on_checkpoint is not None:
                                    on_checkpoint(checkpoint)
                        elif isinstance(message, sdk.ResultMessage):
                            result_message = message
            finally:
                cancel_watch_done.set()
                cancel_watcher.join(timeout=0.2)
                aclose = getattr(stream, "aclose", None)
                if callable(aclose):
                    await aclose()

            if checkpoint_tool_use_id is not None:
                checkpoint_message_uuid = tool_result_uuids.get(
                    checkpoint_tool_use_id
                )
        except BaseException:
            if checkpoint is not None and on_checkpoint is None:
                context.environment_provider.release_checkpoint(checkpoint)
            raise

        if result_message is None:
            raise RuntimeError("Claude Agent SDK returned no terminal result")
        if result_message.is_error:
            raise RuntimeError(result_message.result or "Claude Agent SDK failed")
        if checkpoint is not None and checkpoint_message_uuid is None:
            raise RuntimeError(
                "Claude tool checkpoint has no matching tool_result message UUID"
            )
        if not model_response_ids:
            raise RuntimeError("Claude Agent SDK returned no model response IDs")
        status = "completed"
        if result_message.terminal_reason in {"max_turns", "aborted_streaming", "aborted_tools"}:
            status = "truncated"
        return ClaudeRunResult(
            status=status,
            session_id=result_message.session_id,
            messages=messages,
            model_calls=len(model_message_ids),
            tool_calls=tool_calls,
            tool_seconds=tool_seconds,
            checkpoint=checkpoint,
            checkpoint_tool_use_id=checkpoint_tool_use_id,
            checkpoint_message_uuid=checkpoint_message_uuid,
            synthetic_message_uuids=synthetic_message_uuids,
            model_response_ids=model_response_ids,
        )


def _load_claude_sdk():
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk is required for Claude rollout strategies"
        ) from exc
    return claude_agent_sdk


async def _call_tool(call, name: str, panel, args: dict[str, Any]):
    runtime_name, runtime_args = panel.route(name, dict(args))
    if inspect.iscoroutinefunction(call):
        result = call(runtime_name, runtime_args)
    else:
        # ``AshSession.executor_for`` is synchronous and owns a private event
        # loop. Calling it on the Claude SDK loop would attempt a nested
        # ``run_until_complete``; it also blocks message streaming for the
        # duration of a tool call. Keep that adapter on a worker thread while
        # native async sandbox clients remain on this loop.
        result = await asyncio.to_thread(call, runtime_name, runtime_args)
    if inspect.isawaitable(result):
        return await result
    return result


def _build_claude_panel(name_or_path):
    """Compile the four stable Ash runtime views without process-global state."""
    path = resolve_panel(name_or_path)
    if path.suffix == ".json":
        import json

        manifest = json.loads(path.read_text())
    else:
        import yaml

        manifest = yaml.safe_load(path.read_text())
    specs = [
        parse_agent_tool(item)
        for item in manifest["agent_tools"]
        if item["name"] in ClaudeAgentRuntime.tool_names
    ]
    schemas = compile_panel(specs, load_declaration(RUNTIME_SCHEMA))
    from ..agent.tools import ToolPanel

    return ToolPanel(schema=schemas, views={spec.name: spec for spec in specs})


def _prompt_text(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt
    parts = []
    for message in prompt:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        else:
            parts.append(f"{role}: {content!r}")
    return "\n\n".join(parts)


def _claude_effort(sampling_params: dict[str, Any]) -> str | None:
    value = sampling_params.get("reasoning_effort")
    if value in {"low", "medium", "high", "xhigh", "max"}:
        return value
    return None


def _message_dict(message: Any) -> dict[str, Any]:
    role = None
    message_type = type(message).__name__
    if message_type == "AssistantMessage":
        role = "assistant"
    elif message_type == "UserMessage":
        role = "user"
    content = []
    for block in getattr(message, "content", []) if isinstance(getattr(message, "content", []), list) else []:
        if hasattr(block, "__dataclass_fields__"):
            content.append(
                {
                    name: copy.deepcopy(getattr(block, name))
                    for name in block.__dataclass_fields__
                }
            )
        else:
            content.append(copy.deepcopy(block))
    result = {
        "type": message_type,
        "uuid": getattr(message, "uuid", None),
        "message_id": getattr(message, "message_id", None),
        "content": content if content else getattr(message, "content", None),
    }
    if role is not None:
        result["role"] = role
    return result


def _tool_result_ids(message: Any) -> list[str]:
    return [
        str(block.tool_use_id)
        for block in getattr(message, "content", [])
        if getattr(block, "tool_use_id", None)
    ]


def _tool_use_ids(message: Any) -> list[str]:
    return [
        str(block.id)
        for block in getattr(message, "content", [])
        if getattr(block, "id", None)
        and type(block).__name__ == "ToolUseBlock"
    ]


def _is_empty_user_message(message: Any) -> bool:
    content = getattr(message, "content", None)
    return content == "" or content == []
