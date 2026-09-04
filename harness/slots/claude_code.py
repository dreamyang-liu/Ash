"""Claude Code slot, driven through claude-agent-sdk.

Unlike the CLI slots this one is not a stdout parser: the SDK hands us typed
messages plus a real control seam the other slots do not offer.

The verdict point is a **PreToolUse hook**, not ``can_use_tool``. That choice is
load-bearing: with ``permission_mode="bypassPermissions"`` (what an unattended
eval run needs) the SDK auto-approves every call *before* ``can_use_tool`` is
consulted, so a callback mounted there never fires -- silently. The SDK emits
``CanUseToolShadowedWarning`` and points at PreToolUse. Mounting the verdict on
``can_use_tool`` would have meant no builtin-tool denial and, worse, no step
boundaries at all, i.e. a rollback feature that records zero checkpoints.
``can_use_tool`` is still registered as a backstop for non-bypass modes.

Denial matters for correctness, not just policy: builtin Bash/Read/Edit run on
the *host* where the SDK lives, so their side effects land outside the sandbox
and outside every snapshot. A leaked builtin call silently corrupts both the
environment and the trajectory. ``disallowed_tools`` plus the ``can_use_tool``
backstop is deliberately belt-and-braces (see tests/test_builtin_denied.py).

Tool wiring accepts either an ``McpWiring`` (stdio subprocess / remote URL, same
as the other slots) or, via ``task.extra["sdk_mcp_server"]``, an in-process SDK
MCP server object -- the shape marathon_claude_code.py uses so the harness keeps
ownership of the sandbox after the stream closes (needed for grading).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from harness.core.events import (
    AGENT_ERROR,
    AGENT_MESSAGE,
    RUN_FINISHED,
    RUN_RESULT,
    RUN_STARTED,
    SESSION_REF,
    TOOL_STARTED,
    TURN_COMPLETED,
    Usage,
)
from harness.core.journal import JournalWriter
from harness.core.slot import (
    AgentSlot,
    McpWiring,
    SlotCapabilities,
    SlotResult,
    TaskSpec,
)
from harness.normalize import claude_code as cc_normalize

#: Builtin tools that would execute on the host instead of in the sandbox --
#: plus everything that acts outside this run: scheduling, cross-session
#: messaging, worktrees, subagents. Measured 2026-09-04: one DeepSWE agent
#: created a `* * * * *` CronCreate prompt while its sandbox was gone; Claude
#: Code kept delivering it into OTHER tasks' sessions (same cwd) for hours --
#: 7 deliveries, 2 into unrelated branch attempts. The eval offers exactly the
#: two MCP sandbox tools (plus ToolSearch, which loads their schemas, and the
#: todo-list tools, which touch nothing).
DENIED_BUILTINS = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Edit",
    "Write",
    "NotebookEdit",
    "Glob",
    # out-of-band: scheduling, messaging, background work, host git state
    "Task",
    "Agent",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "Workflow",
    "SendMessage",
    "ListAgents",
    "EnterWorktree",
    "ExitWorktree",
    "Skill",
    "ReportFindings",
    "Grep",
    "WebFetch",
    "WebSearch",
)

_DENY_REASON = (
    "Builtin tools are disabled in this harness: they would run outside the "
    "sandbox. Use the MCP sandbox tools instead."
)


class ClaudeCodeSlot(AgentSlot):
    name = "claude-code"
    binary = "claude"
    capabilities = SlotCapabilities(
        resume=True,
        fork=True,            # SDK: resume + fork_session
        mcp_stdio=True,
        mcp_remote=True,
        emits_usage=True,
        deny_builtin_tools=True,
    )

    def __init__(self, *, on_tool_boundary: Optional[Callable[[int], None]] = None) -> None:
        """``on_tool_boundary(index)`` fires as each tool call is approved.

        That is the step boundary for an external agent (it has no model-call
        hook), i.e. where per-step checkpoints are taken -- the trigger
        marathon_claude_code.py established.
        """
        self._on_tool_boundary = on_tool_boundary
        self._tool_index = 0
        self._denied: List[str] = []
        self._boundaries = 0

    # --- public ------------------------------------------------------------
    @property
    def denied_calls(self) -> List[str]:
        """Names of builtin tools the model attempted (should stay empty)."""
        return list(self._denied)

    @property
    def boundary_count(self) -> int:
        """Approved tool calls seen. Zero after a tool-using run means the
        verdict seam is not wired -- see the PreToolUse note in the docstring."""
        return self._boundaries

    def run(
        self,
        task: TaskSpec,
        journal: JournalWriter,
        mcp: Optional[McpWiring] = None,
    ) -> SlotResult:
        return asyncio.run(self.run_async(task, journal, mcp))

    async def run_async(
        self,
        task: TaskSpec,
        journal: JournalWriter,
        mcp: Optional[McpWiring] = None,
    ) -> SlotResult:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except ImportError as exc:  # pragma: no cover - optional dependency
            message = "claude-agent-sdk is required for the claude-code slot: %s" % exc
            journal.emit(RUN_FINISHED, status="error", error=message)
            return SlotResult(status="error", error=message)

        mcp_servers, allowed = self._mcp_config(task, mcp)
        options = self._build_options(ClaudeAgentOptions, task, mcp_servers, allowed)

        journal.emit(
            RUN_STARTED,
            slot=self.name,
            slot_version=self.version(),
            model=task.model,
            cwd=task.cwd,
            task_prompt=task.prompt,
            config={
                "mcp_servers": sorted(mcp_servers),
                "allowed_tools": allowed,
                "disallowed_tools": list(DENIED_BUILTINS),
                "permission_mode": getattr(options, "permission_mode", None),
            },
        )

        usage = Usage()
        session_id: Optional[str] = None
        result_text = ""
        last_text = ""
        status = "completed"
        error: Optional[str] = None
        stream = None

        try:
            stream = query(prompt=task.prompt, options=options)
            async for message in _with_timeout(stream, task.timeout_s):
                for event_type, payload in cc_normalize.normalize(message):
                    journal.emit(event_type, **payload)
                    if event_type == TURN_COMPLETED and payload.get("usage"):
                        usage.add_dict(payload["usage"])
                    elif event_type == SESSION_REF and payload.get("native_session_id"):
                        session_id = payload["native_session_id"]
                    elif event_type == RUN_RESULT and payload.get("text"):
                        result_text = payload["text"]
                    elif event_type == AGENT_MESSAGE and payload.get("text"):
                        last_text = payload["text"]
        except asyncio.TimeoutError:
            status = "timeout"
            error = "timed out after %ss" % task.timeout_s
            journal.emit(AGENT_ERROR, message=error)
        except Exception as exc:  # noqa: BLE001 - slot must journal, never crash caller
            status = "error"
            error = "%s: %s" % (type(exc).__name__, exc)
            journal.emit(AGENT_ERROR, message=error)
        finally:
            await _aclose(stream)

        journal.emit(
            RUN_FINISHED,
            status=status,
            usage=usage.as_dict(),
            error=error,
            denied_builtin_calls=self._denied,
        )
        return SlotResult(
            status=status,
            final_text=result_text or last_text,
            usage=usage.as_dict(),
            native_session_id=session_id,
            error=error,
        )

    # --- internals ---------------------------------------------------------
    def _mcp_config(self, task: TaskSpec, mcp: Optional[McpWiring]):
        """Return ``(mcp_servers dict, allowed_tools list)``."""
        extra = task.extra or {}
        servers: Dict[str, Any] = {}
        name = mcp.name if mcp else str(extra.get("mcp_name", "ash"))

        sdk_server = extra.get("sdk_mcp_server")
        if sdk_server is not None:
            servers[name] = sdk_server
        elif mcp and mcp.command:
            entry: Dict[str, Any] = {
                "type": "stdio",
                "command": mcp.command[0],
                "args": list(mcp.command[1:]),
            }
            if mcp.env:
                entry["env"] = dict(mcp.env)
            servers[name] = entry
        elif mcp and mcp.url:
            entry = {"type": "http", "url": mcp.url}
            if mcp.headers:
                entry["headers"] = dict(mcp.headers)
            servers[name] = entry

        tools = extra.get("mcp_tools")
        if tools:
            allowed = ["mcp__%s__%s" % (name, t) for t in tools]
        elif servers:
            allowed = ["mcp__%s" % name]  # whole server
        else:
            allowed = []
        return servers, allowed

    def _build_options(self, options_cls, task: TaskSpec, mcp_servers, allowed):
        extra = task.extra or {}
        kwargs: Dict[str, Any] = {
            "cwd": task.cwd,
            "mcp_servers": mcp_servers,
            "allowed_tools": allowed,
            "disallowed_tools": list(DENIED_BUILTINS),
            "permission_mode": extra.get("permission_mode", "bypassPermissions"),
            # PreToolUse is the seam that actually fires under bypassPermissions.
            "hooks": self._hooks(),
        }
        if kwargs["permission_mode"] != "bypassPermissions":
            # Only useful (and only non-shadowed) in stricter modes; registering
            # it under bypassPermissions just triggers CanUseToolShadowedWarning.
            kwargs["can_use_tool"] = self._can_use_tool
        if task.model:
            kwargs["model"] = task.model
        if extra.get("system_prompt"):
            kwargs["system_prompt"] = extra["system_prompt"]
        if extra.get("setting_sources") is not None:
            # Empty list == ignore local CLAUDE.md / .claude settings, the
            # SDK equivalent of `claude --bare`. Eval runs should pass [].
            kwargs["setting_sources"] = extra["setting_sources"]
        if extra.get("resume_session_id"):
            kwargs["resume"] = extra["resume_session_id"]
            if extra.get("fork"):
                kwargs["fork_session"] = True
            if extra.get("resume_session_at"):
                # Truncate the resumed conversation at this transcript entry
                # (uuid). Without it a fork carries the WHOLE parent
                # conversation -- including everything after the step whose
                # filesystem snapshot the branch starts from.
                kwargs["resume_session_at"] = extra["resume_session_at"]
        if extra.get("max_turns"):
            kwargs["max_turns"] = extra["max_turns"]
        if task.env:
            env = dict(task.env)
            if "ANTHROPIC_BASE_URL" in env:
                # The run routed this agent's LLM traffic somewhere -- a gateway,
                # a vLLM, an RL checkpoint. The CLI's provider-direct modes
                # (Bedrock/Vertex, usually turned on by the developer's own
                # ~/.claude settings) IGNORE the base URL entirely, so the run
                # completes against the wrong provider while the gateway records
                # nothing: a silently bypassed gateway is worse than a failed
                # run, because the budget it was supposed to enforce was not.
                # Explicit settings in the task env still win.
                env.setdefault("CLAUDE_CODE_USE_BEDROCK", "0")
                env.setdefault("CLAUDE_CODE_USE_VERTEX", "0")
            kwargs["env"] = env

        for key in list(kwargs):
            # Tolerate SDK version differences rather than crashing on an
            # unknown kwarg; contracts/claude-code.yaml asserts the ones we
            # actually depend on.
            if not _accepts(options_cls, key):
                kwargs.pop(key)
        return options_cls(**kwargs)

    def _hooks(self) -> Optional[dict]:
        """PreToolUse matcher carrying the verdict + step-boundary callback."""
        try:
            from claude_agent_sdk import HookMatcher
        except ImportError:  # pragma: no cover - optional dependency
            return None
        return {"PreToolUse": [HookMatcher(hooks=[self._pre_tool_use])]}

    async def _pre_tool_use(self, input_data: dict, tool_use_id, context: Any = None):
        """The live verdict point (fires even under bypassPermissions)."""
        tool_name = (input_data or {}).get("tool_name") or ""
        tool_input = (input_data or {}).get("tool_input") or {}

        if self._is_denied(tool_name):
            self._denied.append(tool_name)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": _DENY_REASON,
                }
            }

        self._note_boundary()
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": tool_input,
            }
        }

    async def _can_use_tool(self, tool_name: str, input_data: dict, context: Any = None):
        """Backstop for permission modes that do consult this callback."""
        if self._is_denied(tool_name):
            self._denied.append(tool_name)
            return {"behavior": "deny", "message": _DENY_REASON}
        self._note_boundary()
        return {"behavior": "allow", "updatedInput": input_data}

    @staticmethod
    def _is_denied(tool_name: str) -> bool:
        if tool_name.startswith("mcp__"):
            return False
        return tool_name in DENIED_BUILTINS

    def _note_boundary(self) -> None:
        self._tool_index += 1
        self._boundaries += 1
        if self._on_tool_boundary is not None:
            try:
                self._on_tool_boundary(self._tool_index)
            except Exception:  # noqa: BLE001 - a checkpoint failure must not kill the run
                pass


def _accepts(cls, name: str) -> bool:
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        return name in {f.name for f in dataclasses.fields(cls)}
    try:
        return name in inspect.signature(cls).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return True


async def _with_timeout(stream, timeout_s: Optional[float]):
    """Yield from an async iterator with a per-item timeout budget."""
    if not timeout_s:
        async for item in stream:
            yield item
        return

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    iterator = stream.__aiter__()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return
        yield item


async def _aclose(stream) -> None:
    if stream is None:
        return
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass
