"""Agent loop for SWE-bench.

Everything around the loop lives elsewhere: model calls + caching + streaming +
retries in `llm.py`, prompts in `prompts.py`, guardrails in `guardrails.py`,
cross-cutting concerns (budget warnings, output truncation) as pluggable
`hooks.py`, conversation/trajectory state in `conversation.py`, and the sandbox
SDK behind the `executor` callable. What remains here is the loop: query the
model, run tool calls, repeat.
"""

import json
import time
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from ..models import AgentConfig, CostTracker, ToolResult, Trajectory
from .prompts import build_system_prompt, build_instance_message  # re-exported
from .conversation import Conversation
from .guardrails import Guardrails
from .llm import LLMClient, ThinkingLoopError
from .tools import tool_summary, TOOLS_SCHEMA, BASH_ONLY_SCHEMA, route_agent_tool
from .trace import ToolTraceWriter, new_run_id
from . import hooks

__all__ = ["AshAgent", "build_system_prompt", "build_instance_message",
           "TOOLS_SCHEMA", "BASH_ONLY_SCHEMA"]


class AshAgent:
    """Agent that uses tool calls to solve SWE-bench tasks."""

    def __init__(self, config: AgentConfig,
                 executor: Callable[[str, dict], ToolResult],
                 on_step: Optional[Callable[[int, str, str], None]] = None,
                 trace_dir: Optional[Path] = None,
                 run_id: Optional[str] = None,
                 agent_id: str = "agent",
                 sandbox_id: str = "default"):
        self.config = config
        self.executor = executor          # executor(tool_name, args) -> ToolResult
        self.on_step = on_step            # on_step(step_num, kind, text)
        self.trace_dir = trace_dir
        self.run_id = run_id
        self.agent_id = agent_id
        self.sandbox_id = sandbox_id
        self.trajectory = Trajectory()
        self.cost = CostTracker()
        self._tools_schema: list[dict] = []
        self._trace_file = None
        self._event_trace: Optional[ToolTraceWriter] = None
        self._warned = False
        self.stream = True                # set False to disable streaming (parallel mode)
        self.before_query_hooks = list(hooks.DEFAULT_BEFORE_QUERY)
        self.result_processors = list(hooks.DEFAULT_RESULT_PROCESSORS)

    def set_tools_schema(self, schema: list[dict]):
        self._tools_schema = schema

    def _trace(self, text: str):
        if self._trace_file:
            self._trace_file.write(text)
            self._trace_file.flush()

    def _run_tool(self, tc, conv: Conversation, guardrails: Guardrails,
                  turn_id: str) -> None:
        """Execute one tool call, trace it, and record its result on the conversation."""
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            args = {}
        summary = tool_summary(name, args)
        if self.on_step:
            self.on_step(self.cost.api_calls, name, summary)
        self._trace(f"\n> {name} {summary}\n")

        result = None
        error_kind = None
        if name == "bash":  # bash_only mode alias
            exec_name, exec_args = "shell", dict(args)
        else:
            try:
                exec_name, exec_args = route_agent_tool(name, args)
            except KeyError as exc:
                exec_name, exec_args = name, dict(args)
                result = ToolResult(success=False, output="", error=str(exc))
                error_kind = "routing"

        call_id = uuid4().hex
        if self._event_trace:
            self._event_trace.emit(
                "tool.started",
                turn_id=turn_id,
                call_id=call_id,
                agent={"name": name, "args": args},
                runtime={"name": exec_name, "args": exec_args},
            )

        started_at = time.perf_counter()
        if result is None:
            if exec_name != name:
                self._trace(f"[runtime] {exec_name} {tool_summary(exec_name, exec_args)}\n")
            warning = guardrails.check(exec_name, exec_args)
            result = self.executor(exec_name, exec_args)
            if not result.success:
                error_kind = "runtime"
        else:
            warning = None
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        raw_content = result.output if result.success else \
            f"Error: {result.error or 'Unknown error'}\n{result.output}"
        content = raw_content
        if warning:
            content += f"\n\n{warning}"
        for proc in self.result_processors:
            content = proc(content, name, args, result)

        if self._event_trace:
            output_truncated = _runtime_output_truncated(exec_name, result.output)
            result_event = {
                "output": result.output,
                "error": result.error,
                "output_bytes": len(result.output.encode("utf-8")),
                "output_truncated": output_truncated,
            }
            payload = {
                "turn_id": turn_id,
                "call_id": call_id,
                "status": "ok" if result.success else "error",
                "result": result_event,
                "duration_ms": duration_ms,
            }
            if error_kind:
                payload["error_kind"] = error_kind
            if content != result.output:
                payload["observation"] = content
            process_id = _background_process_id(exec_name, exec_args, result)
            if process_id:
                payload["process_id"] = process_id
            self._event_trace.emit("tool.finished", **payload)

        self._trace(f"{content}\n")
        conv.add_tool_result(tc.id, content, tool_name=name, tool_args=args, success=result.success)

    def _query(self, llm: LLMClient, conv: Conversation, step_n: int):
        """Return the model message, or None on an error that should end the run."""
        try:
            return llm.query_with_recovery(conv.messages).choices[0].message
        except ThinkingLoopError:
            err = "repeated thinking loop"
        except Exception as e:
            err = str(e)
            if self.on_step:
                self.on_step(step_n, "error", err)
        self._trace(f"\n[ERROR] {err}\n")
        conv.add_error(err)
        return None

    def _nudge(self, conv: Conversation, message) -> Optional[str]:
        """Handle a turn with no tool calls. Return 'completed' to stop, else None."""
        if not (message.content or "").strip():
            # Empty response — always reprompt
            conv.add_user("You must call a tool to proceed.")
            self._trace("\n[NUDGE] empty response, prompting retry\n")
        elif conv.consecutive_no_tool >= 2:
            return "completed"
        else:
            # Text-only response: continue, but the conversation must end with a user
            # message — Bedrock rejects assistant-prefill.
            conv.add_user("If your fix is complete, stop. Otherwise proceed by calling a tool "
                          "(e.g. run the failing test to verify, or make the next edit).")
            self._trace("\n[NUDGE] text-only response, prompting continuation\n")
        return None

    def run(self, task: str, instance_id: str = "") -> str:
        """Run the agent loop. Returns exit status: completed | step_limit |
        cost_limit | error."""
        self.trajectory = Trajectory()
        self.trajectory.instance_id = instance_id
        self.cost = CostTracker()
        self._warned = False
        active_run_id = self.run_id or new_run_id()
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_name = instance_id or "trace"
            self._trace_file = open(self.trace_dir / f"{trace_name}.log", "w", encoding="utf-8")
            self._event_trace = ToolTraceWriter(
                self.trace_dir / f"{trace_name}.events.jsonl",
                run_id=active_run_id,
                agent_id=self.agent_id,
                sandbox_id=self.sandbox_id,
            )

        llm = LLMClient(self.config, self.cost, self._tools_schema, trace=self._trace, on_step=self.on_step)
        llm.stream = self.stream
        guardrails = Guardrails()

        conv = Conversation(self.trajectory)
        conv.add_system(build_system_prompt(task, self.config))
        conv.add_user(build_instance_message(task, self.config))

        try:
            while True:
                if self.cost.api_calls >= self.config.step_limit:
                    return "step_limit"
                if self.cost.total_cost >= self.config.cost_limit:
                    return "cost_limit"
                for hook in self.before_query_hooks:
                    hook(self, conv)

                message = self._query(llm, conv, self.cost.api_calls + 1)
                if message is None:
                    return "error"
                conv.add_assistant(message)

                if message.tool_calls:
                    turn_id = f"turn-{self.cost.api_calls}"
                    for tc in message.tool_calls:
                        self._run_tool(tc, conv, guardrails, turn_id)
                elif self._nudge(conv, message) == "completed":
                    return "completed"
        finally:
            if self._trace_file:
                self._trace_file.close()
                self._trace_file = None
            if self._event_trace:
                self._event_trace.close()
                self._event_trace = None


def _background_process_id(name: str, args: dict, result: ToolResult) -> Optional[str]:
    if name != "shell" or not args.get("background") or not result.success:
        return None
    try:
        payload = json.loads(result.output)
    except (json.JSONDecodeError, TypeError):
        return None
    pid = payload.get("pid") if isinstance(payload, dict) else None
    return pid if isinstance(pid, str) and pid else None


def _runtime_output_truncated(name: str, output: str) -> bool:
    if "[output truncated:" in output:
        return True
    if name != "process":
        return False
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and bool(
        payload.get("stdout_truncated") or payload.get("stderr_truncated")
    )
