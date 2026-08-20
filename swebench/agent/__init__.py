"""Agent loop for SWE-bench.

Everything around the loop lives elsewhere: model calls + caching + streaming +
retries in `llm.py`, prompts in `prompts.py`, conversation/trajectory state in
`conversation.py`, and the sandbox SDK behind the `executor` callable -- which
also owns tool dispatch, so a manifest-defined tool is handed over by name and
expanded there rather than here. Tool-path
concerns (guardrails, output truncation) are L2 interceptors on a `ToolPipeline`
(`interceptors.py`, `guardrails.py`) rather than loop code, so the MCP proxy and
harness-side mounts get them too; model-path concerns (budget warnings) remain
`hooks.py`. What remains here is the loop: query the model, run tool calls,
repeat.
"""

import json
import time
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from ..models import AgentConfig, CostTracker, ToolResult, Trajectory
from .prompts import build_system_prompt, build_instance_message  # re-exported
from .conversation import Conversation
from .llm import LLMClient, ThinkingLoopError
from .pipeline import EXECUTOR, CallContext, ToolPipeline, mounted_pipeline
from .tools import tool_summary, TOOLS_SCHEMA, BASH_ONLY_SCHEMA, route_agent_tool, is_custom_tool
from .trace import ToolTraceWriter, new_run_id
from . import hooks, interceptors

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
                 sandbox_id: str = "default",
                 pipeline: "ToolPipeline | None" = None):
        self.config = config
        self.executor = executor          # executor(tool_name, args) -> ToolResult
        # Caller-supplied L2 chain, or None to let run() mount the loop's
        # default (guardrails + truncation). Pass ToolPipeline([]) for no
        # interception, or one shared instance to govern several agents
        # together — coordination state lives inside the interceptors.
        self.pipeline = pipeline
        # The chain actually in force, resolved on first use. run() clears it so
        # a defaulted chain is rebuilt per run and guardrail state (files read,
        # edit streaks) never leaks between runs; a caller-supplied chain is
        # reused as-is, because its state is the caller's to own.
        self._pipeline: "ToolPipeline | None" = None
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

    def _resolve_pipeline(self) -> ToolPipeline:
        """The chain to run around this agent's calls.

        An explicit ``pipeline=`` always wins (including ``ToolPipeline([])`` to
        opt out). Otherwise the default is mounted — unless the executor already
        carries one (``piped_executor``, i.e. ``executor_for(pipeline=…)``), in
        which case the caller has stated their governance and stacking the
        default on top would double every rule it shares.
        """
        if self.pipeline is not None:
            return self.pipeline
        if mounted_pipeline(self.executor) is not None:
            return ToolPipeline([])
        return interceptors.default_pipeline()

    def _governed(self, metadata: dict) -> Callable[[str, dict], ToolResult]:
        """This call's executor, wrapped in the L2 pipeline if one is mounted.

        ``metadata`` is the call's shared scratch space: interceptors read the
        raw executor from it (Waggle's probe traffic) and write back facts the
        loop needs afterwards (the pre-truncation output).
        """
        if self._pipeline is None:
            self._pipeline = self._resolve_pipeline()
        pipeline = self._pipeline
        if not pipeline.interceptors:
            return self.executor
        metadata.setdefault(EXECUTOR, self.executor)

        def run(tool_name: str, args: dict) -> ToolResult:
            ctx = CallContext(agent_id=self.agent_id, sandbox_id=self.sandbox_id,
                              tool_name=tool_name, args=dict(args),
                              metadata=metadata)
            return pipeline.execute(ctx, self.executor)
        return run

    def _run_tool(self, tc, conv: Conversation, turn_id: str) -> None:
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
        if is_custom_tool(name):
            # Passed through under its own name. The executor expands it into
            # artifact->shell (ash_sandbox.Sandbox.call_agent_tool), which also
            # remembers where the binary landed, so a repeat call skips the
            # download round-trip. Interceptors therefore see one opaque call --
            # tell Waggle about any such tool via `opaque_writers` so its drift
            # scan still runs (see waggle.py).
            exec_name, exec_args = name, dict(args)
        else:
            # Builtin names ARE translated here, even though the executor would
            # do it too: `bash` must reach the interceptors as `shell`, or a
            # bash_only run would hide every command from the seats that key on
            # the runtime tool -- Waggle's drift scan above all.
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

        metadata: dict = {}
        started_at = time.perf_counter()
        if result is None:
            execute = self._governed(metadata)
            if exec_name != name:
                self._trace(f"[runtime] {exec_name} {tool_summary(exec_name, exec_args)}\n")
            result = execute(exec_name, exec_args)
            if not result.success:
                error_kind = "runtime"
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        # Interceptors may have annotated or bounded the result; the trace
        # records what the runtime returned, the conversation gets what the
        # interceptors produced.
        runtime_output = metadata.get(interceptors.RAW_OUTPUT, result.output)
        runtime_error = metadata.get(interceptors.RAW_ERROR, result.error)
        content = _observation(result.success, result.output, result.error)
        for proc in self.result_processors:
            content = proc(content, name, args, result)

        if self._event_trace:
            result_event = {
                "output": runtime_output,
                "error": runtime_error,
                "output_bytes": len(runtime_output.encode("utf-8")),
                "output_truncated": _output_truncated(runtime_output, result.outcome),
            }
            if result.outcome is not None:
                result_event["command"] = _command_facts(result.outcome)
            payload = {
                "turn_id": turn_id,
                "call_id": call_id,
                "status": "ok" if result.success else "error",
                "result": result_event,
                "duration_ms": duration_ms,
            }
            if error_kind:
                payload["error_kind"] = error_kind
            # Recorded whenever the model saw something other than the
            # runtime's raw output: error formatting (a failure's text is never
            # `output` alone), a guardrail warning, or an elision.
            if content != runtime_output:
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
        self._pipeline = None             # re-resolved per run (see __init__)

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
                        self._run_tool(tc, conv, turn_id)
                elif self._nudge(conv, message) == "completed":
                    return "completed"
        finally:
            if self._trace_file:
                self._trace_file.close()
                self._trace_file = None
            if self._event_trace:
                self._event_trace.close()
                self._event_trace = None


def _observation(success: bool, output: str, error: Optional[str]) -> str:
    """Render a result the way the model sees it.

    A failure is announced, then explained by whichever field holds the reason.
    A command that ran and exited non-zero has output and no error; a call the
    tool refused has an error and no output. Only the fields that carry
    something are shown, so neither case gets a filler line -- and a result
    holding one message in both fields would not print it twice.
    """
    if success:
        return output
    reason = error or ""
    if reason and output and reason != output:
        return f"Error: {reason}\n{output}"
    return f"Error: {reason or output or 'Unknown error'}"


def _background_process_id(name: str, args: dict, result: ToolResult) -> Optional[str]:
    if name != "shell" or not args.get("background") or not result.success:
        return None
    try:
        payload = json.loads(result.output)
    except (json.JSONDecodeError, TypeError):
        return None
    pid = payload.get("pid") if isinstance(payload, dict) else None
    return pid if isinstance(pid, str) and pid else None


#: Command facts worth a trace slot. The streams themselves are excluded: the
#: recorded output already holds them, and copying megabytes into every event
#: would bloat the trace for nothing.
_COMMAND_FACT_KEYS = ("exit_code", "running", "timed_out", "stdout_bytes",
                      "stderr_bytes", "stdout_truncated", "stderr_truncated")


def _command_facts(outcome) -> dict:
    return {k: getattr(outcome, k) for k in _COMMAND_FACT_KEYS}


def _output_truncated(output: str, outcome=None) -> bool:
    """Whether the runtime dropped part of this result.

    Prefers the reported fact; falls back to the marker `text_editor`/`grep`
    embed in their own output, which report no outcome.
    """
    if outcome is not None and outcome.truncated:
        return True
    return "[output truncated:" in output
