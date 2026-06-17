"""Agent loop for SWE-bench.

Uses litellm for model abstraction (supports Claude, GPT-4, Gemini, etc.)
and executes tool calls through the ash sandbox SDK.

Tools: shell, text_editor, grep_files, read_file — called directly by name.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

def _get_litellm():
    try:
        from litellm import completion, stream_chunk_builder
        return completion, stream_chunk_builder
    except ImportError:
        raise ImportError("Install litellm: pip install litellm")

from .models import AgentConfig, CostTracker, ToolResult, Trajectory

_PROMPTS_DIR = Path(__file__).parent

_SYSTEM_PROMPT = (_PROMPTS_DIR / "AGENT.md").read_text()

_KICKOFF = (
    "Solve the issue described above. "
    "You have these tools available:\n"
    "- `grep_files`: Search code with ripgrep (use FIRST to locate relevant files)\n"
    "- `read_file`: Read file contents with line numbers (use offset/limit for large files)\n"
    "- `text_editor`: View/edit/create files (view, str_replace, insert, create)\n"
    "- `shell`: Run tests and commands (always use tail= to limit output)\n"
    "- `process`: Manage background processes (read output, kill)\n\n"
    "Start by reproducing the issue, then locate and read the relevant code."
)


def build_system_prompt(task: str) -> str:
    """Build the full system prompt from AGENT.md + task."""
    return _SYSTEM_PROMPT + f"\n\n---\n\n## Task\n\n{task}"


def _tool_summary(name: str, args: dict) -> str:
    """Build a human-readable one-line summary for a tool call."""
    if name == "shell":
        cmd = args.get("command", "")
        bg = " &" if args.get("background") else ""
        return cmd + bg
    elif name == "grep_files":
        pat = args.get("pattern", "")
        path = args.get("path", "")
        inc = args.get("include", "")
        parts = [f"/{pat}/"]
        if path:
            parts.append(path)
        if inc:
            parts.append(f"({inc})")
        return " ".join(parts)
    elif name == "read_file":
        path = args.get("path", "")
        offset = args.get("offset")
        limit = args.get("limit")
        if offset or limit:
            return f"{path}:{offset or 1}+{limit or '?'}"
        return path
    elif name == "text_editor":
        cmd = args.get("command", "")
        path = args.get("path", "")
        if cmd == "str_replace":
            old = args.get("old_str", "")
            preview = old[:40].replace("\n", "\\n")
            return f"{path} [{cmd}] \"{preview}\""
        elif cmd == "view":
            vr = args.get("view_range")
            if vr:
                return f"{path} [{vr[0]}:{vr[1]}]"
            return f"{path} [view]"
        return f"{path} [{cmd}]"
    elif name == "process":
        pid = args.get("pid", "?")
        action = args.get("action", "?")
        return f"{pid} {action}"
    elif name == "web_fetch":
        return args.get("url", "")
    elif name == "web_search":
        return args.get("query", "")
    else:
        return args.get("command", "") or args.get("path", "") or str(args)[:80]


class _ThinkingLoopError(Exception):
    pass


class AshAgent:
    """Agent that uses tool calls to solve SWE-bench tasks."""

    def __init__(
        self,
        config: AgentConfig,
        executor: Callable[[str, dict], ToolResult],
        on_step: Optional[Callable[[int, str, str], None]] = None,
        trace_dir: Optional[Path] = None,
    ):
        self.config = config
        self.executor = executor  # executor(tool_name, args) -> ToolResult
        self.on_step = on_step    # on_step(step_num, kind, text)
        self.trace_dir = trace_dir
        self.trajectory = Trajectory()
        self.cost = CostTracker()
        self._tools_schema: list[dict] = []
        self._trace_file = None
        # Guardrail state
        self._files_read: set[str] = set()       # files read via read_file/text_editor view
        self._consecutive_edits: dict[str, int] = {}  # file -> consecutive edits without test
        self._ran_test_since_edit = True
        self._budget_warned = False
        self.stream = True  # set to False to disable streaming (parallel mode)

    def set_tools_schema(self, schema: list[dict]):
        """Set OpenAI-compatible tool schemas."""
        self._tools_schema = schema

    def _trace(self, text: str):
        """Write text to the trace file (real-time, flushed)."""
        if self._trace_file:
            self._trace_file.write(text)
            self._trace_file.flush()

    @staticmethod
    def _is_repeating(buf: str, window: int = 200, min_repeats: int = 3) -> bool:
        """Detect if the tail of buf is a repeating pattern."""
        tail = buf[-window * min_repeats:] if len(buf) >= window * min_repeats else ""
        if not tail:
            return False
        pattern = tail[-window:]
        count = tail.count(pattern)
        return count >= min_repeats

    def _query_model(self, messages: list[dict]) -> Any:
        completion, stream_chunk_builder = _get_litellm()

        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            tools=self._tools_schema if self._tools_schema else None,
            tool_choice="auto" if self._tools_schema else None,
            max_tokens=self.config.max_tokens,
            stream=self.stream,
        )
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.config.thinking_budget}
            kwargs["drop_params"] = True  # let litellm skip unsupported params for other models
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        raw = completion(**kwargs)

        if not self.stream:
            self.cost.update(raw)
            return raw

        # Stream mode: display thinking tokens as a rolling line
        from . import style as S
        chunks = []
        think_buf = ""
        content_buf = ""
        aborted = False
        w = 76  # display width for rolling line
        step_n = self.cost.api_calls + 1
        check_interval = 500

        self._trace(f"\n{'='*60}\n[step {step_n}] model call\n{'='*60}\n")

        for chunk in raw:
            chunks.append(chunk)
            delta = chunk.choices[0].delta

            # Reasoning / thinking tokens (Qwen3, DeepSeek, etc.)
            think_token = getattr(delta, "reasoning_content", None) or ""
            if think_token:
                if not think_buf:
                    self._trace("<think>\n")
                think_buf += think_token
                self._trace(think_token)
                vis = think_buf.replace("\n", " ")
                if len(vis) > w:
                    vis = "…" + vis[-(w - 1):]
                sys.stdout.write(f"\r  {S.dim(vis)}\033[K")
                sys.stdout.flush()

                # Detect thinking loop
                if len(think_buf) % check_interval < len(think_token):
                    if self._is_repeating(think_buf):
                        self._trace("\n[ABORTED: repetition loop detected]\n")
                        if self.on_step:
                            self.on_step(step_n, "error", "thinking loop detected, aborting")
                        aborted = True
                        break

            # Content tokens
            content_token = delta.content or ""
            if content_token:
                if think_buf and not content_buf:
                    self._trace("\n</think>\n\n")
                content_buf += content_token
                self._trace(content_token)

        # Clear the rolling line
        if think_buf:
            if not content_buf:
                self._trace("\n</think>\n")
            sys.stdout.write(f"\r\033[K")
            sys.stdout.flush()

        if aborted:
            raise _ThinkingLoopError("model stuck in thinking loop")

        response = stream_chunk_builder(chunks)
        self.cost.update(response)
        return response

    def _check_guardrails(self, name: str, args: dict) -> str:
        """Check tool-level guardrails. Returns warning string or empty."""
        warnings = []
        path = args.get("path", "")

        # Track file reads
        if name == "read_file" or (name == "text_editor" and args.get("command") == "view"):
            self._files_read.add(path)

        # Read-before-edit: must have read a file before editing it
        if name == "text_editor" and args.get("command") in ("str_replace", "insert"):
            if path and path not in self._files_read:
                warnings.append(
                    f"[Warning] You are editing {path} without reading it first. "
                    f"Use read_file or text_editor(view) first to see the current content."
                )
            # Track consecutive edits without testing
            self._consecutive_edits[path] = self._consecutive_edits.get(path, 0) + 1
            self._ran_test_since_edit = False
            if self._consecutive_edits[path] >= 3:
                warnings.append(
                    f"[Warning] This is edit #{self._consecutive_edits[path]} to {path} "
                    f"without running tests. Consider testing before making more changes."
                )

        # Reset edit counter when running tests
        if name == "shell":
            cmd = args.get("command", "")
            if "pytest" in cmd or "test_" in cmd or "assert" in cmd:
                self._consecutive_edits.clear()
                self._ran_test_since_edit = True

        return "\n".join(warnings)

    def _execute_tool_calls(self, tool_calls: list) -> list[dict]:
        """Execute tool calls and return tool response messages."""
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Check guardrails before execution
            guardrail_warning = self._check_guardrails(name, args)

            result = self.executor(name, args)

            # Format content
            if result.success:
                content = result.output
            else:
                content = f"Error: {result.error or 'Unknown error'}"
                if result.output:
                    content += f"\n{result.output}"

            # Append guardrail warnings
            if guardrail_warning:
                content += f"\n\n{guardrail_warning}"

            max_len = 15000
            if len(content) > max_len:
                content = content[:max_len] + f"\n... (truncated {len(content) - max_len} chars)"

            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

            self.trajectory.add_message(
                "tool_result",
                content,
                tool_name=name,
                tool_args=args,
                success=result.success,
            )

        return results

    def run(self, task: str, instance_id: str = "") -> str:
        """Run the agent loop. Returns exit status.

        Exit statuses:
        - "completed"   -- agent stopped making tool calls
        - "step_limit"  -- hit step limit
        - "cost_limit"  -- hit cost limit
        - "error"       -- unrecoverable error
        """
        self.trajectory = Trajectory()
        self.trajectory.instance_id = instance_id
        self.cost = CostTracker()

        # Open trace file for real-time logging
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self.trace_dir / f"{instance_id or 'trace'}.log"
            self._trace_file = open(trace_path, "w", encoding="utf-8")

        system_msg = build_system_prompt(task)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": _KICKOFF},
        ]
        self.trajectory.add_message("system", system_msg)
        self.trajectory.add_message("user", task)

        consecutive_no_tool = 0

        while True:
            # Check limits
            if self.cost.api_calls >= self.config.step_limit:
                self._close_trace()
                return "step_limit"
            if self.cost.total_cost >= self.config.cost_limit:
                self._close_trace()
                return "cost_limit"

            # Budget warning at 75%
            if not self._budget_warned:
                cost_pct = self.cost.total_cost / self.config.cost_limit if self.config.cost_limit else 0
                step_pct = self.cost.api_calls / self.config.step_limit if self.config.step_limit else 0
                if cost_pct >= 0.75 or step_pct >= 0.75:
                    self._budget_warned = True
                    budget_msg = (
                        f"[Budget Warning] You have used {self.cost.api_calls}/{self.config.step_limit} steps "
                        f"and ${self.cost.total_cost:.2f}/${self.config.cost_limit:.2f} budget. "
                        f"Wrap up: test your current fix and stop. Do not start new approaches."
                    )
                    messages.append({"role": "user", "content": budget_msg})
                    self._trace(f"\n{budget_msg}\n")

            step_n = self.cost.api_calls + 1

            try:
                response = self._query_model(messages)
            except _ThinkingLoopError:
                self._trace(f"\n[RETRY] thinking loop, retrying with temperature bump\n")
                try:
                    old_temp = self.config.temperature
                    self.config.temperature = max(old_temp + 0.3, 0.6)
                    response = self._query_model(messages)
                    self.config.temperature = old_temp
                except _ThinkingLoopError:
                    self.config.temperature = old_temp
                    self._trace(f"\n[ABORT] repeated thinking loop\n")
                    self._close_trace()
                    self.trajectory.add_message("error", "repeated thinking loop")
                    return "error"
                except Exception as e:
                    self.config.temperature = old_temp
                    if self.on_step:
                        self.on_step(step_n, "error", str(e))
                    self._trace(f"\n[ERROR] {e}\n")
                    self._close_trace()
                    self.trajectory.add_message("error", str(e))
                    return "error"
            except Exception as e:
                if self.on_step:
                    self.on_step(step_n, "error", str(e))
                self._trace(f"\n[ERROR] {e}\n")
                self._close_trace()
                self.trajectory.add_message("error", str(e))
                return "error"

            choice = response.choices[0]
            message = choice.message

            # Record assistant message
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_msg["tool_calls"] = message.tool_calls
            # Preserve thinking_blocks for Anthropic extended thinking + tool use
            thinking_blocks = getattr(message, "thinking_blocks", None)
            if thinking_blocks:
                assistant_msg["thinking_blocks"] = thinking_blocks
            messages.append(assistant_msg)
            self.trajectory.add_message("assistant", message.content or "")

            # Execute tool calls if present
            if message.tool_calls:
                consecutive_no_tool = 0
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        args = {}
                    label = tc.function.name
                    summary = _tool_summary(label, args)
                    if self.on_step:
                        self.on_step(step_n, label, summary)
                    self._trace(f"\n> {label} {summary}\n")
                observations = self._execute_tool_calls(message.tool_calls)
                for obs in observations:
                    self._trace(f"{obs['content']}\n")
                messages.extend(observations)
            else:
                consecutive_no_tool += 1
                if consecutive_no_tool >= 2 or choice.finish_reason == "stop":
                    self._close_trace()
                    return "completed"

        self._close_trace()
        return "completed"

    def _close_trace(self):
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None
