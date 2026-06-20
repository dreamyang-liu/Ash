"""Agent loop for SWE-bench.

Uses litellm for model abstraction (supports Claude, GPT-4, Gemini, etc.)
and executes tool calls through the ash sandbox SDK.

Tools: shell, text_editor, grep_files, read_file — called directly by name.
"""

import json
import sys
import time
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

_DEFAULT_SYSTEM_PROMPT = (_PROMPTS_DIR / "AGENT.md").read_text()

_DEFAULT_KICKOFF = (
    "Solve the issue described above. "
    "You have these tools available:\n"
    "- `grep_files`: Search code with ripgrep (use FIRST to locate relevant files)\n"
    "- `read_file`: Read file contents with line numbers (use offset/limit for large files)\n"
    "- `text_editor`: View/edit/create files (view, str_replace, insert, create)\n"
    "- `shell`: Run tests and commands (always use tail= to limit output)\n"
    "- `process`: Manage background processes (read output, kill)\n\n"
    "Start by reproducing the issue, then locate and read the relevant code."
)


def _get_system_info() -> dict:
    """Get system info for template variables."""
    import platform
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def _render_template(template: str, **kwargs) -> str:
    """Render a template with simple {{var}} substitution."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def build_system_prompt(task: str, config: Optional["AgentConfig"] = None) -> str:
    """Build the full system prompt, using template if provided."""
    if config and config.system_template:
        return _render_template(config.system_template, task=task, **_get_system_info())
    return _DEFAULT_SYSTEM_PROMPT + f"\n\n---\n\n## Task\n\n{task}"


def build_instance_message(task: str, config: Optional["AgentConfig"] = None) -> str:
    """Build the first user message (kickoff), using template if provided."""
    if config and config.instance_template:
        return _render_template(config.instance_template, task=task, **_get_system_info())
    return _DEFAULT_KICKOFF


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

    def _add_cache_breakpoints(self, messages: list[dict]) -> list[dict]:
        """Add cache_control breakpoints for Anthropic/Bedrock prompt caching.

        Strategy: mark system message and the second-to-last message as cache
        breakpoints. This caches the static system prompt and the entire
        conversation prefix (everything except the latest tool result).
        """
        if not messages or "anthropic" not in self.config.model and "bedrock" not in self.config.model:
            return messages

        msgs = [dict(m) for m in messages]

        # Mark system message (always cacheable — same across all calls)
        if msgs and msgs[0]["role"] == "system":
            content = msgs[0]["content"]
            if isinstance(content, str):
                msgs[0]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]

        # Mark the second-to-last message as cache breakpoint
        # (the prefix up to the last user/tool message)
        if len(msgs) >= 3:
            bp_idx = len(msgs) - 2
            content = msgs[bp_idx].get("content", "")
            if isinstance(content, str) and content:
                msgs[bp_idx]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list) and content:
                # Already a list of blocks — add cache_control to the last block
                msgs[bp_idx]["content"] = list(content)
                last_block = dict(msgs[bp_idx]["content"][-1])
                last_block["cache_control"] = {"type": "ephemeral"}
                msgs[bp_idx]["content"][-1] = last_block

        return msgs

    def _query_model(self, messages: list[dict]) -> Any:
        completion, stream_chunk_builder = _get_litellm()

        # Apply prompt caching if enabled
        cached_messages = self._add_cache_breakpoints(messages) if self.config.prompt_cache else messages

        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            messages=cached_messages,
            tools=self._tools_schema if self._tools_schema else None,
            tool_choice="auto" if self._tools_schema else None,
            max_tokens=self.config.max_tokens,
            stream=self.stream,
        )
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
            kwargs["allowed_openai_params"] = kwargs.get("allowed_openai_params", []) + ["reasoning_effort"]
            kwargs["drop_params"] = True  # let litellm skip unsupported params for other models
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        max_retries = 8
        for attempt in range(max_retries):
            try:
                raw = completion(**kwargs)
                break
            except Exception as e:
                err_type = type(e).__name__
                err_str = str(e).lower()
                retryable = (
                    "RateLimitError" in err_type
                    or "rate" in err_str
                    or "Timeout" in err_type
                    or "timed out" in err_str
                    or "timeout" in err_str
                    or "ServiceUnavailableError" in err_type
                    or "InternalServerError" in err_type
                )
                if retryable:
                    wait = min(2 ** attempt, 120)
                    self._trace(f"\n[RETRY] {err_type} attempt {attempt+1}/{max_retries}, waiting {wait}s\n")
                    time.sleep(wait)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise

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

            # Map bash -> shell for bash_only mode
            exec_name = "shell" if name == "bash" else name

            # Check guardrails before execution
            guardrail_warning = self._check_guardrails(exec_name, args)

            result = self.executor(exec_name, args)

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

            max_len = 12000
            if len(content) > max_len:
                head = max_len * 2 // 3  # ~8000 chars
                tail = max_len // 3      # ~4000 chars
                elided = len(content) - head - tail
                content = (
                    content[:head] +
                    f"\n\n... [{elided} characters truncated — output too long. "
                    f"Use `tail` on shell commands, `limit` on grep/read_file, "
                    f"or pipe through `grep` to get more targeted output] ...\n\n" +
                    content[-tail:]
                )

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

        system_msg = build_system_prompt(task, self.config)
        user_msg = build_instance_message(task, self.config)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        self.trajectory.add_message("system", system_msg)
        self.trajectory.add_message("user", user_msg)

        consecutive_no_tool = 0

        while True:
            # Check limits
            if self.cost.api_calls >= self.config.step_limit:
                self._close_trace()
                return "step_limit"
            if self.cost.total_cost >= self.config.cost_limit:
                self._close_trace()
                return "cost_limit"

            # Budget warning only when ~3-5 steps remaining (based on avg cost per step)
            if not self._budget_warned and self.cost.api_calls >= 3:
                avg_cost_per_step = self.cost.total_cost / self.cost.api_calls
                remaining_budget = self.config.cost_limit - self.cost.total_cost
                remaining_steps = self.config.step_limit - self.cost.api_calls
                est_steps_left = min(remaining_steps, int(remaining_budget / avg_cost_per_step)) if avg_cost_per_step > 0 else remaining_steps

                if est_steps_left <= 4 or remaining_steps <= 4:
                    self._budget_warned = True
                    budget_msg = (
                        f"\n\n[Budget Warning] ~{est_steps_left} steps remaining "
                        f"({self.cost.api_calls}/{self.config.step_limit} steps, "
                        f"${self.cost.total_cost:.2f}/${self.config.cost_limit:.2f} budget). "
                        f"Finalize your fix now: run tests and stop. If tests fail, revert and submit your best attempt."
                    )
                    for msg in reversed(messages):
                        if msg["role"] in ("tool", "user"):
                            msg["content"] += budget_msg
                            break
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
                if not (message.content or "").strip():
                    # Empty response — always reprompt
                    messages.append({"role": "user", "content": "You must call a tool to proceed."})
                    self._trace(f"\n[NUDGE] empty response, prompting retry\n")
                elif consecutive_no_tool >= 2:
                    self._close_trace()
                    return "completed"

        self._close_trace()
        return "completed"

    def _close_trace(self):
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None
