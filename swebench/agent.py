"""Agent loop for SWE-bench.

Uses litellm for model abstraction (supports Claude, GPT-4, Gemini, etc.)
and runs commands through the ash CLI session via subprocess.

Single tool: bash — agent writes ash CLI commands directly.
Session routing handled by ASH_SESSION env var.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from litellm import completion, stream_chunk_builder
except ImportError:
    raise ImportError("Install litellm: pip install litellm")

from .models import AgentConfig, CostTracker, ToolResult, Trajectory

_PROMPTS_DIR = Path(__file__).parent

_SYSTEM_PROMPT = (_PROMPTS_DIR / "AGENT.md").read_text()

_KICKOFF = (
    "Solve the issue described above. "
    "You have one tool: `bash`. Use `ash` CLI commands for all operations:\n"
    "- `ash grep \"pattern\" path/` to search code\n"
    "- `ash edit view file.py` to read files\n"
    "- `ash outline file.py` to see code structure\n"
    "- `ash find \"*.py\" path/` to find files\n"
    "- `ash edit replace file.py --old \"...\" --new \"...\"` to edit\n"
    "- `ash run \"command\"` for python, pytest, pip, git, etc.\n"
    "- `ash buffer` for scratch pad across steps\n"
    "Run `ash --help` to see all available commands.\n"
    "Commands are composable: `ash grep ... && ash edit view ...`\n"
    "All commands run in /testbed by default.\n"
    "Start by exploring the repository to understand the issue."
)


def build_system_prompt(task: str) -> str:
    """Build the full system prompt from AGENT.md + task."""
    return _SYSTEM_PROMPT + f"\n\n---\n\n## Task\n\n{task}"


class _ThinkingLoopError(Exception):
    pass


class AshAgent:
    """Agent that uses tool calls to solve SWE-bench tasks."""

    def __init__(
        self,
        config: AgentConfig,
        executor: Callable[[str], ToolResult],
        on_step: Optional[Callable[[int, str, str], None]] = None,
        trace_dir: Optional[Path] = None,
    ):
        self.config = config
        self.executor = executor  # executor(command) -> ToolResult
        self.on_step = on_step    # on_step(step_num, kind, text)
        self.trace_dir = trace_dir
        self.trajectory = Trajectory()
        self.cost = CostTracker()
        self._tools_schema: list[dict] = []
        self._trace_file = None

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
        # Check if the last `window` chars repeat earlier in the tail
        pattern = tail[-window:]
        count = tail.count(pattern)
        return count >= min_repeats

    def _query_model(self, messages: list[dict]) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            tools=self._tools_schema if self._tools_schema else None,
            tool_choice="auto" if self._tools_schema else None,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=bool(self.on_step),
        )
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        raw = completion(**kwargs)

        if not self.on_step:
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
        check_interval = 500  # check for repetition every N thinking tokens

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
                # Show the tail of thinking buffer, overwriting the line
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

    def _execute_tool_calls(self, tool_calls: list) -> list[dict]:
        """Execute tool calls and return tool response messages."""
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            command = args.get("command", "")
            result = self.executor(command)

            # Format content
            if result.success:
                content = result.output
            else:
                content = f"Error: {result.error or 'Unknown error'}"
                if result.output:
                    content += f"\n{result.output}"

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
        kickoff = _KICKOFF
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": kickoff},
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

            step_n = self.cost.api_calls + 1

            try:
                response = self._query_model(messages)
            except _ThinkingLoopError:
                # Model got stuck in a thinking loop — retry with higher temperature
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
            messages.append(assistant_msg)
            self.trajectory.add_message("assistant", message.content or "")

            # Execute tool calls if present
            if message.tool_calls:
                consecutive_no_tool = 0
                for tc in message.tool_calls:
                    try:
                        cmd = json.loads(tc.function.arguments).get("command", "")
                    except (json.JSONDecodeError, AttributeError):
                        cmd = tc.function.arguments or ""
                    tool_label = tc.function.name or self.mode
                    if self.on_step:
                        self.on_step(step_n, tool_label, cmd)
                    self._trace(f"\n> {tool_label} {cmd}\n")
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
