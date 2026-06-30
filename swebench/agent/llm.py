"""LLM client: litellm wrapper handling caching, streaming, retries, and
thinking-loop detection. Keeps all model-call concerns out of the agent loop.
"""

import sys
import time
from typing import Any, Callable, Optional

from ..models import AgentConfig, CostTracker


def _get_litellm():
    try:
        from litellm import completion, stream_chunk_builder
        return completion, stream_chunk_builder
    except ImportError:
        raise ImportError("Install litellm: pip install litellm")


class ThinkingLoopError(Exception):
    """Raised when the model is detected repeating itself in its thinking."""


def _is_repeating(buf: str, window: int = 200, min_repeats: int = 3) -> bool:
    """Detect if the tail of buf is a repeating pattern."""
    tail = buf[-window * min_repeats:] if len(buf) >= window * min_repeats else ""
    if not tail:
        return False
    pattern = tail[-window:]
    return tail.count(pattern) >= min_repeats


class LLMClient:
    """Wraps litellm.completion with the project's call-time behavior."""

    def __init__(
        self,
        config: AgentConfig,
        cost: CostTracker,
        tools_schema: Optional[list[dict]] = None,
        trace: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[int, str, str], None]] = None,
    ):
        self.config = config
        self.cost = cost
        self.tools_schema = tools_schema or []
        self._trace = trace or (lambda _: None)
        self.on_step = on_step
        self.stream = True

    # -- prompt caching -----------------------------------------------------

    def _add_cache_breakpoints(self, messages: list[dict]) -> list[dict]:
        """Add cache_control breakpoints for Anthropic/Bedrock prompt caching.

        Marks the system message and the second-to-last message, caching the
        static system prompt and the conversation prefix (all but the latest
        tool result).
        """
        if not messages or ("anthropic" not in self.config.model and "bedrock" not in self.config.model):
            return messages

        msgs = [dict(m) for m in messages]

        if msgs and msgs[0]["role"] == "system":
            content = msgs[0]["content"]
            if isinstance(content, str):
                msgs[0]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]

        if len(msgs) >= 3:
            bp_idx = len(msgs) - 2
            content = msgs[bp_idx].get("content", "")
            if isinstance(content, str) and content:
                msgs[bp_idx]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list) and content:
                msgs[bp_idx]["content"] = list(content)
                last_block = dict(msgs[bp_idx]["content"][-1])
                last_block["cache_control"] = {"type": "ephemeral"}
                msgs[bp_idx]["content"][-1] = last_block

        return msgs

    # -- model call ---------------------------------------------------------

    def _build_kwargs(self, messages: list[dict]) -> dict:
        c = self.config
        cached = self._add_cache_breakpoints(messages) if c.prompt_cache else messages
        kwargs: dict[str, Any] = dict(
            model=c.model,
            messages=cached,
            tools=self.tools_schema or None,
            tool_choice="auto" if self.tools_schema else None,
            max_tokens=c.max_tokens,
            stream=self.stream,
        )
        if c.temperature is not None:
            kwargs["temperature"] = c.temperature
        if c.reasoning_effort:
            kwargs["reasoning_effort"] = c.reasoning_effort
            kwargs["allowed_openai_params"] = ["reasoning_effort"]
            kwargs["drop_params"] = True
        if c.api_base:
            kwargs["api_base"] = c.api_base
        if c.api_key:
            kwargs["api_key"] = c.api_key
        return kwargs

    def query(self, messages: list[dict]) -> Any:
        """Run one completion, with retry and (in stream mode) loop detection."""
        completion, stream_chunk_builder = _get_litellm()
        kwargs = self._build_kwargs(messages)

        max_retries = 8
        raw = None
        for attempt in range(max_retries):
            try:
                raw = completion(**kwargs)
                break
            except Exception as e:
                if not self._retryable(e):
                    raise
                wait = min(2 ** attempt, 120)
                self._trace(f"\n[RETRY] {type(e).__name__} attempt {attempt+1}/{max_retries}, waiting {wait}s\n")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise

        if not self.stream:
            self.cost.update(raw)
            return raw
        return self._consume_stream(raw, stream_chunk_builder)

    @staticmethod
    def _retryable(e: Exception) -> bool:
        err_type = type(e).__name__
        err_str = str(e).lower()
        return (
            "RateLimitError" in err_type
            or "rate" in err_str
            or "Timeout" in err_type
            or "timed out" in err_str
            or "timeout" in err_str
            or "ServiceUnavailableError" in err_type
            or "InternalServerError" in err_type
        )

    def _consume_stream(self, raw, stream_chunk_builder) -> Any:
        """Display thinking tokens as a rolling line, detect loops, build response."""
        from .. import style as S
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

                if len(think_buf) % check_interval < len(think_token):
                    if _is_repeating(think_buf):
                        self._trace("\n[ABORTED: repetition loop detected]\n")
                        if self.on_step:
                            self.on_step(step_n, "error", "thinking loop detected, aborting")
                        aborted = True
                        break

            content_token = delta.content or ""
            if content_token:
                if think_buf and not content_buf:
                    self._trace("\n</think>\n\n")
                content_buf += content_token
                self._trace(content_token)

        if think_buf:
            if not content_buf:
                self._trace("\n</think>\n")
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        if aborted:
            raise ThinkingLoopError("model stuck in thinking loop")

        response = stream_chunk_builder(chunks)
        self.cost.update(response)
        return response

    def query_with_recovery(self, messages: list[dict]) -> Any:
        """query(), retrying once with a temperature bump on a thinking loop."""
        try:
            return self.query(messages)
        except ThinkingLoopError:
            self._trace("\n[RETRY] thinking loop, retrying with temperature bump\n")
            old_temp = self.config.temperature
            self.config.temperature = max((old_temp or 0.0) + 0.3, 0.6)
            try:
                return self.query(messages)
            finally:
                self.config.temperature = old_temp
