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


def _with_stall_timeout(stream, seconds: float, trace):
    """Yield from a stream, raising if it goes quiet for too long.

    The request timeout does not cover this. It bounds getting the response
    started; once a streaming response is open, iterating it can block
    forever, and does: a provider that stops sending leaves the socket in
    CLOSE-WAIT and the iteration waiting. Measured twice on this stack --
    2h48m of silence on one run, 20 minutes on another with a 900s request
    timeout set and no retry, because the timeout was never in play.

    Implemented with a worker thread rather than a signal so it works off the
    main thread, and daemon so a truly wedged read cannot keep the process
    alive.
    """
    import queue
    import threading

    items: "queue.Queue" = queue.Queue(maxsize=1)
    DONE = object()

    def pump():
        try:
            for item in stream:
                items.put(item)
            items.put(DONE)
        except BaseException as error:      # noqa: BLE001 - forwarded below
            items.put(error)

    threading.Thread(target=pump, daemon=True, name="llm-stream").start()
    while True:
        try:
            item = items.get(timeout=seconds)
        except queue.Empty:
            trace(f"\n[STALL] stream produced nothing for {seconds:.0f}s\n")
            raise TimeoutError(
                f"model stream stalled for {seconds:.0f}s") from None
        if item is DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


#: Ceiling on a single completion. Long enough for a model writing a whole
#: file with extended thinking, short enough that a silently dead connection
#: costs one retry instead of the rest of the run.
REQUEST_TIMEOUT_SECONDS = 900

#: How many times to ask again for a completion that came back with neither
#: text nor a tool call. Few, because a model that keeps returning nothing is
#: telling you something the loop should handle rather than something a retry
#: will fix.
EMPTY_RETRIES = 3

#: Seconds a streaming response may go silent before it is abandoned. The
#: request timeout cannot cover this; see `_with_stall_timeout`.
STREAM_STALL_TIMEOUT_SECONDS = 180


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
        #: Seconds one completion may take before it is treated as failed and
        #: retried. A ceiling, not a target: the point is that there IS one.
        self.request_timeout = REQUEST_TIMEOUT_SECONDS
        self._takes_cache_markers: "bool | None" = None
        #: How long a streaming response may produce nothing before it is
        #: treated as failed. Much shorter than the request timeout: this is
        #: the gap *between* chunks, and a model that is working sends
        #: something continuously.
        self.stream_stall_timeout = STREAM_STALL_TIMEOUT_SECONDS

    # -- prompt caching -----------------------------------------------------

    def _model_takes_cache_markers(self) -> bool:
        """Whether this model accepts explicit cache_control markers.

        The markers are an Anthropic Messages API concept. Bedrock translates
        them to cachePoint blocks, which its non-Anthropic models reject as a
        hard error -- "your request did not allow prompt caching" killed a run
        on its first call. So: models whose metadata says they cache, yes;
        models litellm does not know, only if they are addressed over the
        Anthropic protocol (an `anthropic/` route), where the marker is legal
        for the server to ignore. Everything else caches implicitly or not at
        all, and gets no markers.

        Cached per client: the answer cannot change mid-run, and metadata
        lookups are not free at one per step.
        """
        if self._takes_cache_markers is None:
            supports = None
            try:
                import litellm
                info = litellm.get_model_info(self.config.model) or {}
                supports = info.get("supports_prompt_caching")
            except Exception:
                supports = None
            if supports is None:
                supports = self.config.model.startswith("anthropic/")
            self._takes_cache_markers = bool(supports)
        return self._takes_cache_markers

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
        cached = (self._add_cache_breakpoints(messages)
                  if c.prompt_cache and self._model_takes_cache_markers()
                  else messages)
        kwargs: dict[str, Any] = dict(
            model=c.model,
            messages=cached,
            tools=self.tools_schema or None,
            tool_choice="auto" if self.tools_schema else None,
            max_tokens=c.max_tokens,
            stream=self.stream,
            # Without this the retry loop below is unreachable for the failure
            # that matters most: a provider that stops answering without
            # closing cleanly leaves the socket in CLOSE-WAIT and the process
            # in epoll forever. Measured on a 5-hour marathon run -- 2h48m of
            # silence at step 304, zero CPU, no retry, no error, and the
            # remaining budget lost. Generous rather than tight, because a
            # legitimate call on this path writes whole files.
            timeout=self.request_timeout,
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
        consumed = False
        for attempt in range(max_retries):
            try:
                raw = completion(**kwargs)
                if self.stream:
                    # Assembled first: a stream's emptiness is only visible
                    # once its chunks are joined.
                    raw = self._consume_stream(raw, stream_chunk_builder)
                    consumed = True
                if self._is_empty_completion(raw) and attempt < EMPTY_RETRIES:
                    # A completion with neither text nor a tool call is worth
                    # asking for again rather than handing upstream: the loop
                    # can only re-prompt, which spends a turn and puts a
                    # meaningless assistant message in the transcript -- and
                    # some providers replace that empty message with a
                    # placeholder on the next request, which then reads like
                    # the model talking. Measured cause on one proxy: extended
                    # thinking consumed the whole output budget, so the reply
                    # carried an empty thinking block and no text.
                    self._trace(f"\n[RETRY] empty completion, attempt "
                                f"{attempt + 1}/{EMPTY_RETRIES}\n")
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
            except Exception as e:
                if not self._retryable(e):
                    raise
                wait = min(2 ** attempt, 120)
                self._trace(f"\n[RETRY] {type(e).__name__} attempt {attempt+1}/{max_retries}, waiting {wait}s\n")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise

        if not consumed:
            self.cost.update(raw)
        return raw

    @staticmethod
    def _is_empty_completion(raw: Any) -> bool:
        """Whether a completion carries nothing to act on.

        Streaming responses are not inspected here -- they are consumed later
        and checked once assembled.
        """
        try:
            message = raw.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return False
        if getattr(message, "tool_calls", None):
            return False
        return not (getattr(message, "content", None) or "").strip()

    @staticmethod
    def _retryable(e: Exception) -> bool:
        """Retry unless the failure is clearly permanent.

        This used to be an allowlist of retryable type names, and it kept
        losing to the provider's taxonomy: a 103-step run died because a
        Bedrock 5xx arrived wrapped as MidStreamFallbackError, which no
        allowlist had heard of. The asymmetry decides the policy -- wasting
        eight backoffs (~4 minutes) on a permanent error is nothing against
        losing hours of run to a transient one -- so unknown failures retry,
        and only the provably-permanent classes fail fast.
        """
        err_type = type(e).__name__
        err_str = str(e).lower()
        permanent_types = (
            "AuthenticationError", "PermissionDeniedError", "NotFoundError",
            "BadRequestError", "InvalidRequestError",
            "UnprocessableEntityError", "ContextWindowExceededError",
            "ContentPolicyViolationError", "BudgetExceededError",
        )
        if any(t in err_type for t in permanent_types):
            return False
        # Some gateways misclassify configuration rejections as connection
        # errors; the message still says so.
        permanent_markers = (
            "unsupported model", "did not allow", "invalid api key",
            "authentication", "model not found",
        )
        return not any(m in err_str for m in permanent_markers)

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

        for chunk in _with_stall_timeout(raw, self.stream_stall_timeout,
                                        self._trace):
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
