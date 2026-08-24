"""Keep a marathon-length conversation inside the model's context window.

The conversation is append-only, which is fine for the 20-60 step episodes
this harness grew up on and eventually fatal: once the window fills, the next
model call is an API error that ends the run with its budget unspent.

"Eventually" is the whole design problem, because the horizon depends on the
model, not on the harness. Measured on one real 133-step SWE-Marathon attempt:
~139K input tokens on the last call -- ruinous against a 200K window, 14% of a
1M one. So neither the budget nor the measurement may be guessed; both come
from the provider.

- **Budget** is a fraction of the model's own ``max_input_tokens`` (litellm's
  model metadata), so the same code is conservative on a small window and
  permissive on a large one.
- **Measurement** is ``litellm.token_counter``, which is exact (measured: 3027
  against a real 3026) and counts tool calls -- and tool calls carry the bulk
  on code-heavy runs, where one ``text_editor`` write holds a whole file.
  Character estimates came out 2-3x off in *both* directions depending on
  content, which is precisely the error a window guard cannot afford.

What it does when over budget is elide, not summarize: old *tool outputs*
become one-line stubs while every assistant turn -- the reasoning and the
commands it ran -- stays verbatim. What the agent did is cheap and
irreplaceable; what it saw is bulky and mostly re-obtainable. No extra model
call, no summary that can lie.

Elision runs in bulk and rarely rather than as a sliding window: prompt
caching prices the transcript by its stable prefix, so rewriting one old
message per step would invalidate the cache every step. Cutting once, deeply,
leaves long cacheable plateaus.

The trajectory keeps the full text: elision is about what the MODEL sees next,
not about what the run records.
"""

from __future__ import annotations

#: Window fractions: elide when the transcript passes `budget`, cut down to
#: `target`. The gap is what buys cacheable plateaus between cuts.
DEFAULT_BUDGET_FRACTION = 0.70
DEFAULT_TARGET_FRACTION = 0.45

#: Assumed window when the model's metadata is unknown. Small on purpose: an
#: unknown model that is actually larger only gets earlier elision, while the
#: reverse would be a dead run.
FALLBACK_WINDOW_TOKENS = 200_000

#: Never elide the most recent tool outputs: the model is usually acting on
#: them right now. This is also a floor on what elision can achieve -- the
#: protected tail is untouchable, so a target below what it costs is
#: unreachable by construction.
_KEEP_RECENT_RESULTS = 20

#: Densest plausible characters-per-token, used only to skip the exact count
#: when even this assumption stays under budget. Being wrong the other way
#: costs one tokenizer pass (~0.6s on a 4.6M-char transcript), so the cheap
#: gate is deliberately pessimistic.
_DENSEST_CHARS_PER_TOKEN = 1.5

_STUB = ("[tool output elided to fit the context window -- {chars} chars. "
         "The state it described may be stale; re-run the command if it "
         "matters now.]")


def transcript_chars(messages: list[dict]) -> int:
    """Characters the model-facing transcript spends, tool calls included."""
    total = 0
    for message in messages:
        total += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or ():
            total += len(str(call))
    return total


def model_window_tokens(model: str | None) -> int:
    """The model's input window, or a conservative default."""
    if not model:
        return FALLBACK_WINDOW_TOKENS
    try:
        import litellm
        info = litellm.get_model_info(model) or {}
        window = info.get("max_input_tokens") or info.get("max_tokens")
        return int(window) if window else FALLBACK_WINDOW_TOKENS
    except Exception:
        # Unknown model, offline metadata, litellm absent: assume small.
        return FALLBACK_WINDOW_TOKENS


def count_tokens(messages: list[dict], model: str | None,
                 tools: list[dict] | None = None) -> int:
    """Exact token count when the provider's tokenizer is reachable.

    Falls back to a character estimate only when it is not; that estimate is
    known unreliable, which is why it is never the primary path.
    """
    if model:
        try:
            import litellm
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            return int(litellm.token_counter(**kwargs))
        except Exception:
            pass
    return int(transcript_chars(messages) / 3)


def make_context_window_guard(budget_fraction: float = DEFAULT_BUDGET_FRACTION,
                              target_fraction: float = DEFAULT_TARGET_FRACTION,
                              keep_recent: int = _KEEP_RECENT_RESULTS,
                              window_tokens: int | None = None):
    """A ``before_query`` hook that keeps the transcript inside the window.

    ``window_tokens`` overrides the model's declared window; leave it unset to
    read the model's own metadata. ``target_fraction`` is a target rather than
    a guarantee -- the ``keep_recent`` newest tool outputs are never elided,
    so a target below what that tail costs cannot be reached, and the guard
    says so instead of appearing to succeed.
    """
    def context_window_guard(agent, conv) -> None:
        model = getattr(getattr(agent, "config", None), "model", None)
        window = window_tokens or model_window_tokens(model)
        budget = int(window * budget_fraction)
        target = int(window * target_fraction)
        tools = getattr(agent, "tools_schema", None)

        # Cheap gate: if even the densest plausible encoding stays under
        # budget, no tokenizer pass is needed. Keeps the common case free.
        if transcript_chars(conv.messages) / _DENSEST_CHARS_PER_TOKEN <= budget:
            return
        if count_tokens(conv.messages, model, tools) <= budget:
            return

        tool_indexes = [i for i, m in enumerate(conv.messages)
                        if m.get("role") == "tool"]
        elidable = (tool_indexes[:-keep_recent]
                    if keep_recent and len(tool_indexes) > keep_recent
                    else tool_indexes if not keep_recent else [])

        elided = 0
        for index in elidable:
            message = conv.messages[index]
            content = str(message.get("content") or "")
            if content.startswith("[tool output elided"):
                continue
            message["content"] = _STUB.format(chars=len(content))
            elided += 1
            # Re-count every few elisions rather than every one: counting is a
            # tokenizer pass, and the point is to cut in bulk anyway.
            if elided % 5 == 0 and count_tokens(
                    conv.messages, model, tools) <= target:
                break

        trace = getattr(agent, "_trace", None)
        if not trace:
            return
        remaining = count_tokens(conv.messages, model, tools)
        if elided:
            trace(f"\n[context window: elided {elided} old tool outputs; "
                  f"~{remaining} of {window} tokens used]\n")
        if remaining > target:
            # Everything eligible is already gone: what is left is the
            # protected tail plus the assistant turns.
            trace(f"\n[context window: still ~{remaining} tokens after "
                  f"eliding everything eligible; the {keep_recent} protected "
                  f"recent outputs dominate. Lower keep_recent if this "
                  f"repeats.]\n")

    return context_window_guard
