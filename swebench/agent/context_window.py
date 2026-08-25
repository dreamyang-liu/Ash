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

What it does when over budget is a choice of two strategies, and which one is
right depends on the workload rather than on taste:

- ``elide`` (default) replaces old *tool outputs* with a one-line stub while
  every assistant turn -- the reasoning and the commands it ran -- stays
  verbatim. What the agent did is cheap and irreplaceable; what it saw is
  bulky and mostly re-obtainable. Costs nothing and cannot invent anything.
- ``summarize`` asks the model to compress the folded span into a paragraph of
  findings first, keeping conclusions that would otherwise be lost when only
  the commands survive. It costs one extra model call per firing and can
  compress wrongly -- a summary saying "tests pass" over output that showed
  three failures is not detectable downstream, which is why it is opt-in.

Measured against one real 133-step transcript (48.5K tokens, folding 125-128
old outputs down to 23%): elision took 0.2s and nothing, the summary took
17.7s and $0.09, and what the money bought was the kind of fact the agent
would otherwise re-derive -- the exact build command, that `xxd` and `python3`
are absent from the image, the table of expected test hashes. Both reached the
same token target, so the choice is about what the next hour of the run needs,
not about size.

Both keep assistant turns intact; they differ only in what replaces the old
outputs.

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

#: Prefix shared by every replacement, so a second pass can tell what it has
#: already folded and leave the prompt cache alone.
_FOLD_MARKER = "[tool output"

_STUB = ("[tool output elided to fit the context window -- {chars} chars. "
         "The state it described may be stale; re-run the command if it "
         "matters now.]")

_SUMMARY_STUB = (
    "[tool output summarized to fit the context window -- {count} earlier "
    "outputs. What they established:\n{summary}\n"
    "Details are gone; re-run a command if you need its exact output.]")


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


def _elide(agent, conv, indexes: list[int], model, tools, target: int) -> int:
    """Replace each old tool output with a stub. Returns how many were folded."""
    folded = 0
    for index in indexes:
        message = conv.messages[index]
        content = str(message.get("content") or "")
        if content.startswith(_FOLD_MARKER):
            continue
        message["content"] = _STUB.format(chars=len(content))
        folded += 1
        # Re-count every few rather than every one: counting is a tokenizer
        # pass, and the point is to cut in bulk anyway.
        if folded % 5 == 0 and count_tokens(conv.messages, model, tools) <= target:
            break
    return folded


def _summarize(agent, conv, indexes: list[int], model, tools, target: int) -> int:
    """Replace a span of old tool outputs with one model-written summary.

    Every folded message is replaced: the first carries the summary and the
    rest carry the plain stub, because a tool message may not simply vanish --
    the provider requires one per tool call in the preceding assistant turn.

    A failed or empty summary falls back to plain elision. Being unable to
    summarize must not become being unable to stay inside the window.
    """
    spans = [i for i in indexes
             if not str(conv.messages[i].get("content") or "").startswith(_FOLD_MARKER)]
    if not spans:
        return 0

    excerpts = []
    for index in spans:
        content = str(conv.messages[index].get("content") or "")
        # Head and tail: a command's verdict tends to sit at one end or the
        # other, and the middle of a 12K dump is rarely where the answer is.
        excerpts.append(content[:1500] + ("\n...\n" + content[-1500:]
                                          if len(content) > 3000 else ""))
    summary = _ask_for_summary(agent, model, excerpts)
    if not summary:
        return _elide(agent, conv, indexes, model, tools, target)

    conv.messages[spans[0]]["content"] = _SUMMARY_STUB.format(
        count=len(spans), summary=summary.strip())
    for index in spans[1:]:
        content = str(conv.messages[index].get("content") or "")
        conv.messages[index]["content"] = _STUB.format(chars=len(content))
    return len(spans)


def _ask_for_summary(agent, model, excerpts: list[str]) -> str:
    """One model call that compresses tool output into findings."""
    if not model:
        return ""
    prompt = ("Below are outputs from tool calls an agent made earlier, in "
              "order. They are about to be dropped from its context to make "
              "room. Write a compact factual summary of what they established "
              "-- results, errors, file and symbol names, numbers -- that the "
              "agent would otherwise have to re-discover. No advice, no "
              "restating the commands, under 300 words.\n\n"
              + "\n\n--- output ---\n".join(excerpts))
    try:
        import litellm
        response = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=600)
        text = response.choices[0].message.content or ""
        cost = getattr(agent, "cost", None)
        if cost is not None and hasattr(cost, "update"):
            # The summary is spent from the run's budget like any other call;
            # hiding it would make a run's cost unaccountable.
            cost.update(response)
        return text
    except Exception:
        return ""


#: Strategies by name, for configuration to select between.
STRATEGIES = {"elide": _elide, "summarize": _summarize}


def make_context_window_guard(strategy: str = "elide",
                              window_tokens: int | None = None,
                              budget_fraction: float = DEFAULT_BUDGET_FRACTION,
                              target_fraction: float = DEFAULT_TARGET_FRACTION,
                              keep_recent: int = _KEEP_RECENT_RESULTS):
    """A ``before_query`` hook that keeps the transcript inside the window.

    ``strategy`` is ``elide`` (free, cannot invent) or ``summarize`` (one extra
    model call per firing, keeps conclusions, can compress wrongly).
    ``window_tokens`` overrides the model's declared window, which is required
    for a model litellm has no metadata for -- a proxy-served model falls back
    to the conservative default and would be folded far earlier than it needs.
    Note that a large window is what a model *accepts*, not what it should be
    filled with: measured on one 1M-context proxy, latency grew ~13s per 100K
    tokens of input (29s at 200K, 137s at 1M), so the budget fraction is a
    speed and cost decision as much as a capacity one. ``target_fraction`` is a target rather than
    a guarantee -- the ``keep_recent`` newest tool outputs are never folded, so
    a target below what that tail costs cannot be reached, and the guard says
    so instead of appearing to succeed.
    """
    fold = STRATEGIES.get(strategy)
    if fold is None:
        raise ValueError(
            f"unknown context strategy {strategy!r}; "
            f"choose from {sorted(STRATEGIES)}")
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
        foldable = (tool_indexes[:-keep_recent]
                    if keep_recent and len(tool_indexes) > keep_recent
                    else tool_indexes if not keep_recent else [])

        folded = fold(agent, conv, foldable, model, tools, target)

        trace = getattr(agent, "_trace", None)
        if not trace:
            return
        remaining = count_tokens(conv.messages, model, tools)
        if folded:
            trace(f"\n[context window: {strategy}d {folded} old tool outputs; "
                  f"~{remaining} of {window} tokens used]\n")
        if remaining > target:
            # Everything eligible is already gone: what is left is the
            # protected tail plus the assistant turns.
            trace(f"\n[context window: still ~{remaining} tokens after "
                  f"eliding everything eligible; the {keep_recent} protected "
                  f"recent outputs dominate. Lower keep_recent if this "
                  f"repeats.]\n")

    return context_window_guard
