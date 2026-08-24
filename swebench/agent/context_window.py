"""Keep a marathon-length conversation inside the model's context window.

The conversation is append-only, which is fine for the 20-60 step episodes
this harness grew up on and fatal past ~150: tool outputs dominate the
transcript (each bounded to ~12K characters by the truncation interceptor,
but unbounded in number), and once the window fills the next model call is
an API error that kills the run with the budget mostly unspent.

The fix is elision, not summarization: old *tool outputs* are replaced with a
one-line stub while every assistant message -- the reasoning and the commands
it ran -- stays verbatim. What the agent DID is cheap and irreplaceable; what
it SAW is bulky and mostly re-obtainable (the file is still on disk; rerun
the command if it matters). No extra model call, no summary that might lie.

Elision runs in bulk, rarely, rather than as a sliding window: prompt caching
prices the transcript by its stable prefix, and rewriting one old message per
step would invalidate the cache every step. Cutting once from 70% down to 40%
of the budget leaves long cacheable plateaus between cuts.

The trajectory keeps the full text: elision is about what the MODEL sees on
the next call, not about what the run records.
"""

from __future__ import annotations

#: Characters per token before any call has reported real usage. Deliberately
#: denser than prose (~4): the transcripts that fill a window are code, JSON
#: and hex dumps, and a measured probe put "x "-filler at 2. Guessing dense
#: means eliding slightly early rather than overflowing.
_FALLBACK_CHARS_PER_TOKEN = 3

#: Never elide the most recent tool outputs: the model is usually acting on
#: them right now. This is also a floor on what elision can achieve -- the
#: protected tail is untouchable, so a cut target below it is unreachable by
#: construction (20 results at the truncation interceptor's 12K ceiling is
#: ~80K tokens of tail on its own).
_KEEP_RECENT_RESULTS = 20

_STUB = ("[tool output elided to fit the context window -- {chars} chars. "
         "The state it described may be stale; re-run the command if it "
         "matters now.]")


def transcript_chars(messages: list[dict]) -> int:
    """Characters the model-facing transcript spends, tool calls included.

    Tool calls count because they carry the bulk on code-heavy runs: a
    `text_editor` write holds the whole file, a shell heredoc a whole
    program. (The saved trajectory does not record them, which is why
    measuring the trajectory understates what the model sees.)
    """
    total = 0
    for message in messages:
        total += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or ():
            total += len(str(call))
    return total


def chars_per_token(agent, messages: list[dict]) -> float:
    """Characters-per-token calibrated against what the provider last charged.

    The provider's own count for the previous call is the only honest measure
    of window pressure, but it is a single number attached to a transcript
    that has since grown. Turning it into a ratio makes it usable as the
    transcript changes -- including while eliding, where the loop needs to
    know whether it has cut enough yet.
    """
    reported = getattr(getattr(agent, "cost", None), "last_input_tokens", 0) or 0
    chars = transcript_chars(messages)
    if reported <= 0 or chars <= 0:
        return _FALLBACK_CHARS_PER_TOKEN
    # The reported count also covers the system prompt and tool schemas,
    # which this ratio then spreads over transcript characters. That inflates
    # the estimate slightly, and keeps inflating it as the transcript shrinks
    # -- again the safe direction.
    return max(chars / reported, 0.5)


def transcript_tokens(messages: list[dict], agent=None) -> int:
    """Window cost of the transcript in tokens, calibrated when possible."""
    return int(transcript_chars(messages) / chars_per_token(agent, messages))


def make_context_window_guard(budget_tokens: int = 140_000,
                              cut_to_tokens: int = 60_000,
                              keep_recent: int = _KEEP_RECENT_RESULTS):
    """A ``before_query`` hook that keeps the transcript under budget.

    ``budget_tokens`` should sit well below the model's window: the system
    prompt, tool schemas, and the next response all live in the same window.
    ``cut_to_tokens`` is a target, not a guarantee -- the ``keep_recent``
    newest tool outputs are never elided, so a target below what that tail
    costs cannot be reached and the guard says so rather than pretending.
    """
    def context_window_guard(agent, conv) -> None:
        # Calibrate once per firing: the ratio is derived from the transcript
        # as it stands now, and re-deriving it mid-elision would chase its
        # own tail (chars shrink, so the same reported token count would
        # imply an ever-denser transcript).
        ratio = chars_per_token(agent, conv.messages)
        used = lambda: int(transcript_chars(conv.messages) / ratio)
        if used() <= budget_tokens:
            return

        # Oldest first, keeping the newest _KEEP_RECENT_RESULTS untouched.
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
            if used() <= cut_to_tokens:
                break

        trace = getattr(agent, "_trace", None)
        if elided and trace:
            trace(f"\n[context window: elided {elided} old tool outputs; "
                  f"transcript now ~{used()} tokens "
                  f"({ratio:.1f} chars/token as last charged)]\n")
        if used() > cut_to_tokens and trace:
            # Everything elidable is already gone: what remains is the
            # protected tail plus the assistant turns, and only a smaller
            # keep_recent (or a bigger window) can help.
            trace(f"\n[context window: still ~{used()} tokens after eliding "
                  f"everything eligible; the {keep_recent} protected recent "
                  f"outputs dominate. Lower keep_recent if this repeats.]\n")

    return context_window_guard
