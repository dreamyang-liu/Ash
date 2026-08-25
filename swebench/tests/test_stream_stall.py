"""A streaming response that goes quiet is abandoned, not waited on forever."""

import time

import pytest

from swebench.agent.llm import (STREAM_STALL_TIMEOUT_SECONDS, LLMClient,
                                _with_stall_timeout)


def test_a_silent_stream_raises_instead_of_hanging():
    """The request timeout bounds getting a response started, not iterating one.
    Measured twice on this stack: 2h48m of silence on one run, and 20 minutes on
    another that had a 900s request timeout set -- the timeout was never in
    play, because the socket was already open and the provider had simply
    stopped sending."""
    def stalling():
        yield "first"
        time.sleep(30)          # longer than the watchdog below
        yield "never reached"

    seen, notes = [], []
    with pytest.raises(TimeoutError, match="stalled"):
        for item in _with_stall_timeout(stalling(), 0.5, notes.append):
            seen.append(item)
    assert seen == ["first"], "what did arrive is still delivered"
    assert any("STALL" in n for n in notes)


def test_a_healthy_stream_passes_through_untouched():
    items = list(_with_stall_timeout(iter(["a", "b", "c"]), 5.0, lambda _t: None))
    assert items == ["a", "b", "c"]


def test_an_error_inside_the_stream_is_forwarded():
    """The watchdog must not swallow the provider's own failure."""
    def failing():
        yield "partial"
        raise RuntimeError("provider hung up")

    with pytest.raises(RuntimeError, match="hung up"):
        list(_with_stall_timeout(failing(), 5.0, lambda _t: None))


def test_a_stalled_stream_is_retryable():
    """It only helps if the retry loop treats it as worth another attempt."""
    assert LLMClient._retryable(TimeoutError("model stream stalled for 180s"))


def test_the_stall_budget_is_far_below_the_request_budget():
    """It measures the gap between chunks, and a model that is working sends
    something continuously."""
    from swebench.agent.llm import REQUEST_TIMEOUT_SECONDS
    assert STREAM_STALL_TIMEOUT_SECONDS < REQUEST_TIMEOUT_SECONDS / 2
