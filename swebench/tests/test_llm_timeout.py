"""Every completion carries a deadline, so a dead connection costs one retry."""

from swebench.agent.llm import REQUEST_TIMEOUT_SECONDS, LLMClient
from swebench.models import AgentConfig, CostTracker


def client(**overrides) -> LLMClient:
    config = AgentConfig(model="gpt-4o-mini", **overrides)
    return LLMClient(config, CostTracker(), tools_schema=[])


def test_completion_kwargs_carry_a_timeout():
    """The retry loop already handles timeouts; without this it is unreachable
    for the failure that matters. Observed on a 5-hour marathon run: the
    provider stopped answering without closing the socket, and the process sat
    in epoll for 2h48m with zero CPU, no retry and no error, losing the rest of
    its budget."""
    kwargs = client()._build_kwargs([{"role": "user", "content": "hi"}])
    assert kwargs["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert REQUEST_TIMEOUT_SECONDS >= 600, (
        "a legitimate call on this path writes whole files with thinking on")


def test_timeout_is_adjustable_per_client():
    c = client()
    c.request_timeout = 120
    assert c._build_kwargs([{"role": "user", "content": "hi"}])["timeout"] == 120


def test_timeouts_are_retryable():
    """The ceiling only helps if hitting it leads to a retry rather than a
    crash."""
    import litellm

    assert LLMClient._retryable(litellm.exceptions.Timeout(
        message="timed out", model="m", llm_provider="p"))
