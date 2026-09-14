from __future__ import annotations

from swebench.agent.llm import LLMClient
from swebench.models import AgentConfig, CostTracker


def test_model_transport_controls_are_forwarded_to_litellm():
    client = LLMClient(
        AgentConfig(
            model="openai/local",
            request_timeout=1800.0,
            request_max_retries=0,
            retry_attempts=1,
        ),
        CostTracker(),
    )

    kwargs = client._build_kwargs([{"role": "user", "content": "hello"}])

    assert kwargs["timeout"] == 1800.0
    assert kwargs["max_retries"] == 0


def test_retryable_transport_failure_is_retried_before_returning(monkeypatch):
    calls = 0

    response = object()

    def fail_once(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out")
        return response

    monkeypatch.setattr(
        "swebench.agent.llm._get_litellm", lambda: (fail_once, object())
    )
    monkeypatch.setattr("swebench.agent.llm.time.sleep", lambda _seconds: None)
    client = LLMClient(
        AgentConfig(model="openai/local", retry_attempts=3), CostTracker()
    )
    client.stream = False

    assert client.query([{"role": "user", "content": "hello"}]) is response
    assert calls == 2


def test_non_retryable_model_failure_is_not_retried(monkeypatch):
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("invalid request")

    monkeypatch.setattr("swebench.agent.llm._get_litellm", lambda: (fail, object()))
    monkeypatch.setattr("swebench.agent.llm.time.sleep", lambda _seconds: None)
    client = LLMClient(
        AgentConfig(model="openai/local", retry_attempts=3), CostTracker()
    )

    try:
        client.query([{"role": "user", "content": "hello"}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    assert calls == 1
