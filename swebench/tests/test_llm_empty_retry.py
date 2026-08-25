"""A completion with nothing in it is asked for again, not passed upstream."""

from swebench.agent.llm import EMPTY_RETRIES, LLMClient
from swebench.models import AgentConfig, CostTracker


class Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class Response:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [type("C", (), {"message": Message(content, tool_calls),
                                       "finish_reason": "stop"})()]
        self.usage = None


def client(monkeypatch, responses):
    """An LLMClient whose completion() returns each response in turn."""
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    import swebench.agent.llm as llm
    monkeypatch.setattr(llm, "_get_litellm",
                        lambda: (fake_completion, lambda **k: None))
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    c = LLMClient(AgentConfig(model="m"), CostTracker(), tools_schema=[])
    c.stream = False
    return c, calls


def test_an_empty_completion_is_retried(monkeypatch):
    """Handing it upstream costs a turn on a meaningless message, and some
    providers then replace that empty assistant message with a placeholder on
    the next request -- which reads like the model talking and once ended a run
    as 'completed' with nothing built."""
    c, calls = client(monkeypatch, [Response(content=""),
                                    Response(content=""),
                                    Response(content="here is the fix")])
    result = c.query([{"role": "user", "content": "go"}])
    assert result.choices[0].message.content == "here is the fix"
    assert len(calls) == 3, "asked again until something came back"


def test_a_tool_call_without_text_is_not_empty(monkeypatch):
    """Most turns are exactly this: a tool call and no prose."""
    c, calls = client(monkeypatch, [Response(tool_calls=[{"id": "1"}])])
    c.query([{"role": "user", "content": "go"}])
    assert len(calls) == 1, "not retried"


def test_retries_are_bounded(monkeypatch):
    """A model that keeps returning nothing is telling you something a retry
    will not fix; the loop's re-prompt handles it from there."""
    c, calls = client(monkeypatch, [Response(content="")])
    result = c.query([{"role": "user", "content": "go"}])
    assert len(calls) == EMPTY_RETRIES + 1
    assert not (result.choices[0].message.content or "")


def test_streaming_emptiness_is_judged_after_assembly(monkeypatch):
    """A stream's content only exists once its chunks are joined, so the check
    has to happen after consuming it -- and consuming it must not be paid for
    twice."""
    import swebench.agent.llm as llm

    assembled = [Response(content=""), Response(content="done")]
    consumed = []

    def fake_completion(**kwargs):
        return "raw-stream"

    monkeypatch.setattr(llm, "_get_litellm",
                        lambda: (fake_completion, lambda **k: None))
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    c = LLMClient(AgentConfig(model="m"), CostTracker(), tools_schema=[])
    c.stream = True
    c._consume_stream = lambda raw, builder: (
        consumed.append(raw), assembled[min(len(consumed) - 1,
                                            len(assembled) - 1)])[1]

    result = c.query([{"role": "user", "content": "go"}])
    assert result.choices[0].message.content == "done"
    assert len(consumed) == 2, "each attempt assembled once"


def test_cache_markers_only_go_to_models_that_take_them(monkeypatch):
    """cache_control is an Anthropic Messages API concept. Bedrock turns the
    markers into cachePoint blocks, and its non-Anthropic models reject those
    as a hard error -- "your request did not allow prompt caching" killed a
    luna run on its very first call. Metadata decides; unknown models get
    markers only when addressed over the Anthropic protocol, where an ignored
    marker is legal."""
    import swebench.agent.llm as llm

    def fake_info(model):
        return {"bedrock/us.anthropic.claude-sonnet-4-6":
                    {"supports_prompt_caching": True},
                "bedrock/us.openai.gpt-5.6-luna": {}}[model]

    class FakeLitellm:
        get_model_info = staticmethod(fake_info)
    monkeypatch.setitem(__import__("sys").modules, "litellm", FakeLitellm)

    def client_for(model):
        c = LLMClient(AgentConfig(model=model), CostTracker(), tools_schema=[])
        return c

    assert client_for("bedrock/us.anthropic.claude-sonnet-4-6")._model_takes_cache_markers()
    assert not client_for("bedrock/us.openai.gpt-5.6-luna")._model_takes_cache_markers()
    # Unknown to litellm, but spoken to over the Anthropic protocol: the
    # marker is protocol-legal there and the server may simply ignore it.
    class NoInfo:
        @staticmethod
        def get_model_info(model):
            raise ValueError("unmapped")
    monkeypatch.setitem(__import__("sys").modules, "litellm", NoInfo)
    assert client_for("anthropic/deepseek-v4-flash")._model_takes_cache_markers()
    assert not client_for("openai/some-proxy-model")._model_takes_cache_markers()


def test_unknown_failures_retry_and_permanent_ones_do_not():
    """The classifier was an allowlist of retryable names, and a 103-step run
    died because a Bedrock 5xx arrived wrapped as MidStreamFallbackError --
    a name no allowlist had heard of. Eight wasted backoffs on a permanent
    error cost ~4 minutes; one unretried transient costs the run."""
    class MidStreamFallbackError(Exception): ...
    class ServiceUnavailableError(Exception): ...
    class APIConnectionError(Exception): ...
    class BadRequestError(Exception): ...
    class AuthenticationError(Exception): ...
    class ContextWindowExceededError(Exception): ...

    retry = LLMClient._retryable
    # Transients -- including wrapper types invented after this code was written.
    assert retry(MidStreamFallbackError(
        "litellm.ServiceUnavailableError: BedrockException - internalServerException"))
    assert retry(ServiceUnavailableError("503"))
    assert retry(APIConnectionError("connection reset by peer"))
    assert retry(TimeoutError("model stream stalled for 180s"))
    # Permanent -- retrying these eight times would just delay the diagnosis.
    assert not retry(BadRequestError("max_tokens out of range"))
    assert not retry(AuthenticationError("bad key"))
    assert not retry(ContextWindowExceededError("prompt too long"))
    # Permanent-by-message even when the wrapper type looks transient: the
    # cache-marker misconfiguration arrived as an APIConnectionError.
    assert not retry(APIConnectionError(
        "BedrockException - You invoked an unsupported model or your request "
        "did not allow prompt caching."))
