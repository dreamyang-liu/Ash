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
