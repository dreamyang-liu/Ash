"""Elision keeps marathon transcripts inside the model's window."""

from swebench.agent.context_window import (FALLBACK_WINDOW_TOKENS, count_tokens,
                                           make_context_window_guard,
                                           model_window_tokens,
                                           transcript_chars)
from swebench.agent.conversation import Conversation
from swebench.models import AgentConfig, Trajectory


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.thinking_blocks = None


class FakeAgent:
    """Stands in for AshAgent: a model name and a trace sink."""

    def __init__(self, model="gpt-4o-mini", window=None):
        self.config = AgentConfig(model=model)
        self.tools_schema = None
        self.notes: list[str] = []

    def _trace(self, text):
        self.notes.append(text)


def conversation_with(results: int, chars_each: int,
                      call_chars: int = 0) -> Conversation:
    conv = Conversation(Trajectory())
    conv.add_system("sys")
    conv.add_user("task")
    for i in range(results):
        calls = None
        if call_chars:
            # Real arguments are a JSON string -- a `text_editor` write carries
            # the file in one, which is what makes tool calls heavy.
            import json as _json
            calls = [{"id": f"c{i}", "type": "function",
                      "function": {"name": "text_editor",
                                   "arguments": _json.dumps(
                                       {"command": "write", "path": f"/f{i}.c",
                                        "file_text": "a" * call_chars})}}]
        conv.add_assistant(FakeMessage(f"turn {i}", tool_calls=calls))
        conv.add_tool_result(f"id-{i}", "x" * chars_each)
    return conv


# --- measurement ----------------------------------------------------------- #

def test_tool_calls_count_toward_the_transcript():
    """They carry the bulk on code-heavy runs: one text_editor write holds a
    whole file. Ignoring them is what made character estimates look plausible
    while being 3x low."""
    without = transcript_chars(conversation_with(3, 100).messages)
    with_calls = transcript_chars(
        conversation_with(3, 100, call_chars=5_000).messages)
    assert with_calls > without + 3 * 5_000 * 0.9


def test_window_comes_from_the_model_not_a_constant():
    """A 133-step run measured ~139K tokens: fatal against 200K, fine against
    1M. The budget therefore cannot be hardcoded."""
    small = model_window_tokens("gpt-4o-mini")
    big = model_window_tokens("bedrock/us.anthropic.claude-sonnet-4-6")
    assert big > small, (small, big)
    # Unknown models assume small -- earlier elision, never a dead run.
    assert model_window_tokens("no-such-model-xyz") == FALLBACK_WINDOW_TOKENS
    assert model_window_tokens(None) == FALLBACK_WINDOW_TOKENS


def test_exact_counting_is_used_when_available():
    conv = conversation_with(5, 400)
    exact = count_tokens(conv.messages, "gpt-4o-mini")
    estimate = transcript_chars(conv.messages) // 3
    assert exact > 0
    # The estimate is in the same ballpark but not the same number; the point
    # is that the exact path is what runs.
    assert exact != estimate


def test_counting_falls_back_without_a_model():
    conv = conversation_with(5, 400)
    assert count_tokens(conv.messages, None) == transcript_chars(
        conv.messages) // 3


# --- elision policy -------------------------------------------------------- #

def test_under_budget_is_untouched():
    conv = conversation_with(results=10, chars_each=1_000)
    before = [m.get("content") for m in conv.messages]
    make_context_window_guard(window_tokens=200_000)(FakeAgent(), conv)
    assert [m.get("content") for m in conv.messages] == before


def test_over_budget_elides_oldest_tool_outputs_only():
    conv = conversation_with(results=60, chars_each=12_000)
    agent = FakeAgent()
    # keep_recent=5 leaves the target reachable; the floor is covered below.
    make_context_window_guard(window_tokens=20_000, keep_recent=5)(agent, conv)

    tool_messages = [m for m in conv.messages if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("[tool output elided")
    assert "12000 chars" in tool_messages[0]["content"]
    # The newest results are intact -- the model is acting on them.
    assert all(not m["content"].startswith("[tool output elided")
               for m in tool_messages[-5:])
    # Assistant turns -- what the agent did -- are all verbatim.
    assert all(m["content"].startswith("turn ")
               for m in conv.messages if m["role"] == "assistant")
    assert count_tokens(conv.messages, "gpt-4o-mini") < 20_000


def test_trajectory_keeps_the_full_text():
    conv = conversation_with(results=60, chars_each=12_000)
    make_context_window_guard(window_tokens=20_000,
                              keep_recent=5)(FakeAgent(), conv)
    saved = [m for m in conv.trajectory.messages if m["role"] == "tool_result"]
    assert all(len(m["content"]) == 12_000 for m in saved), (
        "elision is about what the model sees, not what the run records")


def test_elision_is_idempotent_and_bulk():
    """Cutting deep leaves a cache-friendly plateau: calling the guard again
    right away must not rewrite anything further."""
    conv = conversation_with(results=60, chars_each=12_000)
    guard = make_context_window_guard(window_tokens=20_000, keep_recent=5)
    agent = FakeAgent()
    guard(agent, conv)
    after_first = [m.get("content") for m in conv.messages]
    guard(agent, conv)
    assert [m.get("content") for m in conv.messages] == after_first


def test_unreachable_target_is_reported_not_hidden():
    """The protected tail is a floor: a target below it cannot be reached, and
    the guard must say so instead of looking like it succeeded."""
    conv = conversation_with(results=30, chars_each=12_000)
    agent = FakeAgent()
    make_context_window_guard(window_tokens=2_000, keep_recent=20)(agent, conv)
    assert any("protected recent" in note for note in agent.notes)


def test_large_window_leaves_a_long_transcript_alone():
    """The same transcript that would be eliminated against a small window is
    untouched against a large one -- the 133-step run's 139K tokens were 14%
    of a 1M window, and eliding there would only throw away context."""
    conv = conversation_with(results=40, chars_each=12_000)
    before = [m.get("content") for m in conv.messages]
    make_context_window_guard(window_tokens=1_000_000)(FakeAgent(), conv)
    assert [m.get("content") for m in conv.messages] == before


# --- wiring ---------------------------------------------------------------- #

def test_harnesses_install_the_guard():
    """A guard nobody mounts protects nothing. This exists because the first
    version of this module was written, tested, and left unwired -- only a
    hand-rolled driver ever installed it, so `python -m swebench` runs had no
    protection at all."""
    import inspect
    from swebench.harnesses import litellm as harness
    from swebench import rollout_server

    for module in (harness, rollout_server):
        source = inspect.getsource(module)
        assert "make_context_window_guard(" in source, module.__name__
        assert "before_query_hooks.append" in source, module.__name__


def test_guard_is_configurable_and_can_be_disabled():
    from swebench.__main__ import _flatten_config
    flat = _flatten_config({"execution": {"context_budget_fraction": 0.0,
                                          "context_target_fraction": 0.3}})
    assert flat["context_budget_fraction"] == 0.0
    assert flat["context_target_fraction"] == 0.3


# --- strategies ------------------------------------------------------------ #

def test_unknown_strategy_is_refused_at_construction():
    """A typo in config must fail loudly, not silently leave the transcript
    unmanaged until the window overflows."""
    import pytest
    with pytest.raises(ValueError, match="unknown context strategy"):
        make_context_window_guard(strategy="compress")


def test_summarize_replaces_the_span_with_one_summary(monkeypatch):
    """The findings survive in prose where elision would have kept only the
    commands. Every folded message is still replaced, because a provider
    requires one tool message per tool call in the preceding turn."""
    import swebench.agent.context_window as cw
    monkeypatch.setattr(cw, "_ask_for_summary",
                        lambda agent, model, excerpts:
                        f"Three tests failed in fse.c (saw {len(excerpts)} outputs).")

    conv = conversation_with(results=30, chars_each=12_000)
    agent = FakeAgent()
    make_context_window_guard(strategy="summarize", window_tokens=20_000,
                              keep_recent=5)(agent, conv)

    tools = [m for m in conv.messages if m["role"] == "tool"]
    assert tools[0]["content"].startswith("[tool output summarized")
    assert "Three tests failed in fse.c" in tools[0]["content"]
    # The rest of the span is stubbed, not deleted.
    assert all(m["content"].startswith("[tool output") for m in tools[:-5])
    assert all(not m["content"].startswith("[tool output") for m in tools[-5:])


def test_summarize_falls_back_to_elision_when_the_model_cannot(monkeypatch):
    """Being unable to summarize must not become being unable to stay inside
    the window."""
    import swebench.agent.context_window as cw
    monkeypatch.setattr(cw, "_ask_for_summary",
                        lambda agent, model, excerpts: "")

    conv = conversation_with(results=30, chars_each=12_000)
    make_context_window_guard(strategy="summarize", window_tokens=20_000,
                              keep_recent=5)(FakeAgent(), conv)
    tools = [m for m in conv.messages if m["role"] == "tool"]
    assert tools[0]["content"].startswith("[tool output elided")
    assert count_tokens(conv.messages, "gpt-4o-mini") < 20_000


def test_summary_cost_is_charged_to_the_run(monkeypatch):
    """A hidden model call would make a run's reported cost wrong."""
    import swebench.agent.context_window as cw

    class Response:
        class Choice:
            class Message:
                content = "findings"
            message = Message()
        choices = [Choice()]
        usage = None

    class FakeLitellm:
        @staticmethod
        def completion(**kwargs):
            return Response()

    monkeypatch.setitem(__import__("sys").modules, "litellm", FakeLitellm)

    charged = []

    class CountingAgent(FakeAgent):
        class cost:
            @staticmethod
            def update(response):
                charged.append(response)

    text = cw._ask_for_summary(CountingAgent(), "gpt-4o-mini", ["some output"])
    assert text == "findings"
    assert len(charged) == 1, "the summary call is spent from the run's budget"


def test_both_strategies_are_selectable_from_config():
    from swebench.__main__ import _flatten_config
    assert _flatten_config({"execution": {"context_strategy": "summarize"}})[
        "context_strategy"] == "summarize"
    from swebench.agent.context_window import STRATEGIES
    assert set(STRATEGIES) == {"elide", "summarize"}


def test_truncated_tool_call_does_not_poison_the_conversation():
    """A model that hits its output limit mid-call emits arguments that stop
    partway through their own JSON. Kept verbatim, every later request fails
    converting that message and the run dies unrecoverably -- seen on a real
    marathon attempt 37 messages in. The call is repaired to something a
    provider accepts, and the loop tells the model what happened."""
    import json
    from swebench.agent.conversation import plain_tool_calls

    truncated = [{"id": "t1", "type": "function",
                  "function": {"name": "text_editor",
                               "arguments": '{"command": "write", "path": "/x.c"'}}]
    repaired = plain_tool_calls(truncated)
    assert json.loads(repaired[0]["function"]["arguments"]) == {}

    conv = Conversation(Trajectory())
    conv.add_assistant(FakeMessage("writing the file", tool_calls=truncated))
    # Every tool call in the message must be convertible, or the next call dies.
    for call in conv.messages[-1]["tool_calls"]:
        json.loads(call["function"]["arguments"])


def test_valid_large_tool_calls_are_left_alone():
    import json
    from swebench.agent.conversation import plain_tool_calls

    big = json.dumps({"command": "write", "path": "/x.c", "file_text": "y" * 50_000})
    kept = plain_tool_calls([{"id": "k", "type": "function",
                             "function": {"name": "text_editor", "arguments": big}}])
    assert kept[0]["function"]["arguments"] == big
