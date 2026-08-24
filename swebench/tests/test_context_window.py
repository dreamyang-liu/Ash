"""Elision keeps marathon transcripts inside the model's window."""

from swebench.agent.context_window import (make_context_window_guard,
                                           transcript_tokens)
from swebench.agent.conversation import Conversation
from swebench.models import Trajectory


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.thinking_blocks = None


class FakeAgent:
    _trace = None


def conversation_with(results: int, chars_each: int) -> Conversation:
    conv = Conversation(Trajectory())
    conv.add_system("sys")
    conv.add_user("task")
    for i in range(results):
        conv.add_assistant(FakeMessage(f"turn {i}", tool_calls=None))
        conv.add_tool_result(f"id-{i}", "x" * chars_each)
    return conv


def test_under_budget_is_untouched():
    conv = conversation_with(results=10, chars_each=1000)
    before = [m.get("content") for m in conv.messages]
    make_context_window_guard(budget_tokens=140_000)(FakeAgent(), conv)
    assert [m.get("content") for m in conv.messages] == before


def test_over_budget_elides_oldest_tool_outputs_only():
    # 60 results x 12K chars: well past a 140K budget at 3 chars/token.
    conv = conversation_with(results=60, chars_each=12_000)
    before = transcript_tokens(conv.messages)
    # keep_recent=5 so the protected tail (5 x 12K chars = ~20K tokens) leaves
    # the target reachable; the floor itself is covered separately below.
    guard = make_context_window_guard(budget_tokens=140_000,
                                      cut_to_tokens=60_000, keep_recent=5)
    guard(FakeAgent(), conv)

    assert transcript_tokens(conv.messages) <= 60_000, (
        f"{before} -> {transcript_tokens(conv.messages)}")
    tool_messages = [m for m in conv.messages if m["role"] == "tool"]
    # The newest results are intact -- the model is acting on them.
    assert all(not m["content"].startswith("[tool output elided")
               for m in tool_messages[-5:])
    # The oldest were stubbed, and the stub says how much went missing.
    assert tool_messages[0]["content"].startswith("[tool output elided")
    assert "12000 chars" in tool_messages[0]["content"]
    # Assistant turns -- what the agent did -- are all verbatim.
    assert all(m["content"].startswith("turn ")
               for m in conv.messages if m["role"] == "assistant")


def test_trajectory_keeps_the_full_text():
    conv = conversation_with(results=60, chars_each=12_000)
    make_context_window_guard(budget_tokens=140_000, cut_to_tokens=60_000,
                              keep_recent=5)(FakeAgent(), conv)
    saved = [m for m in conv.trajectory.messages if m["role"] == "tool_result"]
    assert all(len(m["content"]) == 12_000 for m in saved), (
        "elision is about what the model sees, not what the run records")


def test_elision_is_idempotent_and_bulk():
    """Cutting to well under budget leaves a cache-friendly plateau: calling
    the guard again right away must not rewrite anything further."""
    conv = conversation_with(results=60, chars_each=12_000)
    guard = make_context_window_guard(budget_tokens=140_000,
                                      cut_to_tokens=60_000, keep_recent=5)
    guard(FakeAgent(), conv)
    after_first = [m.get("content") for m in conv.messages]
    guard(FakeAgent(), conv)
    assert [m.get("content") for m in conv.messages] == after_first


def test_real_reported_tokens_beat_the_character_estimate():
    """The provider's count for the last call is the only honest measure of
    window pressure: a code-heavy transcript measured 3x denser than the
    character estimate assumed, which would have let the window overflow
    before the guard ever fired."""
    class Cost:
        last_input_tokens = 120_000

    class CalibratedAgent:
        _trace = None
        cost = Cost()

    conv = conversation_with(results=30, chars_each=4_000)   # 120K chars
    # Uncalibrated: 120K chars / 3 = ~40K tokens, comfortably "under budget".
    assert transcript_tokens(conv.messages) < 60_000
    # Calibrated by what was actually charged, the same transcript is 120K.
    agent = CalibratedAgent()
    assert transcript_tokens(conv.messages, agent) >= 110_000

    # And the guard therefore fires on it, where the estimate would not have.
    make_context_window_guard(budget_tokens=100_000,
                              cut_to_tokens=60_000, keep_recent=5)(agent, conv)
    stubbed = sum(1 for m in conv.messages
                  if m["role"] == "tool" and m["content"].startswith("[tool output elided"))
    assert stubbed > 0


def test_unreachable_target_is_reported_not_hidden():
    """The protected tail is a floor: a target below it cannot be reached, and
    the guard must say so instead of looking like it succeeded."""
    notes = []

    class NoisyAgent:
        cost = None
        def _trace(self, text):
            notes.append(text)

    conv = conversation_with(results=30, chars_each=12_000)
    make_context_window_guard(budget_tokens=50_000, cut_to_tokens=1_000,
                              keep_recent=20)(NoisyAgent(), conv)
    assert any("protected recent" in n for n in notes)
