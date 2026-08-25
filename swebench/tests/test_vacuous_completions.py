"""A completion that says nothing must not be read as "the agent is done"."""

from swebench.agent import _is_vacuous
from swebench.agent.conversation import Conversation
from swebench.models import Trajectory


class Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.thinking_blocks = None


def test_empty_and_provider_placeholders_are_vacuous():
    """The placeholder is the provider's, not the model's. Reading it as an
    answer is how a 57-step run finished as "completed" with nothing built:
    three empty turns were correctly re-prompted, then the proxy substituted a
    notice for the fourth, which looked like a text-only reply and tripped the
    two-strikes finish rule."""
    assert _is_vacuous(None)
    assert _is_vacuous("")
    assert _is_vacuous("   \n ")
    assert _is_vacuous("[System: Empty message content sanitised to satisfy protocol]")
    assert _is_vacuous("[system: empty message content]")


def test_real_answers_are_not_vacuous():
    assert not _is_vacuous("The fix is complete; all tests pass.")
    # Bracketed prose that is the model talking, not a provider notice.
    assert not _is_vacuous("[note] I will now run the tests")
    assert not _is_vacuous("[System: ...] and then I fixed the parser")


def test_vacuous_turns_do_not_count_toward_finishing():
    conv = Conversation(Trajectory())
    conv.add_assistant(Msg("Let me look at the parser.", tool_calls=[
        {"id": "c1", "type": "function",
         "function": {"name": "shell", "arguments": "{}"}}]))
    assert conv.consecutive_no_tool == 0

    # Two vacuous turns in a row: still not "done".
    conv.add_assistant(Msg(""))
    conv.add_assistant(Msg("[System: Empty message content sanitised to satisfy protocol]"))
    assert conv.consecutive_no_tool == 0, (
        "a model whose completions came back empty has not decided it is finished")

    # Two real text-only turns: that IS the finish signal.
    conv.add_assistant(Msg("I believe the implementation is complete."))
    conv.add_assistant(Msg("Nothing further to do."))
    assert conv.consecutive_no_tool == 2
