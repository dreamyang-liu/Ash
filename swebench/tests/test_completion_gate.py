"""Claimed completion is challenged when it arrives implausibly early."""

from swebench.agent.completion_gate import (DEFAULT_MAX_CHALLENGES,
                                            make_completion_challenge)
from swebench.agent.conversation import Conversation
from swebench.models import CostTracker, Trajectory


class Agent:
    def __init__(self, calls):
        self.cost = CostTracker()
        self.cost.api_calls = calls
        self.traced = []

    def _trace(self, text):
        self.traced.append(text)


def conversation(*, verified_turns_ago=None):
    """A conversation whose last verification command was N assistant turns ago."""
    conv = Conversation(Trajectory())
    total = 6
    for turn in range(total):
        calls = None
        ago = total - turn
        if verified_turns_ago is not None and ago == verified_turns_ago:
            calls = [{"id": f"c{turn}", "type": "function",
                      "function": {"name": "shell",
                                   "arguments": '{"command": "make test"}'}}]
        elif turn % 2 == 0:
            calls = [{"id": f"c{turn}", "type": "function",
                      "function": {"name": "shell",
                                   "arguments": '{"command": "sed -n 1,20p x.c"}'}}]
        conv.messages.append({"role": "assistant", "content": "working",
                              **({"tool_calls": calls} if calls else {})})
    return conv


def test_an_early_completion_is_challenged():
    """Measured on a real batch: eleven of twenty tasks stopped between 13 and
    59 steps of 2000, on work estimated at 5-12 expert hours, one announcing
    that 68,186 golden tests passed -- a number it got by counting the golden
    file, with the test script last run three edits earlier."""
    gate = make_completion_challenge(step_limit=2000, expert_hours=12)
    agent, conv = Agent(14), conversation(verified_turns_ago=None)
    assert gate(agent, conv) is True, "the loop is asked for another turn"
    challenge = conv.messages[-1]
    assert challenge["role"] == "user"
    assert "14 of 2000" in challenge["content"]
    assert "12 hours" in challenge["content"]
    assert "not run any build or test command" in challenge["content"]


def test_a_completion_after_real_effort_stands():
    """Above the threshold the agent has spent real effort and its judgement is
    not second-guessed."""
    gate = make_completion_challenge(step_limit=2000)
    agent, conv = Agent(900), conversation()
    assert gate(agent, conv) is False
    assert conv.messages[-1]["role"] == "assistant", "nothing appended"


def test_stale_verification_is_named_as_such():
    gate = make_completion_challenge(step_limit=1000)
    agent, conv = Agent(20), conversation(verified_turns_ago=5)
    assert gate(agent, conv) is True
    assert "5 turns ago" in conv.messages[-1]["content"]


def test_recent_verification_gets_a_narrower_prompt():
    gate = make_completion_challenge(step_limit=1000)
    agent, conv = Agent(20), conversation(verified_turns_ago=1)
    assert gate(agent, conv) is True
    text = conv.messages[-1]["content"]
    assert "exit status" in text, "the doubt is about reading the output"


def test_challenges_are_bounded():
    """If the agent holds its position, asking again is nagging -- and a run
    that cannot finish is worse than one that finishes early."""
    gate = make_completion_challenge(step_limit=2000)
    agent = Agent(10)
    conv = conversation()
    fired = sum(1 for _ in range(6) if gate(agent, conv))
    assert fired == DEFAULT_MAX_CHALLENGES


def test_the_gate_never_touches_the_grader():
    """The task's verifier is the hidden test; consulting it here would be
    cheating. Everything the challenge says comes from the transcript."""
    import inspect

    from swebench.agent import completion_gate
    source = inspect.getsource(completion_gate)
    for forbidden in ("grade(", "reward", "metrics", "test.sh"):
        assert forbidden not in source, forbidden
