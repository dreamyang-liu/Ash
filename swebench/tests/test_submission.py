"""Asking the agent for its answer (swebench/submission.py).

The agent knows which files it fixed; a harness reading git state can only guess,
and guessed wrong once already (66 files, of which one was the fix). So the agent
produces the diff, and the eval layer reserves the budget to do it in.

Covered:
- the reserve fires by whichever ceiling binds first, not by steps alone
- it also fires when the agent stops on its own, which is the common ending
- it fires once across both paths, and only near the end
- extraction reads the last diff the agent sent, and nothing else
- a reply with no diff yields no submission -- there is no fallback
- prose and example code are not mistaken for a patch
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from swebench.models import CostTracker, Trajectory
from swebench.submission import (
    DEFAULT_RESERVE_STEPS,
    extract_submission,
    reserve_submission,
    submission_prompt,
)

PATCH = ("diff --git a/pkg/mod.py b/pkg/mod.py\n"
         "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
         "@@ -1,2 +1,2 @@\n-    return 1\n+    return 2\n")


class FakeConv:
    """Records which way a prompt was added, since the two paths differ."""

    def __init__(self):
        self.appended = []      # onto the last user/tool message
        self.user_turns = []    # as a new user turn

    def append_to_last(self, text):
        self.appended.append(text)

    def add_user(self, text):
        self.user_turns.append(text)

    @property
    def asks(self):
        return self.appended + self.user_turns


def fake_agent(api_calls: int, cost: float, step_limit: int = 100,
               cost_limit: float = 3.0):
    tracker = CostTracker(total_cost=cost, api_calls=api_calls)
    return SimpleNamespace(
        cost=tracker,
        config=SimpleNamespace(step_limit=step_limit, cost_limit=cost_limit,
                               workdir="/testbed"),
        hook_state={},
        _trace=lambda text: None,
    )


# --------------------------------------------------------------------------- #
#  The reserve
# --------------------------------------------------------------------------- #

def test_the_reserve_fires_when_money_runs_out_before_steps():
    """The real interruption we saw: 59 of 100 steps used, $3.07 of $3.00 spent.
    Counting steps alone would have called that comfortable with 41 to go."""
    agent, conv = fake_agent(api_calls=57, cost=2.90), FakeConv()
    before_query, _ = reserve_submission()
    before_query(agent, conv)
    assert conv.asks, "budget was nearly gone and nothing was asked for"
    assert "Submit now" in conv.asks[0]


def test_the_reserve_fires_when_steps_run_out_before_money():
    agent, conv = fake_agent(api_calls=98, cost=0.10), FakeConv()
    before_query, _ = reserve_submission()
    before_query(agent, conv)
    assert conv.asks


def test_the_reserve_stays_quiet_early_on():
    """Asking at step 5 of 100 would spend the agent's turn on paperwork."""
    agent, conv = fake_agent(api_calls=5, cost=0.20), FakeConv()
    before_query, _ = reserve_submission()
    before_query(agent, conv)
    assert conv.asks == []


def test_an_agent_that_stops_on_its_own_is_still_asked():
    """The ending 24 of 25 recorded runs took. A budget-triggered reserve alone
    would have produced an empty prediction for nearly every successful run."""
    agent, conv = fake_agent(api_calls=12, cost=0.40), FakeConv()
    _, before_finish = reserve_submission()
    assert before_finish(agent, conv) is True, "run ended with nothing asked for"
    assert "Submit now" in conv.user_turns[0]


def test_the_finish_ask_is_a_new_turn_not_an_append():
    """At that point the last message is the assistant saying it is done, so
    appending would bury the request in a turn already answered."""
    agent, conv = fake_agent(api_calls=12, cost=0.40), FakeConv()
    _, before_finish = reserve_submission()
    before_finish(agent, conv)
    assert conv.user_turns and conv.appended == []


def test_the_mid_run_ask_appends_rather_than_adding_a_turn():
    """Mid-run the conversation ends with a tool result; a bare user turn after
    one would orphan the tool call."""
    agent, conv = fake_agent(api_calls=98, cost=0.10), FakeConv()
    before_query, _ = reserve_submission()
    before_query(agent, conv)
    assert conv.appended and conv.user_turns == []


def test_the_second_finish_lets_the_run_end():
    """The hook buys exactly one extra turn -- returning True forever would trap
    the loop until the step limit."""
    agent, conv = fake_agent(api_calls=12, cost=0.40), FakeConv()
    _, before_finish = reserve_submission()
    assert before_finish(agent, conv) is True
    assert before_finish(agent, conv) is False
    assert len(conv.asks) == 1


def test_the_two_paths_ask_only_once_between_them():
    """An agent asked at the budget line, that then submits and stops, must not
    be asked again: that would spend the reserve it was just given."""
    agent, conv = fake_agent(api_calls=98, cost=0.10), FakeConv()
    before_query, before_finish = reserve_submission()
    before_query(agent, conv)
    assert before_finish(agent, conv) is False
    assert len(conv.asks) == 1


def test_the_reserve_asks_only_once():
    """Repeating the request every turn would consume the reserve it created."""
    agent, conv = fake_agent(api_calls=98, cost=0.10), FakeConv()
    before_query, _ = reserve_submission()
    for _ in range(4):
        before_query(agent, conv)
    assert len(conv.asks) == 1


def test_the_ask_is_recorded_in_per_run_state():
    """Not on the agent: a reused agent's later runs must not inherit the flag.
    That the loop actually clears it is checked through the real loop below."""
    agent, conv = fake_agent(api_calls=98, cost=0.10), FakeConv()
    before_query, _ = reserve_submission()
    before_query(agent, conv)
    assert agent.hook_state.get("submission_asked") is True


def test_a_bigger_reserve_asks_sooner():
    small_q, _ = reserve_submission(1)
    big_q, _ = reserve_submission(20)
    small, big = FakeConv(), FakeConv()
    small_q(fake_agent(api_calls=80, cost=0.80), small)   # ~19 steps of budget
    big_q(fake_agent(api_calls=80, cost=0.80), big)
    assert small.asks == []
    assert big.asks


def test_the_prompt_names_files_rather_than_staging_everything():
    """`git add -A` is what put an image's build/ tree into a prediction."""
    text = submission_prompt("/testbed")
    assert "git diff --" in text
    assert "git add -A" not in text
    assert "/testbed" in text
    for excluded in ("tests", "reproduce", "build output"):
        assert excluded in text


# --------------------------------------------------------------------------- #
#  Extraction
# --------------------------------------------------------------------------- #

def _traj(*contents: str) -> Trajectory:
    t = Trajectory()
    for c in contents:
        t.add_message("assistant", c)
    return t


def test_a_fenced_diff_is_the_submission():
    assert extract_submission(_traj(f"Here is the fix:\n```diff\n{PATCH}```")) == PATCH


def test_the_last_submission_wins():
    """If the agent submits twice, the second is its final answer."""
    first = PATCH.replace("return 2", "return 99")
    got = extract_submission(_traj(f"```diff\n{first}```", f"```diff\n{PATCH}```"))
    assert "return 2" in got and "return 99" not in got


def test_the_last_diff_within_one_message_wins():
    """A model that shows a first attempt and then corrects it in the same turn
    means the second one; taking the first would submit the version it rejected."""
    rejected = PATCH.replace("return 2", "return 99")
    got = extract_submission(_traj(
        f"First I tried:\n```diff\n{rejected}```\nThat was wrong. Actually:"
        f"\n```diff\n{PATCH}```"))
    assert "return 2" in got and "return 99" not in got


def test_an_untagged_fence_is_still_accepted():
    """Models drop the language tag often enough that refusing it would throw
    away correct answers over punctuation."""
    assert extract_submission(_traj(f"```\n{PATCH}```")) == PATCH


def test_no_diff_means_no_submission():
    """There is no fallback: an agent that did not hand anything in produces an
    empty prediction, and the exit status says why."""
    for content in ("I could not reproduce the issue.",
                    "```python\nprint('hello')\n```",
                    "Here is what I would change: edit pkg/mod.py line 2.",
                    ""):
        assert extract_submission(_traj(content)) == ""


def test_example_code_is_not_mistaken_for_a_patch():
    """Requires the markers `git apply` needs anyway -- a file header and a hunk."""
    assert extract_submission(_traj("```diff\nsome notes about a diff\n```")) == ""


def test_a_submission_is_newline_terminated():
    """`git apply` rejects a patch whose last line has no newline."""
    assert extract_submission(_traj(f"```diff\n{PATCH.rstrip()}```")).endswith("\n")


def test_tool_and_user_turns_are_ignored():
    """A diff quoted back by a tool result is not the agent's submission."""
    t = Trajectory()
    t.add_message("user", f"```diff\n{PATCH}```")
    t.add_message("tool", f"```diff\n{PATCH}```")
    assert extract_submission(t) == ""


def test_defaults_leave_room_for_the_sequence_the_prompt_asks_for():
    """Measured: every agent in a 10-instance run took four turns to comply
    (status, diff, reply, plus a narrating turn). A reserve of three was one
    short, and the single run that hit its cost ceiling died mid-submission."""
    assert DEFAULT_RESERVE_STEPS >= 5


# --------------------------------------------------------------------------- #
#  Through the real loop
# --------------------------------------------------------------------------- #
#  The fakes above prove the hooks decide correctly; these prove they are
#  actually reached, which is the part that was wrong. The first version
#  installed only the budget hook, and every test passed while the common
#  ending -- the agent stopping on its own -- produced an empty patch.

class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def _scripted_agent(monkeypatch, replies):
    """An AshAgent whose model returns `replies` in order, then repeats the last."""
    from swebench.agent import AshAgent
    from swebench.models import AgentConfig

    agent = AshAgent(AgentConfig(step_limit=20, cost_limit=1.0),
                     executor=lambda name, args: None)
    sent = []

    def fake_query(self, llm, conv, step_n):
        sent.append([dict(m) for m in conv.messages])
        self.cost.api_calls += 1
        return replies[min(step_n - 1, len(replies) - 1)]

    monkeypatch.setattr(AshAgent, "_query", fake_query)
    return agent, sent


def test_the_loop_asks_before_reporting_completed(monkeypatch):
    """End to end: agent stops, gets asked, answers, and the diff is extracted."""
    agent, sent = _scripted_agent(monkeypatch, [
        _Msg("The fix is in place; I am done."),
        _Msg("Done."),                                  # 2nd no-tool turn -> finish
        _Msg(f"Here it is:\n```diff\n{PATCH}```"),      # the answer, after the ask
    ])
    before_query, before_finish = reserve_submission()
    agent.before_query_hooks.append(before_query)
    agent.before_finish_hooks.append(before_finish)

    assert agent.run("fix the bug") == "completed"
    asked = [m for turn in sent for m in turn
             if m["role"] == "user" and "Submit now" in m["content"]]
    assert asked, "the loop reported completed without ever asking"
    assert extract_submission(agent.trajectory) == PATCH


def test_the_loop_still_finishes_when_the_agent_sends_no_diff(monkeypatch):
    """No fallback and no hang: the run ends, the prediction is simply empty."""
    agent, _ = _scripted_agent(monkeypatch, [_Msg("I could not fix it.")])
    before_query, before_finish = reserve_submission()
    agent.before_query_hooks.append(before_query)
    agent.before_finish_hooks.append(before_finish)

    assert agent.run("fix the bug") == "completed"
    assert extract_submission(agent.trajectory) == ""


def test_an_unhooked_loop_is_unchanged(monkeypatch):
    """The ask is the eval layer's choice; the default loop knows nothing of it."""
    agent, sent = _scripted_agent(monkeypatch, [_Msg("Done.")])
    assert agent.run("fix the bug") == "completed"
    assert not any("Submit now" in m["content"]
                   for turn in sent for m in turn if m["role"] == "user")


def test_the_harness_installs_both_hooks():
    """Guards the defect this file was written after: installing only the
    budget hook left the common ending unhandled."""
    import inspect
    from swebench.harnesses import litellm as harness
    source = inspect.getsource(harness)
    assert "before_query_hooks.append" in source
    assert "before_finish_hooks.append" in source
    assert "extract_submission(" in source


def test_a_reused_agent_is_asked_again_on_its_second_run(monkeypatch):
    """The loop clears hook_state per run. If it did not, an agent reused across
    instances would be asked once ever and every later run would come back empty
    -- silently, since nothing else depends on that flag."""
    agent, sent = _scripted_agent(monkeypatch, [
        _Msg("Done."), _Msg("Done."),
        _Msg(f"```diff\n{PATCH}```"),
    ])
    before_query, before_finish = reserve_submission()
    agent.before_query_hooks.append(before_query)
    agent.before_finish_hooks.append(before_finish)

    agent.run("first instance")
    assert extract_submission(agent.trajectory) == PATCH

    sent.clear()
    agent.run("second instance")
    assert extract_submission(agent.trajectory) == PATCH, \
        "second run was never asked to submit"


def test_only_the_single_agent_topologies_ask_for_a_submission():
    """A shared worktree has no single agent to ask: manager_worker and best_of_n
    run several agents in one tree, so the combined diff is the answer and
    get_patch() is right there. Documented in submission.py; asserted here so the
    boundary is not crossed by accident."""
    import inspect
    from swebench.harnesses import best_of_n, manager_worker
    for module in (best_of_n, manager_worker):
        source = inspect.getsource(module)
        assert "reserve_submission" not in source, \
            f"{module.__name__} shares one worktree; it cannot ask one agent"
        assert "get_patch()" in source


def test_an_appended_ask_is_recorded_in_the_trajectory():
    """append_to_last used to update only the model-facing copy, so a mid-run ask
    left no trace in the record the eval layer reads. Found on a real cost_limit
    run: the agent had been asked and answered, and the trajectory showed neither.
    Extraction reads the trajectory, so the submission would have been lost."""
    from swebench.agent.conversation import Conversation
    traj = Trajectory()
    conv = Conversation(traj)
    conv.add_user("fix the bug")
    conv.add_tool_result("call-1", "M src/mod.py")

    before_query, _ = reserve_submission()
    before_query(fake_agent(api_calls=98, cost=0.10), conv)

    assert "Submit now" in conv.messages[-1]["content"], "model never saw it"
    assert "Submit now" in traj.messages[-1]["content"], "record does not show it"


def test_a_budget_path_submission_survives_into_the_prediction():
    """The whole point of the fix above: an agent asked mid-run and answering must
    end up with a patch, not an empty prediction."""
    from swebench.agent.conversation import Conversation
    traj = Trajectory()
    conv = Conversation(traj)
    conv.add_user("fix the bug")
    before_query, _ = reserve_submission()
    before_query(fake_agent(api_calls=98, cost=0.10), conv)
    traj.add_message("assistant", f"```diff\n{PATCH}```")
    assert extract_submission(traj) == PATCH
