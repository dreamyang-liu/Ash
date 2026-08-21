"""Asking the agent to hand in its answer — the eval layer's business.

What counts as an answer is defined by whoever is grading. For SWE-bench that is
a diff of the source files the agent changed, and the agent is the only party that
knows which those are: it wrote them. A harness inferring the set from git state
guesses, and guessed wrong -- a `psf/requests` prediction once came back as 66
files because the SWE-bench image ships an untracked `build/` tree.

So the agent produces the patch, and this module is where the eval layer reserves
the room for it. Three pieces:

    submission_prompt()   what to ask for, in the grader's terms
    reserve_submission()  the hooks that ask once, near the end
    extract_submission()  the diff the agent handed in, or nothing

The reservation matters because a run can end without the agent getting a turn to
submit, and it ends two ways. A budget ceiling is predictable: of the four exit
statuses, three -- ``cost_limit``, ``step_limit``, ``error`` -- stop the loop
between turns. One real run stopped at 59 of 100 steps because it had spent $3.07
of a $3.00 budget, so reserving *steps* alone would not have saved it; the reserve
is measured in whichever ceiling binds first (``CostTracker.steps_left``).

The other way is the agent deciding it is done, which cannot be seen coming and is
the common case -- 24 of 25 recorded runs ended that way. So the ask hangs off both
exits. Reserving only against the budget passed every unit test while producing an
empty patch for nearly every successful run.

Deliberately no fallback. If the agent never submits, the prediction is empty and
the exit status says why. Extracting a patch anyway would put our guess about the
agent's work back into the answer, and an agent that ran out of budget mid-edit
usually has nothing worth grading. An empty prediction is the honest report.

Who uses this
-------------
The one-agent-one-tree topologies, where "which files did you change" has a single
answer to ask for: ``harnesses/litellm.py`` and ``runner.py``.

Not ``manager_worker`` or ``best_of_n``. Several agents share one worktree there,
so no single agent can report the change set -- the combined diff *is* the answer,
and ``session.get_patch()`` reads it from the tree that holds it. Not
``rollout_server`` either: it grades the live tree by running tests in it, so its
``get_patch()`` call is a gate ("is there anything to grade?") rather than the
answer being graded. Three call sites, three different questions, which is why
this module is imported by two of them rather than replacing the other two.
"""

from __future__ import annotations

import re

#: Where the agent leaves its answer. The key already existed in
#: ``Trajectory.info``; it is now the agent's to fill rather than the harness's.
SUBMISSION_KEY = "submission"

#: A fenced diff in the agent's reply. ```diff is what the prompt asks for;
#: bare ``` is accepted too, since models drop the language tag often enough that
#: refusing it would throw away correct answers over punctuation.
_FENCED = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.S)

#: Steps held back for handing in the answer.
#:
#: Measured, not guessed: across a 10-instance run every agent took exactly four
#: turns to comply -- `git status`, `git diff`, then the reply, with the model
#: narrating a turn in between. A reserve of three was one short and the one run
#: that reached its cost ceiling died mid-submission with a $3.07 patch of nothing,
#: while the nine that finished early had turns to spare. Five buys the margin for
#: a step lost to a malformed tool call; the cost of over-reserving is one unused
#: turn, and of under-reserving, the whole run.
DEFAULT_RESERVE_STEPS = 5


def submission_prompt(workdir: str = "/testbed") -> str:
    """The instruction that turns the agent's work into a prediction.

    Names the files explicitly rather than using ``git add -A``: the diff should
    hold the fix, not a reproduce script or a `build/` tree that came with the
    image. `git diff` without staging keeps this read-only, so asking does not
    disturb the tree a later step might still want to inspect.
    """
    return (
        "\n\n[Submit now] Your budget is nearly spent, so hand in your work"
        " before it runs out.\n"
        f"1. Run `cd {workdir} && git status --porcelain` to see what changed.\n"
        "2. Run `cd %s && git diff -- <the source files you fixed>` and nothing"
        " else: not tests, not reproduce scripts, not build output.\n"
        "3. Reply with that diff verbatim, wrapped in a ```diff code block, and"
        " stop. A new file that the fix needs is part of the answer; include it"
        " with `git diff --no-index /dev/null <path>`.\n"
        "If you have no working fix, say so instead of sending an unfinished"
        " diff." % workdir
    )


def extract_submission(trajectory) -> str:
    """The diff the agent handed in, or ``""`` if it never did.

    Reads the assistant turns back to front: if the agent submitted more than
    once, the last one is its final answer. Nothing is reconstructed -- a reply
    with no diff in it means no submission, which is a result rather than a
    problem to work around.
    """
    for message in reversed(getattr(trajectory, "messages", [])):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        for block in reversed(_FENCED.findall(content)):
            if _looks_like_a_diff(block):
                return block if block.endswith("\n") else block + "\n"
    return ""


def _looks_like_a_diff(text: str) -> bool:
    """Whether a fenced block is a patch rather than prose or example code.

    Requires the two markers `git apply` needs anyway: a file header and a hunk.
    A block that fails this would fail to apply, so accepting it would only turn
    "no submission" into "a prediction that errors out during grading".
    """
    return ("diff --git" in text or "--- " in text) and "@@" in text


#: Where the hooks record that the question has been asked. Lives in
#: ``agent.hook_state``, which the loop clears per run.
_ASKED = "submission_asked"


def reserve_submission(reserve_steps: int = DEFAULT_RESERVE_STEPS,
                       workdir: str = "/testbed"):
    """Hooks that ask for the answer exactly once, whichever way the run ends.

    Returns ``(before_query, before_finish)``, because a run can end two ways and
    only one of them can be seen coming:

    * the budget runs out -- predictable, so ``before_query`` asks while there
      are still ``reserve_steps`` turns left to answer in;
    * the agent decides it is done -- not predictable, and the common case (24 of
      25 recorded runs), so ``before_finish`` asks and buys one more turn.

    Installing only the first would leave an empty prediction for nearly every
    successful run, which is how this was found. They share ``hook_state`` so a
    run that hits both paths is still asked once: an agent that already submitted
    and then stopped is finished, and re-asking would spend the reserve it was
    given.

    Hooks are built here rather than being methods so the agent loop stays
    unaware that "submitting" is a thing -- it knows only that it runs hooks.
    """
    def ask(agent, conv, add) -> bool:
        if agent.hook_state.get(_ASKED):
            return False
        agent.hook_state[_ASKED] = True
        prompt = submission_prompt(workdir)
        add(prompt)
        agent._trace(prompt)
        return True

    def before_query(agent, conv) -> None:
        left = agent.cost.steps_left(agent.config.step_limit,
                                     agent.config.cost_limit)
        if left <= reserve_steps:
            # Appended to the last user/tool message: mid-run the conversation
            # ends with a tool result, and a bare user turn after one would
            # orphan the tool call.
            ask(agent, conv, conv.append_to_last)

    def before_finish(agent, conv) -> bool:
        # A fresh user turn, not append_to_last: here the last message is the
        # assistant's "I am done", so appending would bury the request in an
        # earlier turn the model has already answered.
        return ask(agent, conv, conv.add_user)

    return before_query, before_finish
