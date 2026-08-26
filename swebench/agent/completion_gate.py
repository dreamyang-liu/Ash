"""Ask for evidence when an agent declares victory implausibly early.

Measured on a 20-task SWE-Marathon batch: eleven tasks reported `completed`
after 13 to 59 model calls out of a 2000-step budget, on work a human expert is
estimated to need 5 to 12 hours for -- and the claims were not modest. One
announced "All 68,186 golden tests pass with a 100% pass rate"; the transcript
shows that number came from counting the lines of the golden file, and the test
script had last been run three edits earlier. Nothing in the loop disagreed,
because two turns of prose is exactly what a finished agent looks like.

This is a `before_finish` hook: it fires when the loop is about to accept
completion, states what the run itself shows, and asks for verification. It
does not run the grader -- the task's verifier is the hidden test, and touching
it here would be cheating. Everything the challenge says is drawn from the
transcript, and it fires a bounded number of times so a genuinely finished
agent still finishes.
"""

from typing import Any, Optional

#: Fraction of the step budget below which a completion claim is challenged.
#: Above it, the agent has spent real effort and its judgement stands.
DEFAULT_SUSPICION_FRACTION = 0.25

#: How many times one run may be challenged. More than this is nagging: if the
#: agent holds its position twice, the disagreement is not going to be resolved
#: by asking again.
DEFAULT_MAX_CHALLENGES = 2

#: Commands that look like verification rather than exploration or editing.
_VERIFY_HINTS = ("test", "pytest", "cargo test", "make", "run_tests", "verify",
                 "npm test", "go test", "./gradlew", "bench", "check")


def _looks_like_verification(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _VERIFY_HINTS)


def _last_verification_gap(conv) -> Optional[int]:
    """How many assistant turns have passed since a verification-ish command.

    None when no such command was ever run, which is the strongest form of the
    same problem.
    """
    turns_since = 0
    for message in reversed(conv.messages):
        if message.get("role") == "assistant":
            turns_since += 1
            for call in (message.get("tool_calls") or []):
                function = call.get("function") or {}
                if _looks_like_verification(str(function.get("arguments") or "")):
                    return turns_since
    return None


def make_completion_challenge(
        *,
        step_limit: int,
        expert_hours: Any = None,
        suspicion_fraction: float = DEFAULT_SUSPICION_FRACTION,
        max_challenges: int = DEFAULT_MAX_CHALLENGES,
):
    """A `before_finish` hook that asks for evidence, then gets out of the way.

    Returns True to give the agent another turn, which is how the loop's
    finish hooks request continuation.
    """
    state = {"challenges": 0}

    def challenge(agent, conv) -> bool:
        used = agent.cost.api_calls
        budget = max(1, step_limit)
        if used >= budget * suspicion_fraction:
            return False
        if state["challenges"] >= max_challenges:
            return False
        state["challenges"] += 1

        gap = _last_verification_gap(conv)
        if gap is None:
            evidence = ("You have not run any build or test command in this "
                        "session, so nothing has confirmed the work.")
        elif gap > 2:
            evidence = (f"The last build or test command you ran was {gap} "
                        "turns ago, before your most recent changes, so its "
                        "result does not describe the current state.")
        else:
            evidence = ("Re-check that the command you ran actually exercises "
                        "the requirement, and that you read its full output "
                        "rather than its exit status.")

        estimate = ""
        try:
            if expert_hours:
                estimate = (f" A human expert is estimated to need "
                            f"{expert_hours} hours for this task.")
        except Exception:
            estimate = ""

        conv.add_user(
            f"Before this is accepted as complete: you have used {used} of "
            f"{step_limit} available steps.{estimate} {evidence}\n\n"
            "Verify your claim now, with commands whose output you show: "
            "build the project, run whatever tests or checks it provides, and "
            "exercise the specific behaviour the task asks for. If the "
            "verification passes, say so and stop. If it does not, keep "
            "working -- you have budget remaining. Do not restate that the "
            "work is complete without evidence from this session.")
        agent._trace(f"\n[GATE] completion challenged at step {used} "
                     f"({state['challenges']}/{max_challenges})\n")
        return True

    return challenge
