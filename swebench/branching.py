"""Branch-plan metadata shared by execution and read-only reporting."""

from __future__ import annotations

import re


BRANCH_COUNT_MODES = ("adaptive", "fixed")


def branch_count_rule(mode: str, count: int) -> str:
    if mode == "adaptive":
        return (
            f"Return at most {count} branches. {count} is an upper bound, not a quota.\n"
            "Return fewer when there are fewer useful directions; do not pad the list.\n"
            "If no useful eligible branch exists, return an empty branches list and\n"
            "explain why in synthesis. Do not invent a checkpoint or filler direction."
        )
    if mode == "fixed":
        return (
            f"Return exactly {count} branches. This is a required count.\n"
            "You choose the positions and directions, not the total number. If the\n"
            "useful work is concentrated at one point, allocate multiple grounded\n"
            "directions there; do not reduce the count because positions repeat.\n"
            "Use only eligible checkpoints and preserve the task's constraints."
        )
    raise ValueError("unknown branch count mode: %r" % mode)


def branch_run_name(round_no: int, index: int, direction: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(direction or "b").lower())[:24]
    return "r%db%d-%s" % (round_no, index, slug)


def review_branches(review: dict) -> list[dict]:
    """Read new per-branch points or historical shared-point plans."""
    return [dict(base=branch.get("base", review.get("base", "parent")),
                 branch_step=branch.get("branch_step", review.get("branch_step")),
                 why=branch.get("why", review.get("why", "")),
                 name=branch.get("name"), hint=branch.get("hint"))
            for branch in review.get("branches") or []]


def planned_branch(review: dict, run_name: str, round_no: int) -> dict | None:
    for index, branch in enumerate(review_branches(review), 1):
        if branch_run_name(round_no, index, branch.get("name")) == run_name:
            return branch
    return None
