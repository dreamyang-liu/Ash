#!/usr/bin/env python3
"""Generate docs/branching-flow.excalidraw — two rounds of fork_eval, as an
information-flow diagram.

The question this diagram answers is not "what runs" but "who knows what, when":
what the analyst reads in each round, what it decides, and what memory a branch
is born with. The load-bearing asymmetries it must show:

- the analyst reads the FAILED attempt's transcript + the grading verdict the
  attempt itself could never see;
- one branch step per round, K divergent hints;
- a branch inherits the parent's CONVERSATION up to the fork step (fork=True)
  plus the environment snapshot of that step -- and its own journal starts
  empty, so round 2's analyst sees only the winner's own steps;
- round 2's genuinely new information is the notes ledger: every round-1
  direction and how it graded, fed back as "do not repeat".

    python3 docs/gen_branching_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exgen import (BLUE, GREEN, GREY, ORANGE, RED, VIOLET, WHITE,  # noqa: E402
                   Canvas, text_size)

OUT = Path(__file__).resolve().parent / "branching-flow.excalidraw"

X0 = 40


def right_arrow(c: Canvas, id_: str, x_from: float, x_to: float, y: float,
                *, color: str = WHITE, dashed: bool = False) -> None:
    c.arrow(id_, x_from + 4, y, [[0, 0], [x_to - x_from - 8, 0]],
            color=color, dashed=dashed)


def down_arrow(c: Canvas, id_: str, x: float, y_from: float, y_to: float,
               *, color: str = WHITE, dashed: bool = False) -> None:
    c.arrow(id_, x, y_from + 4, [[0, 0], [0, y_to - y_from - 8]],
            color=color, dashed=dashed)


def main() -> None:
    c = Canvas()
    y = 30
    c.text("title", X0, y,
           "fork_eval branching: who knows what, when  (--rounds 2 --branches 4,3)",
           size=17)
    y += 46

    # ---- phase 0: the parent -------------------------------------------------
    parent = c.panel(
        "parent", X0, y, 330, "PARENT attempt",
        "fresh microVM from the instance image\n"
        "conversation: task prompt only (problem + workflow + tool primer)\n"
        "every exec step i leaves a pair:  (snapshot_i , session-ckpt_i)\n"
        "steps 1..N recorded in parent.jsonl",
        color=BLUE)
    grade0 = c.panel(
        "grade0", parent["right"] + 90, y, 300, "GRADE (outside the agent)",
        "restore snapshot_N into a NEW microVM\n"
        "revert any test-file edits, apply official test_patch\n"
        "FAIL_TO_PASS first, then PASS_TO_PASS\n"
        "verdict: summary + BROKEN test names + patch + output",
        color=RED)
    right_arrow(c, "a-p-g", parent["right"], grade0["x"],
                parent["y"] + 30)
    c.text("a-p-g-l", parent["right"] + 14, parent["y"] + 8,
           "last snapshot", size=11, color=GREY)
    y = max(parent["bottom"], grade0["bottom"]) + 44

    # ---- round 1: what the analyst reads --------------------------------------
    c.text("r1-title", X0, y, "ROUND 1  (width 4)", size=15, color=ORANGE)
    y += 30
    analyst1 = c.panel(
        "analyst1", X0, y, 380, "ANALYST reads (one LLM call, no tools)",
        "1. problem statement\n"
        "2. the VERDICT -- ground truth the parent never saw:\n"
        "   which target test failed / which regressions BROKE\n"
        "3. parent transcript: one line per step, \"[i] tool(args) -> result\"\n"
        "   (100k-token budget, results kept head AND tail)\n"
        "4. notes: (empty -- nothing tried yet)",
        color=ORANGE)
    decide1 = c.panel(
        "decide1", analyst1["right"] + 90, y, 320, "ANALYST decides (JSON)",
        "branch_step = s     one step for the whole round,\n"
        "                    late enough to keep work, strictly\n"
        "                    BEFORE the decisive wrong turn\n"
        "what_went_wrong     the diagnosis\n"
        "4 divergent hints   fix-forward / revert-and-redo /\n"
        "                    narrower change / different verification",
        color=ORANGE)
    right_arrow(c, "a-a1-d1", analyst1["right"], decide1["x"],
                analyst1["y"] + 30, color=ORANGE)
    y = max(analyst1["bottom"], decide1["bottom"]) + 44

    # ---- round 1: the branches -------------------------------------------------
    fork1 = c.panel(
        "fork1", X0, y, 360, "fork_plan(parent.jsonl, s)",
        "snapshot_s   = the disk exactly as it stood AFTER step s\n"
        "session_s    = parent's conversation reference at step s",
        color=VIOLET)
    y = fork1["bottom"] + 40

    branch_body = (
        "born with:\n"
        "  env    = snapshot_s  (own microVM -- siblings\n"
        "           cannot contaminate each other)\n"
        "  memory = parent conversation UP TO step s\n"
        "           (fork=True: it remembers steps 1..s)\n"
        "  + one NEW user message: verdict + ITS hint\n"
        "    (opens with git diff to ground in the disk)\n"
        "journal: starts EMPTY -- its own steps only")
    bw = text_size(branch_body, 11)[0] + 24
    branches1 = []
    bx = X0
    for k in range(1, 5):
        b = c.panel("r1b%d" % k, bx, y, bw, "r1b%d" % k, branch_body,
                    color=GREEN, body_size=11)
        branches1.append(b)
        bx = b["right"] + 26
        down_arrow(c, "a-f1-b%d" % k, b["cx"], fork1["bottom"], b["y"],
                   color=VIOLET)
    y = branches1[0]["bottom"] + 44

    pick1 = c.panel(
        "pick1", X0, y, 420, "each branch GRADED; winner picked",
        "score: 3 resolved | 2 target ok, regressions broke | 1 patch | 0\n"
        "any score-3 branch  ->  STOP, resolved\n"
        "else best = round-1 winner if winner.score >= best.score\n"
        "notes += \"round 1 <name>: <hint> -> <grade>\"  for ALL 4",
        color=GREEN)
    for k, b in enumerate(branches1, 1):
        down_arrow(c, "a-b%d-p1" % k, b["cx"], branches1[0]["bottom"],
                   pick1["y"], color=GREEN)
    y = pick1["bottom"] + 48

    # ---- round 2 ---------------------------------------------------------------
    c.text("r2-title", X0, y, "ROUND 2  (width 3) -- only if round 1 resolved nothing",
           size=15, color=ORANGE)
    y += 30
    analyst2 = c.panel(
        "analyst2", X0, y, 400, "ANALYST reads -- what is NEW",
        "1. problem (same)\n"
        "2. the WINNER's verdict  (not the parent's)\n"
        "3. the WINNER's transcript -- only ITS OWN steps 1..M;\n"
        "   the parent's steps are not in this journal\n"
        "4. notes: all four round-1 directions AND how each graded\n"
        "   -> \"do not repeat\" is now load-bearing",
        color=ORANGE)
    decide2 = c.panel(
        "decide2", analyst2["right"] + 90, y, 320, "ANALYST decides",
        "branch_step = m   on the WINNER's journal\n"
        "3 new hints, disjoint from everything in notes",
        color=ORANGE)
    right_arrow(c, "a-a2-d2", analyst2["right"], decide2["x"],
                analyst2["y"] + 30, color=ORANGE)
    y = max(analyst2["bottom"], decide2["bottom"]) + 44

    branch2 = c.panel(
        "r2b", X0, y, 430, "r2b1..r2b3  born with",
        "env    = winner's snapshot_m\n"
        "memory = parent[1..s] + winner[1..m]   (forks compose:\n"
        "         the winner's session already contained the parent's)\n"
        "+ new user message: winner's verdict + its round-2 hint",
        color=GREEN)
    stop = c.panel(
        "stop", branch2["right"] + 90, y, 300, "grade; stop",
        "any resolved -> report RESOLVED by <name>\n"
        "else out of rounds: best attempt stands,\n"
        "every journal + snapshot kept for later",
        color=RED)
    right_arrow(c, "a-b2-s", branch2["right"], stop["x"], branch2["y"] + 26,
                color=GREEN)

    # ---- the two long arrows that carry state across phases -------------------
    down_arrow(c, "a-g0-a1", grade0["x"] + 40, grade0["bottom"],
               analyst1["y"], color=RED, dashed=True)
    c.text("a-g0-a1-l", grade0["x"] + 50, grade0["bottom"] + 8,
           "verdict", size=11, color=RED)
    down_arrow(c, "a-d1-f1", decide1["x"] + 40, decide1["bottom"],
               fork1["y"], color=VIOLET, dashed=True)
    c.text("a-d1-f1-l", decide1["x"] + 50, decide1["bottom"] + 8,
           "step s", size=11, color=VIOLET)
    down_arrow(c, "a-p1-a2", pick1["x"] + 40, pick1["bottom"],
               analyst2["y"] - 32, color=GREEN, dashed=True)
    c.text("a-p1-a2-l", pick1["x"] + 50, pick1["bottom"] + 8,
           "winner journal + notes", size=11, color=GREEN)
    down_arrow(c, "a-d2-b2", decide2["x"] + 40, decide2["bottom"],
               branch2["y"], color=VIOLET, dashed=True)
    c.text("a-d2-b2-l", decide2["x"] + 50, decide2["bottom"] + 8,
           "step m", size=11, color=VIOLET)

    c.write(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
