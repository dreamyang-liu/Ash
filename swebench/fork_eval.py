#!/usr/bin/env python3
"""Run one SWE-bench instance through the orchestrator, grade it, branch on failure.

The eval driver for the checkpointed path: it is the piece that turns "we can
snapshot every step and fork any step" into a *score*, by adding the two things
the execution plane deliberately does not know -- what the answer is (a patch
that makes FAIL_TO_PASS pass without breaking PASS_TO_PASS) and what to do when
the answer is wrong.

    python -m swebench.fork_eval --instance sympy__sympy-13091 \
        --slot codex --model openai.gpt-5.6-luna \
        --rounds 2 --branches 3 --fork-full-conversation -o runs/fork-eval

The loop:

1. **Attempt.** One orchestrator run. Every mutating step leaves a rollback pair
   (env snapshot + conversation ref) in the journal.
2. **Grade.** Restore the LAST snapshot into a fresh microVM and run the tests
   there. Grading in a restored sandbox rather than the live one is deliberate:
   it proves the snapshot carries the work, and it lets grading happen after the
   agent's sandbox is gone.
3. **Branch on failure**, in two analyst stages. *Map:* every failed attempt is
   analysed separately -- its transcript and verdict, without its supplied hint --
   into a failure_reason, a lesson, and candidate branch steps. *Reduce:* a
   reviewer reads ALL the analyses and follows the configured branch count rule.
   Each branch has its own base, step and direction; locations may repeat.
   Each becomes an attempt whose
   sandbox image IS that step's snapshot and whose conversation forks the
   base's. The reviewer may go BACK to an earlier attempt when later ones are
   deeper in a dead end -- the escape hatch winner-take-all lacked.
4. Repeat until something resolves or ``--rounds`` is spent.

Why the analyst sees the *verdict* and not just the transcript: on a benchmark the
agent usually believes it succeeded, so "what went wrong" is only answerable from
outside. Which failing test, and whether the failure is the target test or a
regression, is the single most useful bit we can give it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from harness.core.guidance import render_branch_note
from swebench.test_environment import CONDA_INIT, test_command, validate_test_execution
from harness.core.journal import read_journal, volatile_reason
from harness.orchestrator.run import Orchestrator, RunOutcome, RunSpec
from harness.rollback import Checkpoint, branch_checkpoints, load_checkpoints, turn_branch_checkpoints
from harness.normalize.claude_turns import completed_turn_steps
from harness.execution.backends import with_sandbox_budget
from harness.slots.claude_history import PrefixSource, find_prefix_source, prepare_prefix
from swebench.branch_plan import ReviewerPlanError, extract_branch_plan, review_with_feedback
from swebench.branching import BRANCH_COUNT_MODES, branch_count_rule, branch_run_name
from swebench.assistant_branch import (
    ASSISTANT_REVIEW_PROMPT, POINT_REVIEW_PROMPT, BRANCH_GUIDANCE_MODES, require_mini_parent,
    actor_tools_at, reviewer_context, selected_prefix, validate_guidance,
)
from harness.core.assistant_turn import validate_assistant_turn
from swebench.dataset import (SYMPY_RUNNER, build_batch_test_command,
                              load_instances, malformed_test_ids,
                              needs_file_runner, parse_test_list, resolve_image,
                              sympy_runner_spec, test_files_of)

#: Both analyst endpoints on Bedrock, keyed by what the model name says it is.
#: One JSON call needs no agent scaffolding, so neither goes through a translator.
#: Mantle serves the OpenAI catalogue (Responses shape); Converse serves
#: Anthropic's and everything else Bedrock hosts.
MANTLE = "https://bedrock-mantle.%s.api.aws/openai/v1/responses"
CONVERSE = "https://bedrock-runtime.%s.amazonaws.com/model/%s/converse"


# --- the agent's task ------------------------------------------------------
#
# The shape borrows what worked in mini-swe-agent's task prompt -- a numbered
# reproduce-first workflow, environment quirks stated plainly instead of learned
# by failing, and concrete tool invocations to copy -- re-grounded on this
# panel's two tools. The old prompt named the tools and the rules but taught
# nothing, so the model spent its first turns rediscovering the environment.

TOOL_PRIMER = """\
## Your tools, and the quirks that will bite you

You have exactly two tools, served over MCP as `shell` and `text_editor`. Your \
own built-in tools (Bash, Read, Edit, ...) are disabled -- they cannot see the \
repository.

**Every `shell` call runs in a FRESH process.** `cd` and environment variables \
do not survive to the next call. Pass `working_dir` instead of cd, and set \
`timeout` (seconds) for long test runs -- the default is too short for a full \
suite. `tail` limits how much output comes back; use it when a test run is \
noisy.

    shell {"command": "python -m pytest tests/test_x.py -x -q",
           "working_dir": "/testbed", "timeout": 600, "tail": 50}

**`text_editor` is for reading and precise edits:**

    view        {"command": "view", "path": "/testbed/pkg/mod.py",
                 "view_range": [80, 140]}       # numbered lines
    str_replace {"command": "str_replace", "path": "...",
                 "old_str": "...", "new_str": "..."}
                # old_str must match EXACTLY ONCE, whitespace included --
                # include enough surrounding lines to make it unique
    insert      {"command": "insert", "path": "...",
                 "insert_line": 42, "insert_text": "..."}
    write       {"command": "write", "path": "...", "file_text": "..."}
                # whole-file write; for NEW files (e.g. a repro script)
"""

SHELL_TOOL_PRIMER = """\
You have one sandbox tool, served over MCP as `shell`. Use it for searching,
reading, editing files, and running tests. Your built-in tools are disabled.
Each shell call starts a fresh process: set working_dir when needed and use
timeout (seconds) for long commands. Use tail to limit noisy output.

    shell {"command": "python -m pytest tests/test_x.py -x -q",
           "working_dir": "/testbed", "timeout": 600, "tail": 50}
"""


def tool_primer(slot: str = "", workdir: str = "/testbed") -> str:
    if slot == "mini-swe-agent":
        from harness.core.mini_tools import mini_tool_primer

        return mini_tool_primer(workdir)
    primer = SHELL_TOOL_PRIMER if slot == "claude-code" else TOOL_PRIMER
    return primer.replace("/testbed", workdir)


PROMPT = """\
You are fixing a bug in the {repo} repository, checked out at /testbed inside \
your sandbox.

## Problem statement
{problem}

## Recommended workflow

Work step by step; run something after every change so a mistake surfaces while \
it is still one edit deep.

1. Explore the codebase and READ the code paths the problem statement names.
2. Write a small script (e.g. /testbed/repro.py) that reproduces the issue, and \
run it to confirm it fails the way the report says.
3. Edit the source to fix the root cause -- minimal, targeted change; no \
sweeping refactors.
4. Re-run your script and confirm the fix.
5. Probe edge cases around the fix (empty input, negatives, the other code \
paths through the changed function) and handle what breaks.
6. Run the project's own tests covering this area, and iterate until they pass.

## Rules, and why

- Do NOT edit test files. Grading restores the official tests and DISCARDS your \
edits to them -- changing a test cannot make you pass and wastes your time.
- Only changes under /testbed count. Do not create files outside it except \
throwaway scripts.
- A wrong-but-plausible fix that passes one test still fails the hidden \
regression suite: fix causes, not symptoms.

{primer}"""

# --- analysts: one per failed case, then one review over all of them --------
#
# Two stages on purpose. A single analyst reading only the best attempt threw
# away the losing branches' trajectories -- round 2 used to receive one 120-char
# line per sibling. Now every failed attempt gets its own analysis (map), and a
# review agent reads ALL of them -- parent included -- to pick where the next
# round starts (reduce). The reviewer may choose ANY earlier attempt as the
# base, which is also the escape hatch the old winner-take-all flow lacked:
# when the round-1 winner is a deeper dead end, the reviewer can go back to the
# parent.

_CASE_PROMPT = """\
You are analysing ONE failed coding attempt. Your private analysis will be
pooled with other attempts so a reviewer can choose a checkpoint and useful
directions for continuing from it. The continuing agent remembers only the
conversation up to that checkpoint.

The goal is a natural and effective continuation: if the added direction is
later removed, the next reasoning and actions should still make sense from the
retained work and the agent's own observations. Diagnose the missed code-level
connection that could plausibly occur to the agent at that point.

## The problem it was solving
{problem}

## Private grading evidence (not observed by the continuing agent)
{verdict}

## The attempt, one line per tool step ("[N] tool(args) -> result")
{transcript}

## Recorded snapshot/session pairs available for selection
{checkpoint_steps}

## Translate evidence into repair directions
- Your output should help the reviewer choose a repair direction, not reproduce
  the grading report. In every returned field, omit verifier-only filenames,
  test names/IDs and grader paths. Translate their useful information into the
  affected program behavior, code relationship or invariant instead.
- A path in verifier output is not evidence that the actor can open it. Refer
  to a repository file or symbol only when the task or retained prefix
  establishes it is available at that candidate checkpoint. Do not ask the
  actor to read, locate or recreate a hidden test, or retrieve a grading artifact.
  Suggest inspecting accessible implementation/callers or constructing a small
  local check from the public task and available code instead.
- If the cause is established, state the correction directly and explain the
  relevant constraint. Otherwise give the strongest supported hypothesis: what
  code relationship may be involved, why it is plausible, what remains unknown,
  and one accessible check with what would support or rule it out. If no specific
  cause is supported, say the cause is unresolved and identify the nearest
  evidence-backed area to investigate, rather than inventing a precise fix.
- Do not merely add 'possibly' to an invented explanation. In particular, do
  not guess how an unseen test constructs its fixtures, router groups or inputs.
  A symptom does not establish those details. Prefer a check that distinguishes
  plausible mechanisms before recommending a behavior-changing edit.
- Keep private observations distinct from what the actor already knows. In
  lesson and each candidate's why, express a code-level direction and its
  connection to the retained work, not instructions to consult external evidence.

Example of translating a route-lookup failure into a qualified direction:
"The route prefix may be composed inconsistently between the registration helper
and its caller. Trace the group prefix and route suffix in the available code,
then compare the resulting URL with the task's required endpoint. Check for a
missing or duplicated segment before changing registration. The evidence does
not yet establish whether a caller-owned or handler-owned prefix is appropriate."
This is an example of reasoning, not a diagnosis to reuse for unrelated tasks.

## Evidence and output rules
- Use the verdict and the whole transcript as private diagnostic evidence.
  failure_reason and lesson are controller-only reports, not actor messages.
  Retain the specific mechanism, useful correction and important constraints;
  do not dilute a diagnosis into generic advice to inspect code or run tests.
- Separate observed facts from inferences. Distinguish the required behavior,
  a suspected cause, and a possible repair. An external failure can motivate
  a hypothesis without proving its cause. Do not change a behavioral contract
  simply because another implementation would be easier to check.
- For each candidate, identify the code, decision or observation that connects
  the retained work to the next useful question. Explain what a concrete action
  would establish, without assuming the agent has seen the discarded suffix.
- Preserve behavior outside the suspected bug. Identify relevant accepted
  inputs, boundary cases or API responsibilities that a repair must retain.
  Express the issue in program behavior rather than relying on test labels.
- Do not compose the continuing agent's inner monologue or invent observations.
  The reviewer needs a useful lead, not text the agent must repeat or a story
  about receiving feedback.

## Checkpoint rules
- Tool steps span {lo}..{hi}, but only the recorded steps listed above have
  eligible snapshot/session pairs. Choose candidates from that list; a matching
  native conversation cut is checked separately before launch. A branch at
  step N must resume the environment AS IT STOOD AFTER step N.
- branch_candidates: up to {candidate_limit} useful steps, not a quota. LATER IS BETTER when it preserves
  sound work, but choose before the decisive wrong turn. If the approach was
  already wrong at the start, propose the earliest eligible point and explain why.
- Describe salvage relative to the state after each candidate step, not the
  final tree. Later edits, files, dependencies and observations are not already
  present at an earlier checkpoint.
- Each candidate's why should name the prefix connection and the next useful
  check. Avoid requiring the continuation to redo reasoning or checks that
  the retained prefix already established.

Return ONLY a JSON object, no prose:
{{"failure_reason": "<observed behavior; supported mechanism or hypothesis; remaining uncertainty>",
  "lesson": "<concrete repair direction or discriminating check; constraints to preserve>",
  "salvage": "<work present at the candidate steps, or 'nothing'>",
  "branch_candidates": [{{"step": <int>, "why": "<visible prefix connection; next check and what it distinguishes>"}}]}}
"""

_REVIEW_PROMPT = """\
You are the REVIEWER for a failed coding task. Use the pooled analyses to
choose branches following the count requirement below, each with its own checkpoint and direction.
Each agent will have the selected conversation prefix and restored filesystem,
not the other attempts or the discarded suffix.

Your priority is natural and effective continuation. An added direction may
later be removed from the recorded conversation: the agent's subsequent
reasoning should then still read as a plausible next thought about its task,
not a response to an invisible reminder or outside report.

## The problem
{problem}

## Every attempt so far ("parent" is the original; branches were given a hint)
{reports}

## Branch count requirement
{count_rule}

## Selection rules
- For EACH branch, choose its own base attempt and branch_step. Select a step
  from that base's available_steps, using its branch_candidates as diagnostic
  suggestions rather than a mandatory list. The branch inherits that base's
  state AFTER branch_step. Never select a missing or failed checkpoint.
- An earlier attempt, including parent, can be preferable to a deeper dead end.
  Preserve useful work and distinguish prefix evidence from later discoveries.
- Positions do NOT need to be distinct. Several useful directions may share
  the same base and step, or use different positions or bases. Let the evidence
  decide; do not spread branches artificially. Vary useful hypotheses or repair
  strategies, not just the wording. Use prior hint_given records to avoid an
  already-exhausted direction, not to force unrelated new directions.

## Hint and output rules
- Each branches[].hint is delivered VERBATIM to the agent, with no diagnostic
  report or later rewriting stage. Write the final actor-facing direction now.
  The analyses and grades are private diagnostic evidence, not text to forward.
- Start with the specific code-level connection worth pursuing from that
  checkpoint: a missed case, an invariant, a suspicious operation or a candidate
  correction with its reason. Prefer 2-5 concise sentences. Make it useful
  enough to change the next action, not a generic "inspect more carefully".
- Anchor the direction in the task and the selected prefix. Say work is already
  present or already verified only when that prefix supports it. A later
  discovery can suggest what to examine, not become a fabricated prior fact.
- Preserve the intended behavior and technical constraints. Do not make a
  precise diagnosis vague for the sake of smoother prose, turn a chosen
  contract into competing alternatives, or replace it with an easier check.
  A concrete repair suggestion is welcome when supported; an uncertain cause
  should remain a hypothesis the agent can check through its own tools.
- Include a small discriminating case or boundary check when useful. Use the
  real implementation and its caller, preserve already-supported behavior,
  and build on existing checks rather than demanding a fresh ritual every time.
- In branches[].hint, do not include test names or IDs, grader paths,
  pass/fail counts, scores, raw verifier output, or final patch excerpts.
  Do not refer to reviewers, other branches, future failures or external
  feedback. Keep diagnostic specificity by describing the behavior itself.
- Do not label the text as a reminder, review or feedback; ask for an
  acknowledgment; restate the task; or give a long reset/checklist. Do not
  suggest openings such as "let me read the reminder" or "according to the
  feedback". The next response should concern the code, not receiving advice.
- Do not fabricate the agent's reasoning, prescribe first-person self-talk,
  claim it independently discovered something, or invent tool observations.
  Naturalness comes from a good connection to the retained work.
- Before returning, mentally remove the hint and consider the likely next
  response. Would the next question or action still make sense from the prefix,
  and could its conclusions be supported by the agent's observations? Also
  check that the useful diagnosis and behavioral constraints are still intact.
  Simply omitting the word "hint" is not enough.

Example of a code-level lead (adapt the reasoning, not the wording):
"The child still needs every input spelling its deserializer accepts. Check
whether isolating the flattened child's keys drops aliases or passes unrelated
parent keys through; a small parent/child case can distinguish those failures."

Return ONLY a JSON object, no prose:
{{"synthesis": "<the pooled diagnosis and why this allocation>",
  "branches": [{{"name": "<slug>", "base": "<attempt name>",
                 "branch_step": <int>, "why": "<why this point and direction>",
                 "hint": "<concise code-level lead>"}}]}}
"""

#: A diagnosis plus several self-contained branch directions may not fit in
#: 4k, and a truncated JSON object
#: fails to parse -- losing the whole round.
ANALYST_MAX_TOKENS = 32_000


def ask_analyst(model: str, prompt: str, region: str = "us-west-2",
                timeout: float = 300.0) -> str:
    """One analyst call. The endpoint follows from the model name.

    ``openai.*`` is Mantle's catalogue and speaks Responses; everything else is
    asked through Converse, which is how Bedrock serves Anthropic's models. The
    alternative -- one protocol plus a translator -- buys nothing here: this is a
    single request with no tools and no streaming.
    """
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not key:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK is required for the analyst")
    headers = {"Authorization": "Bearer %s" % key,
               "Content-Type": "application/json"}

    if model.startswith("openai."):
        body = json.dumps({"model": model, "input": prompt,
                           "max_output_tokens": ANALYST_MAX_TOKENS}).encode()
        request = urllib.request.Request(MANTLE % region, data=body,
                                        headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        chunks = []
        for item in payload.get("output") or []:
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text"):
                    chunks.append(part.get("text") or "")
        return "\n".join(chunks)

    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": ANALYST_MAX_TOKENS},
    }).encode()
    request = urllib.request.Request(CONVERSE % (region, model), data=body,
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    message = (payload.get("output") or {}).get("message") or {}
    return "\n".join(part.get("text") or ""
                     for part in (message.get("content") or []))


def extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("analyst returned no JSON: %r" % text[:200])
        candidate = text[start:end + 1]
    # strict=False: long hints arrive with literal newlines inside strings --
    # invalid JSON, harmless intent.
    return json.loads(candidate, strict=False)


# --- journal -> transcript --------------------------------------------------
#: Roughly how many characters fit in a token here. Transcripts are JSON, code
#: and test output, which tokenize worse than prose -- 3.2 is the pessimistic end
#: of what was measured on these journals, so a token budget converts to a
#: character budget that will not overshoot.
CHARS_PER_TOKEN = 3.2

#: Per-line caps. Generous on purpose, and measured: tool RESULTS on a real run
#: have a median of ~1.9k characters and a max of ~17k, so the old 300-character
#: cap fed the analyst the first two lines of every test run and threw away the
#: assertion that explains the failure. Arguments are small (median ~112) -- the
#: one that matters is a `str_replace` payload, which is worth showing whole.
RESULT_CHARS = 6000
ARG_CHARS = 4000
MESSAGE_CHARS = 2000


def render_transcript(journal_path, token_budget: int = 100_000
                      ) -> "tuple[str, int, int]":
    """One line per tool step, numbered by the checkpoint step they map to.

    Numbered from the tool calls rather than from the checkpoints, because the
    analyst must name a step the *snapshot map* has -- and both are counted the
    same way (one per exec call, in order).

    The budget is spent per-line first and only then globally, because that is
    where the information was actually going: the old version elided the middle
    of long transcripts (which mattered rarely -- a 66-step run rendered to 34k
    characters) while truncating every tool result to 300 characters (which
    mattered always -- the failing assertion lives past that cut). A result long
    enough to be worth reading is kept head-and-tail, never head-only: a test
    run's verdict is at the END.
    """
    lines: List[str] = []
    step = 0
    for record in read_journal(journal_path):
        kind = record.get("type")
        if kind == "tool.started":
            step += 1
            args = json.dumps(record.get("args") or {}, ensure_ascii=False)
            lines.append("[%d] %s(%s)" % (
                step, str(record.get("name") or "?").split("__")[-1],
                _clip(args, ARG_CHARS)))
        elif kind == "tool.finished" and lines:
            out = str(record.get("output") or "")
            lines[-1] += "  -> %s" % (_clip(out, RESULT_CHARS)
                                      or record.get("status") or "")
        elif kind == "agent.message":
            text = str(record.get("text") or "").replace("\n", " ")
            if text.strip():
                lines.append("    (agent said: %s)"
                             % _clip(text, MESSAGE_CHARS))
    body = "\n".join(lines)
    budget = int(token_budget * CHARS_PER_TOKEN)
    if len(body) > budget:
        # Elide the middle, and give the TAIL two thirds: the late steps contain
        # the failure being diagnosed, the early ones only establish what was
        # understood. An even split looked fair and was not -- with a single step
        # rendering to several thousand characters, half a small budget did not
        # reach the end of step 1, so the last steps vanished entirely.
        head = budget // 3
        tail = budget - head
        body = body[:head] + "\n...[middle elided]...\n" + body[-tail:]
    return body, 1, step


def _clip(text: str, limit: int) -> str:
    """Head AND tail when something is too long -- never head only.

    A test run's verdict is at the end of its output, so head-only truncation
    keeps the banner and drops the answer.
    """
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return "%s …[%d chars elided]… %s" % (text[:head], len(text) - limit,
                                          text[-tail:])


# --- grading ---------------------------------------------------------------
@dataclass
class Grade:
    resolved: bool = False
    f2p_pass: bool = False
    p2p_pass: bool = False
    patch: str = ""
    detail: str = ""
    error: Optional[str] = None
    verifier_artifacts: Optional[str] = None
    verifier_artifact_error: Optional[str] = None
    #: Names of the PASS_TO_PASS tests this attempt broke, when the runner said
    #: which. The single most useful thing an analyst can be told about a
    #: regression: without it a branch can only guess what it broke.
    broken: List[str] = field(default_factory=list)
    #: Test ids skipped because the DATASET damaged them. Recorded so a result
    #: says how much of the suite actually ran -- a grade over a silently
    #: shrunken suite is laxer than the benchmark it claims to be.
    skipped_ids: List[str] = field(default_factory=list)
    #: True when the agent had edited graded test files and those edits were
    #: discarded before grading (the public-leaderboard convention). Kept
    #: visible: the work was graded, the violation still happened.
    reverted_test_edits: bool = False

    #: False until the regression sweep actually ran -- it is skipped when the
    #: target tests fail, and reporting that skip as "FAIL" reads as a
    #: regression that was never measured.
    p2p_ran: bool = False
    grading_snapshot: Optional[dict] = None

    def summary(self) -> str:
        if self.error:
            return "GRADING ERROR: %s" % self.error
        regression = ("PASS" if self.p2p_pass else "FAIL") if self.p2p_ran \
            else "not run"
        skipped = (", %d malformed id(s) skipped" % len(self.skipped_ids)
                   if self.skipped_ids else "")
        return ("resolved=%s (target tests %s, regressions %s%s)"
                % (self.resolved, "PASS" if self.f2p_pass else "FAIL",
                   regression, skipped))


def grade_snapshot(snapshot_id: str, instance: dict, backend: dict,
                   timeout: float = 1800.0) -> Grade:
    """Restore a snapshot into a fresh microVM and run the instance's tests.

    Order matters: the target tests first (cheap, and the only thing that can
    make this a success), regressions second (hundreds of tests, minutes). A run
    that fails its target need not pay for the regression sweep.
    """
    from harness.execution.session import SandboxSession

    grade = Grade()
    session = SandboxSession(quiet=True, backend=with_sandbox_budget(backend, 2 * timeout + 600))
    if not session.create(snapshot_id):
        grade.error = "could not restore %s: %s" % (snapshot_id,
                                                    session.create_error)
        return grade
    owned_session = session
    recorder = None
    try:
        if instance.get("verifier_artifacts_dir"):
            from swebench.verifier_logs import VerifierLogSession
            recorder = VerifierLogSession(session, Path(instance["verifier_artifacts_dir"]))
            session = recorder
        diff = session.execute("shell", {"command": "cd /testbed && git diff",
                                         "timeout": 120})
        grade.patch = (diff.output or "") if diff.success else ""

        # Official protocol: the graded tests are defined by the dataset's
        # test_patch, NOT by the copies the image ships -- those predate the fix,
        # so a target test may not exist at all (measured: the target test was
        # "not found" in every snapshot, pre- and post-edit) or may assert the
        # OLD behaviour and pass on unfixed code. A test_patch that will not
        # apply means the agent's edits collided with the graded tests
        # themselves; that grades 0 rather than falling back to stale copies.
        test_patch = str(instance.get("test_patch") or "")
        if test_patch.strip():
            # Public-leaderboard convention: the model's patch EXCLUDES test
            # files, so an agent's edits to the graded tests are discarded
            # before test_patch lands, not graded as a collision. Measured: 53
            # of the first full 500 were "test_patch did not apply", and the
            # spot-check confirmed all five sampled had genuinely edited graded
            # tests -- under this convention those edits are simply not part of
            # the answer. The revert is per-file: back to HEAD when tracked,
            # deleted when the agent invented it.
            graded_files = sorted(set(
                re.findall(r"^\+\+\+ b/(\S+)", test_patch, re.M) +
                re.findall(r"^--- a/(\S+)", test_patch, re.M)) - {"/dev/null"})
            if graded_files:
                quoted = " ".join("'%s'" % f for f in graded_files)
                revert = session.execute("shell", {
                    "command": "cd /testbed && for f in %s; do "
                               "git checkout HEAD -- \"$f\" 2>/dev/null "
                               "|| rm -f \"$f\"; done" % quoted,
                    "timeout": 120})
                grade.reverted_test_edits = _exit_ok(revert)
            session.execute("text_editor", {
                "command": "write", "path": "/tmp/.swebench_test.patch",
                "file_text": test_patch})
            applied = session.execute("shell", {
                "command": "cd /testbed && git apply /tmp/.swebench_test.patch",
                "timeout": 120})
            if not _exit_ok(applied):
                grade.error = ("test_patch did not apply even after reverting "
                               "the attempt's test edits")
                return grade

        f2p = parse_test_list(instance.get("f2p") or instance.get("FAIL_TO_PASS"))
        p2p = parse_test_list(instance.get("p2p") or instance.get("PASS_TO_PASS"))
        repo = instance["repo"]

        if repo == "django/django":
            # Graded by PARSING the runner's verbose output -- the official
            # semantics, and the only representation in which the dataset's
            # docstring ids exist at all. Handing labels to runtests.py dies at
            # collection on any prose id; that fiction cost 105 verdicts in this
            # batch's first grading.
            _grade_django(session, instance, f2p, p2p, grade, timeout)
            return grade

        # Drop ids the dataset itself damaged, and SAY SO. Silently dropping them
        # would make grading laxer than the benchmark; keeping them makes every
        # attempt fail regardless of what it did (see malformed_test_ids).
        for label, ids in (("FAIL_TO_PASS", f2p), ("PASS_TO_PASS", p2p)):
            bad = malformed_test_ids(ids)
            if bad:
                grade.skipped_ids += bad
                for one in bad:
                    ids.remove(one)
                print("   note: %d malformed %s id(s) skipped (dataset splits "
                      "parametrised ids on their internal commas): %s"
                      % (len(bad), label, ", ".join(b[-40:] for b in bad[:3])))
        if not f2p:
            grade.error = ("every FAIL_TO_PASS id is malformed in the dataset -- "
                           "this instance cannot be graded")
            return grade

        files = instance.get("test_files") or None
        runner = _install_runner(session, repo, f2p)

        result = _run_tests(session, repo, f2p, files, runner, timeout)
        grade.f2p_pass = _exit_ok(result)
        grade.detail = "target: %s" % (result.output or result.error or "")[-1200:]

        if grade.f2p_pass and p2p:
            grade.p2p_ran = True
            result = _run_tests(session, repo, p2p, files, runner, timeout)
            grade.p2p_pass = _exit_ok(result)
            text = result.output or result.error or ""
            grade.broken = _failing_tests(text)
            # Name the tests, not just the tail. Two instances in an 8-run batch
            # stalled at "target passes, regressions fail" across seven branches
            # each, because the analyst got 1200 trailing characters of a 57-test
            # run -- the failing test's name was usually not in them, so every
            # branch guessed at WHICH regression it had caused.
            grade.detail += "\n\nregressions: %s%s" % (
                ("BROKEN: " + ", ".join(grade.broken) + "\n") if grade.broken else "",
                text[-4000:])
        grade.resolved = grade.f2p_pass and (grade.p2p_pass or not p2p)
    except Exception as exc:  # noqa: BLE001 - an ungradeable attempt is a zero
        grade.error = "%s: %s" % (type(exc).__name__, exc)
    finally:
        try:
            if recorder is not None:
                recorder.finish(grade)
        finally:
            owned_session.destroy()
    return grade


def _grade_django(session, instance: dict, f2p: List[str], p2p: List[str],
                  grade: "Grade", timeout: float) -> None:
    """Run the covering modules once per phase and match ids against output."""
    from swebench.dataset import django_modules, parse_django_verbose

    bracket_broken = [t for t in f2p + p2p if t.count("[") != t.count("]")]
    grade.skipped_ids += bracket_broken
    f2p = [t for t in f2p if t not in bracket_broken]
    p2p = [t for t in p2p if t not in bracket_broken]
    if not f2p:
        grade.error = "every FAIL_TO_PASS id is damaged -- cannot grade"
        return

    files = instance.get("test_files") or []

    def run_phase(ids: List[str]) -> tuple:
        modules = django_modules(ids, files)
        if not modules:
            return False, "no runnable module for: %s" % ids[:3]
        # PYTHONIOENCODING: these images run a POSIX/ascii locale, and
        # verbosity 2 makes django print "Creating tables…" -- one ellipsis and
        # the whole run dies of UnicodeEncodeError before any test.
        command = ("cd /testbed && PYTHONIOENCODING=utf-8 "
                   "./tests/runtests.py --verbosity 2 --parallel 1 %s"
                   % " ".join(modules))
        result = session.execute("shell", {"command": test_command(command, needs_pytest=False),
                                           "timeout": int(timeout)})
        validate_test_execution(result)
        text = ""
        try:
            body = json.loads(result.output or "{}")
            text = (body.get("stdout") or "") + (body.get("stderr") or "")
        except (ValueError, TypeError):
            text = result.output or result.error or ""
        passed, failed = parse_django_verbose(text)
        missing = [t for t in ids if t not in passed]
        return not missing, ("%d/%d ids pass; missing/failing: %s\n%s"
                             % (len(ids) - len(missing), len(ids),
                                missing[:5], text[-3000:]))

    grade.f2p_pass, detail = run_phase(f2p)
    grade.detail = "target: %s" % detail
    if grade.f2p_pass and p2p:
        grade.p2p_ran = True
        grade.p2p_pass, detail = run_phase(p2p)
        if not grade.p2p_pass:
            grade.broken = [line for line in detail.splitlines()[:1]]
        grade.detail += "\n\nregressions: %s" % detail
    grade.resolved = grade.f2p_pass and (grade.p2p_pass or not p2p)


def _install_runner(session, repo: str, test_ids: List[str]) -> Optional[str]:
    """Place the file-based runner in the sandbox, if this repo needs one.

    Written with ``text_editor``, whose argument is a JSON string: the script
    arrives byte-exact. Building an equivalent shell command instead cost an hour
    to a regex that reached the interpreter as a literal ``d+``.
    """
    if not needs_file_runner(repo, test_ids):
        return None
    session.execute("text_editor", {"command": "write",
                                    "path": "/tmp/ash_runner.py",
                                    "file_text": SYMPY_RUNNER})
    return "/tmp/ash_runner.py"


def _run_tests(session, repo: str, test_ids: List[str],
               files: Optional[List[str]], runner: Optional[str],
               timeout: float):
    if runner:
        spec = json.dumps(sympy_runner_spec(test_ids, files))
        session.execute("text_editor", {"command": "write",
                                        "path": "/tmp/ash_spec.json",
                                        "file_text": spec})
        command = "cd /testbed && python %s /tmp/ash_spec.json" % runner
    else:
        command = "cd /testbed && %s" % build_batch_test_command(
            repo, test_ids, files)
    result = session.execute("shell", {"command": test_command(command, needs_pytest=runner is None),
                                      "timeout": timeout, "tail": 60})
    validate_test_execution(result)
    return result


#: How each runner announces a failure. sympy's own runner and the direct-call
#: runner print "FAIL <dotted.name>"; pytest prints "FAILED path::test - msg" and
#: also lists them under a "short test summary info" banner; django's runner uses
#: "FAIL: test_x (mod.Cls)".
_FAILURE_PATTERNS = (
    re.compile(r"^FAIL(?:ED)?[: ]+(\S+)", re.M),
    re.compile(r"^ERROR[: ]+(\S+)", re.M),
)


def _failing_tests(text: str, limit: int = 25) -> List[str]:
    """Test names a runner reported as failing, in order, de-duplicated.

    Best-effort across four runners on purpose: a name we fail to extract costs
    the analyst a hint, while a wrong guess about the format would cost nothing
    at all -- the raw tail is still included either way.
    """
    seen, out = set(), []
    for pattern in _FAILURE_PATTERNS:
        for name in pattern.findall(text or ""):
            name = name.strip().rstrip(":,")
            if name and name not in seen:
                seen.add(name)
                out.append(name)
            if len(out) >= limit:
                return out
    return out


def _exit_ok(result) -> bool:
    """Whether a shell ToolResult reports exit code 0.

    The runtime reports the exit code inside its JSON payload, so a non-zero
    test run still comes back "successful" as a *tool call*. Reading only
    ``result.success`` would score every failing test suite as a pass.
    """
    if not result.success:
        return False
    text = result.output or ""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return True          # not the JSON envelope: no exit code to contradict
    if isinstance(payload, dict) and "exit_code" in payload:
        return int(payload.get("exit_code") or 0) == 0
    return True


# --- attempts --------------------------------------------------------------
@dataclass
class Attempt:
    name: str
    outcome: RunOutcome
    grade: Grade
    plan: dict = field(default_factory=dict)
    #: The direction this attempt was given ("" for the parent). Analyses quote
    #: it so the reviewer can see hypothesis -> outcome in one place.
    hint: str = ""
    round_no: int = 0
    assistant_turn: Optional[dict] = None

    def verdict_text(self) -> str:
        """The grading verdict as the analysts and branch prompts see it."""
        broken = ""
        if self.grade.broken:
            broken = ("\n\nTests this attempt BROKE (they passed before it): %s"
                      % ", ".join(self.grade.broken))
        return "%s%s\n\nPatch (%d lines):\n%s\n\nTest output:\n%s" % (
            self.grade.summary(), broken, self.grade.patch.count("\n"),
            self.grade.patch[:40000], self.grade.detail[:20000])

    @property
    def score(self) -> int:
        if self.grade.resolved:
            return 3
        if self.grade.f2p_pass:
            return 2          # target fixed, something else broke
        return 1 if self.grade.patch.strip() else 0


def run_attempt(orch: Orchestrator, args, instance: dict, *, name: str,
                prompt: str, image: str, out_dir: Path,
                resume: Optional[str] = None,
                fork: bool = False, origin: Optional[dict] = None,
                resources: Optional[dict] = None,
                bench: "Optional[Benchmark]" = None,
                resume_at: Optional[str] = None,
                cwd: Optional[Path] = None,
                assistant_turn: Optional[dict] = None,
                resume_without_hint: bool = False) -> RunOutcome:
    """One attempt. Parent and branches differ only in the arguments.

    ``out_dir`` is per instance: eight instances writing `parent.jsonl` into one
    directory would overwrite each other's journals, and the journal is the only
    record a killed run leaves.
    """
    runtime_bin = str(Path(args.runtime_bin).resolve())
    if getattr(args, "agent_network", None) is not None:
        network = network_policy_for(args, bench, "agent")
        prompt += f"\n\nSandbox internet access for this attempt: {network}."
    extra: dict = {}
    if resume_without_hint:
        if args.slot != "mini-swe-agent" or not resume or assistant_turn is not None:
            raise ValueError("Point-only continuation requires a resumed mini without an assistant_turn")
        extra["resume_without_hint"] = True
    if assistant_turn is not None:
        if args.slot != "mini-swe-agent" or not resume:
            raise ValueError("assistant-turn execution requires a resumed mini-swe-agent")
        extra["assistant_turn"] = validate_assistant_turn(assistant_turn)
    if args.slot == "codex":
        # Native Bedrock provider: OpenAI's own models are hosted there, so no
        # translator and no login. Pre-serialized TOML values, which is what
        # codex's -c overrides take.
        extra["config_overrides"] = {"model_provider": '"amazon-bedrock"'}
    if args.slot.startswith("opencode"):
        extra["data_home"] = str(out_dir / "state" / "shared")
    if args.slot == "mini-swe-agent":
        extra["native_home"] = str(out_dir / "state" / "mini-native")
        if resume and origin and origin.get("parent_journal"):
            from runstore.mini_native import reference_at

            reference = reference_at(origin["parent_journal"], origin["branch_step"], resume)
            if reference is None:
                raise ValueError("Selected mini branch has no exact native prefix")
            extra["native_prefix"] = reference
        if (assistant_turn is not None or resume_without_hint) and "native_prefix" not in extra:
            raise ValueError("This branch mode requires an exact native prefix")
    if resume_at:
        # Transcript entry (uuid) the resumed conversation ends at. The slot
        # passes it to the SDK's resume_session_at; see conversation_cut.
        extra["resume_session_at"] = resume_at
    if origin and origin.get("conversation_restore") == "original-prefix":
        extra["setting_sources"] = []
    spec = RunSpec(
        prompt=prompt, slot=args.slot, cwd=str(cwd) if cwd is not None else "/tmp", model=args.model,
        timeout_s=args.timeout, run_id=name,
        journal_path=out_dir / ("%s.jsonl" % name),
        transport="http", tools="shell_only" if args.slot in {"claude-code", "mini-swe-agent"} else "default",
        backend=backend_for(args, bench), runtime_bin=runtime_bin,
        sandbox_image=image, sandbox_resources=resources,
        resume_session_id=resume, fork=fork, origin=origin, extra=extra,
    )
    return orch.run(spec)


def network_policy_for(args, bench: "Optional[Benchmark]" = None, phase: str = "agent") -> str:
    if phase not in ("agent", "verifier"):
        raise ValueError("Network phase must be agent or verifier")
    requested = getattr(args, f"{phase}_network", None)
    if requested is not None:
        if requested not in ("allow", "deny"):
            raise ValueError(f"Invalid {phase} network policy: {requested}")
        return requested
    return "deny" if getattr(bench, "no_network", False) else "backend-default"


def network_summary(args, bench: "Optional[Benchmark]" = None) -> dict:
    policy = {phase: network_policy_for(args, bench, phase) for phase in ("agent", "verifier")}
    return {"network_policy": policy,
            "network_requested": {phase: getattr(args, f"{phase}_network", None) for phase in policy},
            "no_network": policy["agent"] == "deny" if policy["agent"] == policy["verifier"] else None}


def backend_for(args, bench: "Optional[Benchmark]" = None, phase: str = "agent") -> dict:
    microvm: dict = {"from_image": True,
                     "runtime_bin": str(Path(args.runtime_bin).resolve())}
    if bench is not None and getattr(bench, "runtime_port", None) is not None:
        microvm["runtime_port"] = bench.runtime_port
    network = network_policy_for(args, bench, phase)
    if network != "backend-default":
        microvm["allow_internet"] = network == "allow"
    if bench is not None and bench.image_env:
        microvm["image_env"] = True
    if bench is not None and getattr(bench, "runtime_init", ""):
        microvm["runtime_init"] = bench.runtime_init
    return with_sandbox_budget(
        {"backend": "microvm", "microvm": microvm}, getattr(args, "timeout", 1800.0))


def grade_attempt(outcome: RunOutcome, instance: dict, args,
                  bench: "Optional[Benchmark]" = None) -> Grade:
    journal = Path(outcome.journal_path)
    instance = {**instance, "verifier_artifacts_dir": str(journal.parent / (journal.stem + ".verifier"))}
    events = list(read_journal(outcome.journal_path))
    captures = [(index, record) for index, record in enumerate(events)
                if record.get("type") == "checkpoint.captured" and record.get("snapshot_id")
                and (record.get("reason") or "captured") == "captured"
                and record.get("captured") is not False]
    if not captures:
        return Grade(error="no successful snapshot recorded -- nothing to grade")
    selected_index, selected = captures[-1]
    tail = events[selected_index + 1:]
    snapshot = {
        "policy": "last_successful_snapshot",
        "snapshot_id": selected["snapshot_id"],
        "capture_step": selected.get("step"),
        "capture_seq": selected.get("seq"),
        "later_tool_calls": [{key: record.get(key) for key in ("step", "call_id", "name")}
                             for record in tail if record.get("type") == "tool.started"],
        "later_checkpoint_issues": [{key: record.get(key) for key in ("step", "call_id", "reason")}
                                    for record in tail if record.get("type") == "checkpoint.captured"
                                    and record.get("reason") not in ("captured", "clean", "session_ref_backfill")],
    }
    bench = bench or SweBench()
    snapshot["network_policy"] = network_policy_for(args, bench, "verifier")
    grade = bench.grade(selected["snapshot_id"], instance, backend_for(args, bench, "verifier"))
    grade.grading_snapshot = snapshot
    return grade


# --- benchmarks --------------------------------------------------------------
#
# The loop above is benchmark-neutral: it runs an agent, grades a snapshot,
# and branches on a Grade. What differs per benchmark is where tasks come
# from, what the agent is told, what shape its sandbox needs, and how a
# snapshot becomes a Grade. That is this interface; SWE-bench is the default
# and behaves exactly as before, DeepSWE lives in ``deepswe/`` and is imported
# only when asked for, so the SWE-bench path never depends on it.

class Benchmark:
    """What ``fork_eval`` needs from a benchmark. Duck-typed; SweBench is the reference."""
    name: str = ""
    #: True when every task runs with sandbox egress disabled.
    no_network: bool = False
    #: True when the runtime must run under the image's own ENV (PATH etc.),
    #: i.e. the benchmark's verifier assumes the Docker image's environment.
    image_env: bool = False

    def catalogue(self, args) -> dict:                    # id -> raw record
        raise NotImplementedError

    def instance(self, raw) -> dict:                      # raw -> loop's dict
        raise NotImplementedError

    def prompt(self, instance: dict) -> str:
        raise NotImplementedError

    def branch_prompt(self, instance: dict, verdict: str, hint: str, **context) -> str:
        """``context`` (truncated, step, grade, analysis) lets a benchmark phrase
        the branch message for a conversation cut at the fork step."""
        raise NotImplementedError

    def resources(self, instance: dict) -> Optional[dict]:
        return None

    def grade(self, snapshot_id: str, instance: dict, backend: dict) -> Grade:
        raise NotImplementedError


class SweBench(Benchmark):
    name = "swebench"
    runtime_init = CONDA_INIT

    def catalogue(self, args) -> dict:
        return {i["instance_id"]: i for i in load_instances(args.subset)}

    def instance(self, raw: dict) -> dict:
        return {
            "instance_id": raw["instance_id"], "repo": raw["repo"],
            "image": resolve_image(raw),
            "problem": raw.get("problem_statement") or "",
            "f2p": parse_test_list(raw["FAIL_TO_PASS"]),
            "p2p": parse_test_list(raw["PASS_TO_PASS"]),
            "test_files": test_files_of(raw),
            "test_patch": raw.get("test_patch") or "",
        }

    def prompt(self, instance: dict) -> str:
        prompt = PROMPT.format(repo=instance["repo"], problem=instance["problem"],
                               primer=tool_primer(instance.get("slot", "")))
        if instance.get("agent_network") == "deny":
            prompt += "\n\nThe sandbox has no internet access. Use the files and dependencies already available."
        return prompt

    def branch_prompt(self, instance: dict, verdict: str, hint: str, **context) -> str:
        return render_branch_note(
            hint, truncated=bool(context.get("truncated")))

    def grade(self, snapshot_id: str, instance: dict, backend: dict) -> Grade:
        return grade_snapshot(snapshot_id, instance, backend)


def select_benchmark(args) -> Benchmark:
    name = str(getattr(args, "benchmark", "") or "swebench").lower()
    if name == "swebench":
        return SweBench()
    if name == "deepswe":
        from deepswe.bench import DeepSWE
        return DeepSWE(getattr(args, "tasks_dir", None))
    if name == "swebench-pro":
        from swebench_pro.bench import SWEbenchPro
        return SWEbenchPro(args)
    raise SystemExit("unknown --benchmark %r; choose swebench, deepswe or swebench-pro" % name)


# --- truncating the forked conversation ---------------------------------------
#
# The checkpoint pair for step N is (snapshot after tool call N, session id).
# The session id alone cannot express "up to step N": resuming it with
# fork_session copies the WHOLE transcript -- measured 2026-09-04, every branch
# of the 93.0% SWE-bench run and of the first DeepSWE branching round carried
# the parent's post-fork steps and its closing "done" summary, while its
# filesystem was at step N. Claude Code's transcript has one entry per tool
# result, keyed by the same tool_use id the journal records as call_id, so the
# cut point is the uuid of the user entry holding step N's tool_result; the SDK's
# resume_session_at loads the conversation up to and including it.

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _compaction_preserves_cut(entry: dict, cut: str) -> bool:
    metadata = entry.get("compactMetadata")
    if not isinstance(metadata, dict):
        return False
    preserved = metadata.get("preservedMessages")
    if not isinstance(preserved, dict):
        return False
    uuids = preserved.get("uuids")
    return (isinstance(uuids, list) and all(isinstance(uuid, str) for uuid in uuids)
            and cut in uuids)


def _find_conversation_cut(journal_path, step: int, session_ref: Optional[str] = None,
                           *, allow_prefix: bool = False) -> Optional[str]:
    """uuid of the transcript entry that ends step ``step`` of ``journal_path``.

    None when it cannot be found -- the journal has no such step, the native
    session transcript is not on this host, or Claude Code recorded the tool
    result differently. Callers must not fall back to an untruncated fork
    silently: that is the bug this exists to fix.
    """
    journal_path = Path(journal_path)
    events = read_journal(journal_path)
    turns = completed_turn_steps(events)
    if turns is not None and step not in turns:
        return None
    calls: List[str] = []
    session_id = session_ref
    for record in events:
        kind = record.get("type")
        if kind == "tool.started" and record.get("call_id"):
            calls.append(str(record["call_id"]))
        elif not session_ref and kind == "session.ref" and record.get("native_session_id"):
            session_id = str(record["native_session_id"])
    if not session_id or step < 1 or step > len(calls):
        return None
    call_id = calls[step - 1]
    # New records name their tool explicitly. Never repair a mismatched label by
    # silently cutting at a different native call. Legacy journals lack this field.
    checkpoint = next((c for c in reversed(load_checkpoints(journal_path))
                       if c.step == step and c.reason != "session_ref_backfill"), None)
    if checkpoint and checkpoint.pairing is not None:
        if (checkpoint.call_id != call_id or checkpoint.prefix_complete is not True
                or checkpoint.reason not in ("captured", "clean")):
            return None
    for transcript in CLAUDE_PROJECTS_DIR.glob("*/%s.jsonl" % session_id):
        # One response can be split across native assistant entries, including
        # entries after an early tool result. Scan its complete tool group first.
        entries = []
        groups, call_groups = {}, {}
        with transcript.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                entries.append(entry)
                message = entry.get("message") or {}
                if entry.get("type") != "assistant" or not isinstance(message.get("content"), list):
                    continue
                group_id = message.get("id") or entry.get("uuid")
                for block in message["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                        cid = block["id"]
                        groups.setdefault(group_id, set()).add(cid)
                        call_groups[cid] = group_id
        required_group = groups.get(call_groups.get(call_id), {call_id})
        found = None
        post_cut_message = False
        preserved_compaction = False
        pending = set()
        seen_results = set()
        call_positions = {cid: n for n, cid in enumerate(calls, 1)}
        for entry in entries:
            if found is not None:
                if entry.get("subtype") == "compact_boundary":
                    if (preserved_compaction or post_cut_message
                            or not _compaction_preserves_cut(entry, found)):
                        if not allow_prefix:
                            return None
                    preserved_compaction = True
                elif (entry.get("type") in ("assistant", "user")
                      and (entry.get("message") or {}).get("content")):
                    post_cut_message = True
                elif entry.get("type") == "attachment":
                    attachment = entry.get("attachment")
                    if (not isinstance(attachment, dict)
                            or attachment.get("type") != "total_tokens_reminder"):
                        post_cut_message = True
                continue
            content = (entry.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("id"):
                    pending.add(block["id"])
                elif block.get("type") == "tool_result" and block.get("tool_use_id"):
                    cid = block["tool_use_id"]
                    pending.discard(cid)
                    seen_results.add(cid)
            if entry.get("type") == "user" and call_id in seen_results:
                # Never cut through a response, an unresolved tool, or future results.
                if (pending or not required_group <= seen_results
                        or any(call_positions.get(cid, 0) > step for cid in seen_results)
                        or not set(calls[:step]).issubset(seen_results)):
                    return None
                found = entry.get("uuid")
        if found is not None:
            return found
    return None


def conversation_cut(journal_path, step: int, session_ref: Optional[str] = None) -> Optional[str]:
    """A UUID loadable from the original session without prefix reconstruction."""
    return _find_conversation_cut(journal_path, step, session_ref)


def conversation_restore(journal_path, step: int, session_ref: str) -> tuple[str, PrefixSource | None] | None:
    """Choose a direct native cut or a validated original-prefix source."""
    if any(row.get("type") == "run.started" and row.get("slot") == "mini-swe-agent"
           for row in read_journal(journal_path)):
        from runstore.mini_native import reference_at

        reference = reference_at(journal_path, step, session_ref)
        return (reference["cut"], None) if reference else None
    cut = conversation_cut(journal_path, step, session_ref)
    if cut is not None:
        return cut, None
    cut = _find_conversation_cut(journal_path, step, session_ref, allow_prefix=True)
    if cut is not None:
        source = find_prefix_source(CLAUDE_PROJECTS_DIR, session_ref, cut)
        if source is not None:
            return cut, source
    return None


def available_branch_points(journal_path, *, full_conversation=False):
    """The same candidate definition for analysts, reviewers and batch canaries."""
    points = turn_branch_checkpoints(journal_path)
    if full_conversation:
        return points
    return {s: p for s, p in points.items()
            if conversation_restore(journal_path, s, p.session_ckpt)}


# --- reusing a recorded parent -----------------------------------------------
#
# Branching is about the FAILED trajectory: its snapshots and its verdict are
# what the analysts fork from. Re-running the parent first (what branch134 did,
# because its prompt had changed) is a blind retry that costs a full attempt
# per task and tells the branches nothing. With the prompt unchanged, the
# recorded single-pass journal IS the parent: copy it into the run directory,
# grade its last snapshot, and go straight to round 1.

@dataclass(frozen=True)
class BranchChoice:
    run_name: str
    base: Attempt
    checkpoint: Checkpoint
    cut: Optional[str]
    hint: str
    why: str
    prefix: Optional[PrefixSource] = None
    assistant_turn: Optional[dict] = None


def prepare_branches(plan: dict, *, limit: int, round_no: int,
                     attempts: dict, checkpoints: dict,
                     full_conversation: bool = False,
                     count_mode: str = "adaptive",
                     guidance_mode: str = "user-hint") -> List[BranchChoice]:
    """Resolve each selected point exactly before any continuation starts."""
    if count_mode not in BRANCH_COUNT_MODES:
        raise ValueError("unknown branch count mode: %r" % count_mode)
    validate_guidance(guidance_mode, "mini-swe-agent", full_conversation)
    branches = plan.get("branches") if isinstance(plan, dict) else None
    if not isinstance(branches, list):
        raise ReviewerPlanError("reviewer must return a branches list")
    if count_mode == "fixed" and len(branches) != limit:
        raise ReviewerPlanError("fixed branch count requires exactly %d branches; reviewer returned %d" %
                         (limit, len(branches)))
    if len(branches) > limit:
        raise ReviewerPlanError("reviewer returned %d branches above limit %d" % (len(branches), limit))
    choices = []
    cuts = {}
    for index, branch in enumerate(branches, 1):
        if not isinstance(branch, dict):
            raise ReviewerPlanError("branch %d is not an object" % index)
        base_name = branch.get("base")
        if not isinstance(base_name, str) or base_name not in attempts:
            raise ReviewerPlanError("branch %d names an unknown base %r" % (index, base_name))
        step = branch.get("branch_step")
        if type(step) is not int or step not in checkpoints.get(base_name, {}):
            raise ReviewerPlanError("branch %d has no eligible checkpoint at %s:%r" % (index, base_name, step))
        base = attempts[base_name]
        assistant_turn = None
        if guidance_mode == "none":
            if set(branch) - {"name", "base", "branch_step", "why"}:
                raise ReviewerPlanError("none branches may contain only point-selection fields, not guidance")
            selected_prefix(base.outcome.journal_path, checkpoints[base_name][step])
            hint = ""
        elif guidance_mode == "assistant-turn":
            if "hint" in branch:
                raise ReviewerPlanError("assistant-turn branches must not include a user hint")
            native = selected_prefix(base.outcome.journal_path, checkpoints[base_name][step])
            tools = actor_tools_at(base.outcome.journal_path, step)
            try:
                assistant_turn = validate_assistant_turn(
                    branch.get("assistant_turn"),
                    history=[e["message"] for e in native if e["type"] == "mini.message"], tools=tools)
            except ValueError as error:
                raise ReviewerPlanError(f"branch {index} ({base_name}@{step}): {error}") from error
            hint = ""
        else:
            if "assistant_turn" in branch:
                raise ReviewerPlanError("assistant_turn requires --branch-guidance assistant-turn")
            hint = branch.get("hint")
            if not isinstance(hint, str) or not hint.strip():
                raise ReviewerPlanError("branch %d must supply a non-empty hint" % index)
        cut = None
        prefix = None
        if not full_conversation:
            key = (base_name, step)
            if key not in cuts:
                cuts[key] = conversation_restore(
                    base.outcome.journal_path, step, checkpoints[base_name][step].session_ckpt)
            restoration = cuts[key]
            if restoration is None:
                raise ValueError("branch %d has no native conversation cut at %s:%d" %
                                 (index, base_name, step))
            cut, prefix = restoration
        choices.append(BranchChoice(
            branch_run_name(round_no, index, branch.get("name")), base,
            checkpoints[base_name][step], cut, hint, str(branch.get("why") or ""), prefix, assistant_turn))
    return choices


def existing_parent(source: str, instance_id: str) -> Optional[Path]:
    """The recorded parent journal for ``instance_id`` under ``source``.

    ``source`` is either an aggregate file (``scripts/deepswe_aggregate.py``
    output: the final journal per task, reruns already layered) or a batch
    directory searched for ``**/<instance_id>/parent.jsonl``.
    """
    root = Path(source)
    if root.suffix == ".json":
        for entry in json.loads(root.read_text()).get("tasks", []):
            if entry.get("task") == instance_id and entry.get("journal"):
                return Path(entry["journal"])
        return None
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("failure_policy") == "isolated":
            item = next((item for item in manifest["tasks"] if item["id"] == instance_id), None)
            if item is None:
                return None
            worker_path = root / f"shard-{item['index']:03d}" / "worker.json"
            worker = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
            if not (worker.get("finished_at") and worker.get("evidence_valid") and worker.get("journal_path")):
                return None
            journal = Path(worker["journal_path"])
            if not journal.resolve().is_relative_to(root.resolve()) or not journal.is_file():
                return None
            return journal
    hits = sorted(root.glob("**/%s/parent.jsonl" % instance_id))
    return hits[0] if hits else None


def outcome_from_journal(journal: Path, run_id: str = "parent") -> RunOutcome:
    """A ``RunOutcome`` rebuilt from what a finished run left in its journal."""
    status, usage, error, final_text = "unknown", {}, None, ""
    for record in read_journal(journal):
        kind = record.get("type")
        if kind == "run.finished":
            status = record.get("status") or status
            usage = record.get("usage") or {}
            error = record.get("error")
        elif kind == "run.result":
            final_text = record.get("text") or ""
    pairs = sum(1 for c in load_checkpoints(journal) if c.snapshot_id)
    return RunOutcome(run_id=run_id, journal_path=Path(journal), status=status,
                      final_text=final_text, usage=usage, checkpoints=pairs,
                      error=error)


def report(attempt: Attempt) -> None:
    print("   status     %s%s" % (attempt.outcome.status,
                                  " (%s)" % attempt.outcome.error
                                  if attempt.outcome.error else ""))
    print("   pairs      %d" % attempt.outcome.checkpoints)
    print("   grade      %s" % attempt.grade.summary())
    print("   patch      %d 行" % attempt.grade.patch.count("\n"))


def regrade(args, out_dir: Path, bench: "Optional[Benchmark]" = None) -> int:
    """Re-grade a finished run's snapshots with the CURRENT grader.

    Needed because a grader defect invalidates results without invalidating the
    *runs*: every attempt left a snapshot, so the verdict can be recomputed
    without spending another agent. (This exists because 7 of a 32-instance
    batch's 14 failures turned out to be the dataset's malformed test ids, not
    the agent -- five of those instances had been solved by their first attempt.)

    Attempts are graded in the order the loop would have made them, stopping at
    the first resolved one: had the grader been right, a run whose parent already
    passed would never have branched, so crediting those branches -- or paying to
    grade them -- would both be wrong.
    """
    bench = bench or SweBench()
    catalogue = bench.catalogue(args)
    previous = {}
    old_path = out_dir / "summary.json"
    if old_path.exists():
        for entry in (json.loads(old_path.read_text()).get("instances") or []):
            previous[entry["instance"]] = entry

    results, changed = [], []
    directories = sorted(d for d in out_dir.iterdir() if d.is_dir())
    expected_ids = [directory.name for directory in directories if directory.name in catalogue]
    for position, directory in enumerate(directories, 1):
        instance_id = directory.name
        raw = catalogue.get(instance_id)
        if raw is None:
            continue
        instance = bench.instance(raw)
        # parent first, then rounds in order -- the order attempts were made in.
        journals = sorted(directory.glob("*.jsonl"),
                          key=lambda p: (p.stem != "parent", p.stem))
        print("\n[%d/%d] %s" % (position, len(directories), instance_id))
        verdicts, resolved_by = [], None
        for journal in journals:
            pairs = [c for c in load_checkpoints(journal) if c.snapshot_id]
            if not pairs and not getattr(bench, "stop_on_grading_error", False):
                print("   %-30s no snapshot" % journal.stem)
                continue
            grade = grade_attempt(outcome_from_journal(journal), instance, args, bench)
            print("   %-30s %s" % (journal.stem, grade.summary()))
            verdicts.append({"name": journal.stem, "resolved": grade.resolved,
                             "f2p_pass": grade.f2p_pass,
                             "p2p_ran": grade.p2p_ran,
                             "p2p_pass": grade.p2p_pass,
                             "skipped_ids": len(grade.skipped_ids),
                             "broken": grade.broken,
                             "verifier_artifacts": grade.verifier_artifacts,
                             "grading_snapshot": grade.grading_snapshot,
                             "verifier_artifact_error": grade.verifier_artifact_error,
                             "grading_error": grade.error})
            if grade.resolved:
                resolved_by = journal.stem
                break
        was = bool(previous.get(instance_id, {}).get("resolved"))
        if bool(resolved_by) != was:
            changed.append((instance_id, was, bool(resolved_by)))
        results.append({"instance": instance_id, "resolved": bool(resolved_by),
                        "resolved_by": [resolved_by] if resolved_by else [],
                        "attempts": verdicts})
        (out_dir / "regrade.json").write_text(json.dumps(
            {"slot": args.slot, "model": args.model, "regraded": True,
             "network_policy": {"agent": "not-rerun", "verifier": network_policy_for(args, bench, "verifier")},
             "network_requested": {"verifier": getattr(args, "verifier_network", None)},
             "instances": results,
             **(bench.summary(results, expected_ids=expected_ids) if hasattr(bench, "summary") else {})}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    solved = [r for r in results if r["resolved"]]
    print("\n" + "=" * 72)
    print("RE-GRADED %d / %d resolved" % (len(solved), len(results)))
    for r in results:
        print("  %-4s %-34s %s" % ("PASS" if r["resolved"] else "fail",
                                   r["instance"],
                                   ", ".join(r["resolved_by"])))
    if changed:
        print("\nVERDICT CHANGED for %d:" % len(changed))
        for name, was, now in changed:
            print("  %-34s %s -> %s" % (name, "pass" if was else "fail",
                                        "pass" if now else "fail"))
    print("\nwrote %s" % (out_dir / "regrade.json"))
    if hasattr(bench, "summary") and not bench.summary(results, expected_ids=expected_ids).get("grading_complete", True):
        return 2
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="",
                        help="instance id, or a comma list of them. Not needed "
                             "with --regrade, which reads what is on disk.")
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--slot", default="codex")
    parser.add_argument("--model", default="openai.gpt-5.6-luna")
    parser.add_argument("--analyst-model", default="openai.gpt-5.6-luna")
    parser.add_argument("--rounds", type=int, default=2,
                       help="branching rounds after the first attempt")
    parser.add_argument("--branches", default="3", metavar="N[,N...]",
                        help="branch count per round: a maximum in adaptive mode, "
                             "an exact count in fixed mode. "
                             "One number for every round, or a comma list "
                             "(e.g. 4,3). A "
                             "round only happens if the one before it produced "
                             "no resolved attempt.")
    parser.add_argument("--branch-count-mode", choices=BRANCH_COUNT_MODES, default="adaptive",
                        help="adaptive: reviewer chooses up to --branches; "
                             "fixed: require exactly --branches valid directions per round")
    parser.add_argument("--branch-guidance", choices=BRANCH_GUIDANCE_MODES, default="user-hint",
                        help="user-hint: existing reviewer hint; assistant-turn: mini-only "
                             "reviewer response executed before continuation; none: mini-only "
                             "point selection followed by direct continuation without added messages")
    parser.add_argument("--reviewer-max-attempts", type=int, default=3,
                        help="maximum reviewer responses per round, including validation corrections (default: 3)")
    parser.add_argument("--analyst-tokens", type=int, default=100_000,
                        help="transcript budget handed to the analyst")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    parser.add_argument("--agent-network", choices=["allow", "deny"], default=None,
                        help="actor sandbox egress; omitted uses the benchmark default")
    parser.add_argument("--verifier-network", choices=["allow", "deny"], default=None,
                        help="verifier/patch-collector sandbox egress; omitted uses the benchmark default")
    parser.add_argument("-o", "--out", default="runs/fork-eval")
    parser.add_argument("--volatile-ok", action="store_true",
                        help="allow -o under /tmp and friends. Refused by "
                             "default: a reboot mid-batch already destroyed one "
                             "32-instance run's journals.")
    parser.add_argument("--regrade", action="store_true",
                        help="re-grade a finished run in -o with the current "
                             "grader, spending no agent time: every attempt left "
                             "a snapshot, so a grader fix can be applied to "
                             "results that already exist")
    parser.add_argument("--benchmark", default="swebench",
                        choices=["swebench", "deepswe", "swebench-pro"],
                        help="which benchmark supplies tasks, prompts and the "
                             "grader (default: swebench, unchanged behaviour)")
    parser.add_argument("--tasks-dir", default=None,
                        help="deepswe: the dataset's tasks/ directory")
    parser.add_argument("--pro-repo", help="swebench-pro: pinned official SWE-bench_Pro-os checkout")
    parser.add_argument("--pro-data", help="swebench-pro: local CSV/JSONL instead of the public dataset")
    parser.add_argument("--pro-dataset-revision", help="swebench-pro: dataset commit SHA (default: pinned public revision)")
    parser.add_argument("--pro-cpus", type=int, default=4)
    parser.add_argument("--pro-memory-mb", type=int, default=16384)
    parser.add_argument("--pro-verifier-timeout", type=int, default=3600)
    parser.add_argument("--pro-runtime-port", type=int, default=None)
    parser.add_argument("--pro-collector-runtime-port", type=int, default=None)
    parser.add_argument("--pro-block-network", action="store_true",
                        help="legacy Pro default: deny both phases unless their explicit network flags override it")
    parser.add_argument("--fork-full-conversation", action="store_true",
                        help="branch with the parent's WHOLE conversation (the "
                             "pre-2026-09-04 behaviour, tag branching-fullconv-"
                             "2026-09-04) instead of cutting it at the fork step; "
                             "normal mode rejects unavailable cuts without automatic fallback")
    parser.add_argument("--parent-from", default=None,
                        help="reuse each instance's recorded parent journal from "
                             "this batch dir or aggregate .json instead of running "
                             "a fresh parent: grade its last snapshot, then branch. "
                             "Refuses an instance that has none.")
    args = parser.parse_args(argv)
    if args.reviewer_max_attempts < 1:
        parser.error("--reviewer-max-attempts must be positive")
    try:
        validate_guidance(args.branch_guidance, args.slot, args.fork_full_conversation)
    except ValueError as error:
        parser.error(str(error))
    bench = select_benchmark(args)
    try:
        schedule = [int(x) for x in str(args.branches).split(",") if x.strip()]
    except ValueError:
        raise SystemExit("--branches wants numbers, got %r" % args.branches)
    if not schedule:
        raise SystemExit("--branches cannot be empty")
    if any(limit < 1 for limit in schedule):
        raise SystemExit("--branches limits must be positive; use --rounds 0 for no branching")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.regrade:
        # Not guarded: regrade only READS journals, and they are wherever the
        # original run put them.
        return regrade(args, out_dir, bench)
    if not args.instance:
        raise SystemExit("--instance is required (or use --regrade)")
    reason = volatile_reason(out_dir)
    if reason and not args.volatile_ok:
        raise SystemExit("refusing: %s (pass --volatile-ok to override)" % reason)

    wanted = [x.strip() for x in str(args.instance).split(",") if x.strip()]
    catalogue = bench.catalogue(args)
    if bench.name == "swebench-pro" and wanted == ["all"]:
        wanted = list(catalogue)
    missing = [x for x in wanted if x not in catalogue]
    if missing:
        raise SystemExit("not in %s: %s" % (
            args.tasks_dir if bench.name == "deepswe" else args.subset,
            ", ".join(missing)))

    orch = Orchestrator(out_dir=out_dir)
    results = []
    for position, instance_id in enumerate(wanted, 1):
        print("\n" + "=" * 72)
        print("INSTANCE %d/%d  %s" % (position, len(wanted), instance_id))
        print("=" * 72)
        attempts = run_one(orch, args, catalogue[instance_id], schedule,
                           out_dir / instance_id, bench)
        resolved = [a for a in attempts if a.grade.resolved]
        results.append({
            "instance": instance_id,
            "resolved": bool(resolved),
            "resolved_by": [a.name for a in resolved],
            "attempts": [{"name": a.name, "status": a.outcome.status,
                          "pairs": a.outcome.checkpoints, "score": a.score,
                          "resolved": a.grade.resolved,
                          "f2p_pass": a.grade.f2p_pass,
                          "p2p_ran": a.grade.p2p_ran,
                          "p2p_pass": a.grade.p2p_pass,
                          "grading_error": a.grade.error,
                          "verifier_artifacts": a.grade.verifier_artifacts,
                          "grading_snapshot": a.grade.grading_snapshot,
                          "verifier_artifact_error": a.grade.verifier_artifact_error,
                          "patch_lines": a.grade.patch.count("\n"),
                          "network_policy": {
                              "agent": "recorded" if getattr(args, "parent_from", None) and a.name == "parent"
                              else network_policy_for(args, bench, "agent"),
                              "verifier": network_policy_for(args, bench, "verifier")},
                          "journal": str(a.outcome.journal_path)}
                         for a in attempts],
        })
        # Written after every instance, not at the end: a run of eight that dies
        # on the sixth should still report the five it finished.
        (out_dir / "summary.json").write_text(json.dumps(
            {"slot": args.slot, "model": args.model, "subset": args.subset,
             "benchmark": bench.name, "tasks_dir": args.tasks_dir,
             "parent_from": getattr(args, "parent_from", None),
             "fork_conversation": ("full" if getattr(args, "fork_full_conversation", False)
                                   else "truncated_at_fork_step"),
             "timeout": args.timeout, **network_summary(args, bench),
             "sandbox_ttl": backend_for(args, bench)["microvm"]["sandbox_ttl"],
             "branch_schedule": schedule, "rounds": args.rounds,
             "branch_schedule_semantics": ("exact_counts" if args.branch_count_mode == "fixed" else "upper_bounds"),
             "branch_count_mode": args.branch_count_mode,
             "branch_guidance": args.branch_guidance,
             "reviewer_max_attempts": args.reviewer_max_attempts,
             "branch_policy": "per-branch",
             "instances": results,
             **(bench.summary(results, expected_ids=wanted) if hasattr(bench, "summary") else {})}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    print("\n" + "=" * 72)
    solved = [r for r in results if r["resolved"]]
    print("RESOLVED %d / %d" % (len(solved), len(results)))
    for r in results:
        mark = "PASS" if r["resolved"] else "fail"
        best = max((a["score"] for a in r["attempts"]), default=0)
        print("  %-4s %-34s best score %d  %s"
              % (mark, r["instance"], best,
                 ", ".join(r["resolved_by"]) or ""))
    print("\nwrote %s" % (out_dir / "summary.json"))
    if hasattr(bench, "summary") and not bench.summary(results, expected_ids=wanted).get("grading_complete", True):
        print("Grading incomplete; resolved count is a lower bound. Inspect grading_error_ids.")
        return 2
    return 0 if len(solved) == len(results) else 1


def run_one(orch: Orchestrator, args, raw, schedule: List[int],
            out_dir: Path, bench: "Optional[Benchmark]" = None) -> List["Attempt"]:
    """One instance: attempt, grade, and branch until resolved or out of rounds."""
    count_mode = getattr(args, "branch_count_mode", "adaptive")
    guidance_mode = getattr(args, "branch_guidance", "user-hint")
    reviewer_max_attempts = getattr(args, "reviewer_max_attempts", 3)
    if type(reviewer_max_attempts) is not int or reviewer_max_attempts < 1:
        raise ValueError("reviewer_max_attempts must be a positive integer")
    validate_guidance(guidance_mode, args.slot, bool(getattr(args, "fork_full_conversation", False)))
    if count_mode not in BRANCH_COUNT_MODES:
        raise ValueError("unknown branch count mode: %r" % count_mode)
    bench = bench or SweBench()
    out_dir.mkdir(parents=True, exist_ok=True)
    instance = bench.instance(raw)
    instance["slot"] = args.slot
    instance["agent_network"] = network_policy_for(args, bench, "agent")
    resources = bench.resources(instance)
    print("== %s (%s) ==" % (instance["instance_id"], instance["repo"]))
    print("   image %s" % instance["image"])
    print("   F2P %d · P2P %d · slot %s · model %s%s%s"
          % (len(instance["f2p"]), len(instance["p2p"]), args.slot, args.model,
             " · offline" if instance["agent_network"] == "deny" else "",
             " · %s" % resources if resources else ""))
    attempts: List[Attempt] = []

    print("\n== attempt: parent ==")
    started = time.time()
    source = getattr(args, "parent_from", None)
    recorded = existing_parent(source, instance["instance_id"]) if source else None
    if recorded is not None:
        # Same file name the loop would have written, so fork_plan, the
        # analysts' transcript rendering and --regrade all find it here.
        import shutil
        target = out_dir / "parent.jsonl"
        if recorded.resolve() != target.resolve():
            shutil.copyfile(recorded, target)
        print("   reused    %s" % recorded)
        outcome = outcome_from_journal(target)
    elif source:
        raise SystemExit("--parent-from %s has no parent journal for %s"
                         % (source, instance["instance_id"]))
    else:
        image = instance["image"]
        prepare_image = getattr(bench, "prepare_image", None)
        if prepare_image is not None:
            image = prepare_image(instance, backend_for(args, bench), out_dir / "preparation")
        outcome = run_attempt(orch, args, instance, name="parent",
                              prompt=bench.prompt(instance),
                              image=image, out_dir=out_dir,
                              resources=resources, bench=bench)
    if guidance_mode in {"assistant-turn", "none"}:
        require_mini_parent(outcome.journal_path)
    parent = Attempt("parent", outcome, grade_attempt(outcome, instance, args, bench))
    attempts.append(parent)
    report(parent)
    print("   wall       %.0fs" % (time.time() - started))
    if getattr(bench, "stop_on_grading_error", False) and (
            parent.grade.error or parent.grade.verifier_artifact_error):
        return attempts

    case_reports: dict = {}
    by_name: dict = {"parent": parent}
    for round_no in range(1, args.rounds + 1):
        if any(a.grade.resolved for a in attempts):
            print("\n== resolved; no further rounds ==")
            break
        width = schedule[min(round_no - 1, len(schedule) - 1)]
        print("\n== round %d (%s %d branches) ==" %
              (round_no, "exactly" if count_mode == "fixed" else "up to", width))
        points = {a.name: available_branch_points(
            a.outcome.journal_path,
            full_conversation=bool(getattr(args, "fork_full_conversation", False))) for a in attempts}

        for attempt in attempts:
            if attempt.name in case_reports:
                continue
            transcript, lo, hi = render_transcript(
                attempt.outcome.journal_path, token_budget=args.analyst_tokens)
            if hi < 1:
                case_reports[attempt.name] = {
                    "failure_reason": "no tool steps recorded", "steps": 0,
                    "lesson": "", "salvage": "nothing", "branch_candidates": []}
                continue
            print("   analysing %s (%d steps)..." % (attempt.name, hi))
            try:
                case = extract_json(ask_analyst(
                    args.analyst_model, _CASE_PROMPT.format(
                        problem=instance["problem"][:20000],
                        verdict=attempt.verdict_text(), transcript=transcript,
                        checkpoint_steps=json.dumps(sorted(s for s in points[attempt.name] if s <= hi)),
                        candidate_limit=width, lo=lo, hi=hi)))
            except Exception as exc:
                case = {"failure_reason": "analysis failed: %s" % exc,
                        "lesson": "", "salvage": "unknown", "branch_candidates": []}
            case["steps"] = hi
            case_reports[attempt.name] = case
            print("      %s" % str(case.get("failure_reason"))[:150])

        points = {name: {step: pair for step, pair in pairs.items()
                         if step <= case_reports.get(name, {}).get("steps", 0)}
                  for name, pairs in points.items()}
        plan_path = out_dir / ("plan-round%d.json" % round_no)
        plan_record = {
            "reports": case_reports, "branch_policy": "per-branch",
            "branch_limit": width, "branch_count_mode": count_mode,
            "branch_guidance": guidance_mode,
            "reviewer_max_attempts": reviewer_max_attempts,
            "available_steps": {name: sorted(pairs) for name, pairs in points.items()},
        }
        if not any(points.values()):
            plan_record.update(review=None, validation_error="no eligible snapshot/session pairs: no native conversation cut at a complete-turn boundary")
            plan_path.write_text(json.dumps(plan_record, indent=2, ensure_ascii=False), encoding="utf-8")
            print("   no eligible snapshot/session pairs -- stopping")
            break

        reports = [
            {**case_reports.get(a.name, {}), "name": a.name, "round": a.round_no,
              "hint_given": a.hint or None, "grade": a.grade.summary(),
              "available_steps": sorted(points[a.name])} for a in attempts]
        review_prompt = _REVIEW_PROMPT
        try:
            if guidance_mode in {"assistant-turn", "none"}:
                review_prompt = ASSISTANT_REVIEW_PROMPT if guidance_mode == "assistant-turn" else POINT_REVIEW_PROMPT
                for report_row, attempt in zip(reports, attempts):
                    report_row.pop("hint_given")
                    report_row.update(
                        assistant_turn_given=attempt.assistant_turn,
                        native_history=reviewer_context(attempt.outcome.journal_path, points[attempt.name]))
            reports_text = json.dumps(reports, indent=1, ensure_ascii=False)
            parse_review = extract_json if guidance_mode == "user-hint" else extract_branch_plan
            prompt = review_prompt.format(
                problem=instance["problem"][:20000],
                reports=reports_text, count_rule=branch_count_rule(count_mode, width))
        except Exception as exc:
            plan_record.update(review=None, validation_error="reviewer failed: %s" % exc)
            plan_path.write_text(json.dumps(plan_record, indent=2, ensure_ascii=False), encoding="utf-8")
            print("   reviewer failed: %s -- stopping" % exc)
            break
        full_conversation = bool(getattr(args, "fork_full_conversation", False))
        result = review_with_feedback(
            lambda text: ask_analyst(args.analyst_model, text), prompt, parse_review,
            lambda plan: prepare_branches(
                plan, limit=width, round_no=round_no, attempts=by_name,
                checkpoints=points, full_conversation=full_conversation, count_mode=count_mode,
                guidance_mode=guidance_mode),
            max_attempts=reviewer_max_attempts, record=plan_record,
            persist=lambda data: plan_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"))
        if result is None:
            print("   reviewer did not produce a valid plan: %s" % plan_record.get("validation_error"))
            break
        plan, choices = result
        plan_record["selected_branches"] = [
            {"run_name": choice.run_name, "base": choice.base.name,
             "branch_step": choice.checkpoint.step,
             "snapshot_id": choice.checkpoint.snapshot_id,
             "conversation_cut": choice.cut,
             "conversation_restore": "original-prefix" if choice.prefix else "native",
             "branch_guidance": guidance_mode,
             **({"assistant_turn": choice.assistant_turn} if choice.assistant_turn is not None else {})}
            for choice in choices]
        plan_path.write_text(json.dumps(plan_record, indent=2, ensure_ascii=False), encoding="utf-8")
        print("   reviewer selected %d/%d branches" % (len(choices), width))
        print("   synthesis   %s" % str(plan.get("synthesis"))[:200])
        if not choices:
            print("   no useful branches selected -- stopping")
            break

        round_attempts: List[Attempt] = []
        for choice in choices:
            name, base, checkpoint = choice.run_name, choice.base, choice.checkpoint
            print("\n== attempt: %s ==" % name)
            print("   base %s @ step %d -- %s" % (base.name, checkpoint.step, choice.why[:160]))
            print("   guidance  %s" % (
                choice.assistant_turn["content"] if choice.assistant_turn is not None else choice.hint
            )[:200].replace("\n", " "))
            started = time.time()
            resume_session = checkpoint.session_ckpt
            actor_cwd = None
            prefix_origin = {}
            if choice.prefix is not None:
                try:
                    prepared = prepare_prefix(
                        choice.prefix, out_dir / "actor-workspaces" / name,
                        out_dir / "conversation-prefixes" / name, CLAUDE_PROJECTS_DIR)
                except (OSError, ValueError, ImportError) as error:
                    plan_record["validation_error"] = "prefix preparation failed: %s" % error
                    plan_path.write_text(json.dumps(plan_record, indent=2, ensure_ascii=False))
                    raise
                resume_session = prepared["resume_session_id"]
                actor_cwd = Path(prepared["cwd"])
                prefix_origin = {"conversation_restore": "original-prefix",
                                 "source_session_id": checkpoint.session_ckpt,
                                 "resume_session_id": resume_session,
                                 "conversation_prefix_manifest": prepared["manifest_path"]}
                selected = next(item for item in plan_record["selected_branches"] if item["run_name"] == name)
                selected.update(prefix_origin)
                plan_path.write_text(json.dumps(plan_record, indent=2, ensure_ascii=False))
            outcome = run_attempt(
                orch, args, instance, name=name, out_dir=out_dir,
                prompt="" if guidance_mode != "user-hint" else bench.branch_prompt(
                    instance, verdict="", hint=choice.hint,
                    truncated=choice.cut is not None, step=checkpoint.step),
                image=checkpoint.snapshot_id, resume=resume_session, cwd=actor_cwd,
                fork=True, resume_at=choice.cut,
                origin={"parent_run_id": base.name, "branch_step": checkpoint.step,
                        "parent_journal": str(base.outcome.journal_path),
                        "snapshot_id": checkpoint.snapshot_id,
                        "conversation_cut": choice.cut,
                        "cut_note": "explicit-full-conversation" if full_conversation else None,
                        "actor_hint": choice.hint,
                        "hint_delivery": guidance_mode if guidance_mode != "user-hint" else "reviewer-direct",
                        "branch_guidance": guidance_mode,
                        **({"assistant_turn": choice.assistant_turn, "assistant_turn_source": "reviewer"}
                           if choice.assistant_turn is not None else {}),
                        "branch_policy": "per-branch", "branch_count_mode": count_mode,
                        "selection_reason": choice.why,
                        "round": round_no, "direction": name.split("-", 1)[-1],
                        **prefix_origin},
                bench=bench, **({"assistant_turn": choice.assistant_turn}
                               if choice.assistant_turn is not None else {}),
                **({"resume_without_hint": True} if guidance_mode == "none" else {}))
            if choice.cut is not None and "No message found with message.uuid" in str(outcome.error or ""):
                grade = Grade(error="selected native cut refused; no full-conversation retry")
            else:
                grade = grade_attempt(outcome, instance, args, bench)
            attempt = Attempt(name, outcome, grade, plan, hint=choice.hint, round_no=round_no,
                              assistant_turn=choice.assistant_turn)
            round_attempts.append(attempt)
            attempts.append(attempt)
            by_name[name] = attempt
            report(attempt)
            print("   wall       %.0fs" % (time.time() - started))
            if getattr(bench, "stop_on_grading_error", False) and (
                    grade.error or grade.verifier_artifact_error):
                return attempts

        winner = max(round_attempts, key=lambda a: a.score)
        print("\n   round %d best: %s (score %d)" % (round_no, winner.name, winner.score))

    print("\n== %s: %s ==" % (instance["instance_id"],
                               "RESOLVED" if any(a.grade.resolved for a in attempts)
                               else "unresolved"))
    for attempt in attempts:
        print("  %-30s score=%d  %s" % (attempt.name, attempt.score,
                                        attempt.grade.summary()))
    return attempts


if __name__ == "__main__":
    raise SystemExit(main())
