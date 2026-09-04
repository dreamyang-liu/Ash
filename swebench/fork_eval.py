#!/usr/bin/env python3
"""Run one SWE-bench instance through the orchestrator, grade it, branch on failure.

The eval driver for the checkpointed path: it is the piece that turns "we can
snapshot every step and fork any step" into a *score*, by adding the two things
the execution plane deliberately does not know -- what the answer is (a patch
that makes FAIL_TO_PASS pass without breaking PASS_TO_PASS) and what to do when
the answer is wrong.

    python -m swebench.fork_eval --instance sympy__sympy-13091 \
        --slot codex --model openai.gpt-5.6-luna \
        --rounds 2 --branches 3 -o runs/fork-eval

The loop:

1. **Attempt.** One orchestrator run. Every mutating step leaves a rollback pair
   (env snapshot + conversation ref) in the journal.
2. **Grade.** Restore the LAST snapshot into a fresh microVM and run the tests
   there. Grading in a restored sandbox rather than the live one is deliberate:
   it proves the snapshot carries the work, and it lets grading happen after the
   agent's sandbox is gone.
3. **Branch on failure**, in two analyst stages. *Map:* every failed attempt is
   analysed separately -- its transcript, its verdict, the hint it was given --
   into a failure_reason, a lesson, and candidate branch steps. *Reduce:* a
   reviewer reads ALL the analyses (parent included) and picks ONE base attempt
   + step + K divergent directions. Each direction becomes another attempt whose
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

from harness.core.journal import read_journal, volatile_reason
from harness.orchestrator.run import Orchestrator, RunOutcome, RunSpec
from harness.rollback import fork_plan, load_checkpoints
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

BRANCH_PROMPT = """\
You are continuing work on a bug in the {repo} repository at /testbed in your \
sandbox. The filesystem already holds an earlier attempt's work -- ITS EDITS ARE \
ON DISK. Start from the files as they are: run `git diff` in /testbed to see \
exactly what the earlier attempt changed before you touch anything.

## Problem statement
{problem}

## What the earlier attempt produced, and how it was graded
{verdict}

## Your direction for this attempt
{hint}

## Workflow

1. `git diff` in /testbed -- know what is already changed.
2. Reproduce the remaining failure with a small script before editing.
3. Follow your direction above: fix forward or revert-and-redo, but keep the \
change minimal.
4. Re-run the reproduction, probe edge cases, then the project's own tests.

Do NOT edit test files -- grading restores the official tests and discards your \
edits to them.

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
You are analysing ONE failed attempt at a software bug fix. Your analysis will
be pooled with analyses of the other attempts, and a reviewer will decide where
a fresh attempt should branch from this attempt's per-step snapshots.

## The problem it was solving
{problem}

## The direction this attempt was given (empty if it was the original attempt)
{hint}

## How it was graded (ground truth the attempt itself could not see)
{verdict}

## The attempt, one line per tool step ("[N] tool(args) -> result")
{transcript}

## Rules
- Valid steps: integers {lo}..{hi}. A branch at step N resumes from the
  environment AS IT STOOD AFTER step N.
- branch_candidates: up to 3 steps worth branching from, LATER IS BETTER (every
  step kept preserves work) but strictly BEFORE this attempt's decisive wrong
  turn. If the whole attempt was poisoned from the start, say so with step {lo}.
- Be specific about MECHANISM: "changed __eq__ but numeric subclasses override
  it" is a failure_reason; "the fix was incomplete" is not.

Return ONLY a JSON object, no prose:
{{"failure_reason": "<the mechanism, specific>",
  "lesson": "<what the next attempt must know that this one proved>",
  "salvage": "<what on this attempt's disk is worth keeping, or 'nothing'>",
  "branch_candidates": [{{"step": <int>, "why": "<one sentence>"}}]}}
"""

_REVIEW_PROMPT = """\
You are the REVIEWER for a failed software bug fix. Several attempts have been
made; each failed attempt has been analysed separately below. Decide where the
next round of {branches} parallel attempts should start.

## The problem
{problem}

## Every attempt so far ("parent" is the original; branches were given a hint)
{reports}

## Rules
- Pick ONE base attempt and a branch_step from ITS branch_candidates (you may
  choose a different step of the same attempt if the analyses justify it). The
  new attempts resume from that attempt's environment after that step, and they
  inherit that attempt's conversation up to it.
- Going BACK to an earlier attempt (including parent) is legitimate when later
  attempts are deeper in a dead end -- weigh salvage against contamination.
- Synthesise across the analyses: a hypothesis two attempts each half-proved is
  the most valuable thing you can hand the next round.
- Produce {branches} genuinely DIVERSE directions, disjoint from every
  hint_given above. Each hint must be self-contained (4-10 sentences): what is
  on the base's disk and trustworthy, what the pooled lessons established, and
  what to do differently.

Return ONLY a JSON object, no prose:
{{"base": "<attempt name>", "branch_step": <int>,
  "why": "<why this base and step>",
  "synthesis": "<the pooled diagnosis>",
  "branches": [{{"name": "<slug>", "hint": "<4-10 sentences>"}}]}}
"""


#: A diagnosis plus K self-contained hints (each 4-10 sentences that must stand
#: alone without the conversation) does not fit in 4k, and a truncated JSON object
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
    session = SandboxSession(quiet=True, backend=dict(backend))
    if not session.create(snapshot_id):
        grade.error = "could not restore %s: %s" % (snapshot_id,
                                                    session.create_error)
        return grade
    try:
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
        session.destroy()
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
        result = session.execute("shell", {"command": command,
                                           "timeout": int(timeout)})
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
    return session.execute("shell", {"command": command, "timeout": timeout,
                                     "tail": 60})


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
                bench: "Optional[Benchmark]" = None) -> RunOutcome:
    """One attempt. Parent and branches differ only in the arguments.

    ``out_dir`` is per instance: eight instances writing `parent.jsonl` into one
    directory would overwrite each other's journals, and the journal is the only
    record a killed run leaves.
    """
    runtime_bin = str(Path(args.runtime_bin).resolve())
    extra: dict = {}
    if args.slot == "codex":
        # Native Bedrock provider: OpenAI's own models are hosted there, so no
        # translator and no login. Pre-serialized TOML values, which is what
        # codex's -c overrides take.
        extra["config_overrides"] = {"model_provider": '"amazon-bedrock"'}
    if args.slot.startswith("opencode"):
        extra["data_home"] = str(out_dir / "state" / "shared")
    spec = RunSpec(
        prompt=prompt, slot=args.slot, cwd="/tmp", model=args.model,
        timeout_s=args.timeout, run_id=name,
        journal_path=out_dir / ("%s.jsonl" % name),
        transport="http", tools="default",
        backend=backend_for(args, bench), runtime_bin=runtime_bin,
        sandbox_image=image, sandbox_resources=resources,
        resume_session_id=resume, fork=fork, origin=origin, extra=extra,
    )
    return orch.run(spec)


def backend_for(args, bench: "Optional[Benchmark]" = None) -> dict:
    microvm: dict = {"from_image": True,
                     "runtime_bin": str(Path(args.runtime_bin).resolve())}
    if bench is not None and bench.no_network:
        # One policy for the attempt AND the grading VM: a benchmark that says
        # no-network means the verifier ran offline too.
        microvm["allow_internet"] = False
    if bench is not None and bench.image_env:
        microvm["image_env"] = True
    return {"backend": "microvm", "microvm": microvm}


def grade_attempt(outcome: RunOutcome, instance: dict, args,
                  bench: "Optional[Benchmark]" = None) -> Grade:
    checkpoints = load_checkpoints(outcome.journal_path)
    usable = [c for c in checkpoints if c.snapshot_id]
    if not usable:
        return Grade(error="no snapshot recorded -- nothing to grade")
    bench = bench or SweBench()
    return bench.grade(usable[-1].snapshot_id, instance, backend_for(args, bench))


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

    def branch_prompt(self, instance: dict, verdict: str, hint: str) -> str:
        raise NotImplementedError

    def resources(self, instance: dict) -> Optional[dict]:
        return None

    def grade(self, snapshot_id: str, instance: dict, backend: dict) -> Grade:
        raise NotImplementedError


class SweBench(Benchmark):
    name = "swebench"

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
        return PROMPT.format(repo=instance["repo"], problem=instance["problem"],
                             primer=TOOL_PRIMER)

    def branch_prompt(self, instance: dict, verdict: str, hint: str) -> str:
        return BRANCH_PROMPT.format(repo=instance["repo"], problem=instance["problem"],
                                    verdict=verdict, hint=hint, primer=TOOL_PRIMER)

    def grade(self, snapshot_id: str, instance: dict, backend: dict) -> Grade:
        return grade_snapshot(snapshot_id, instance, backend)


def select_benchmark(args) -> Benchmark:
    name = str(getattr(args, "benchmark", "") or "swebench").lower()
    if name == "swebench":
        return SweBench()
    if name == "deepswe":
        from deepswe.bench import DeepSWE
        return DeepSWE(getattr(args, "tasks_dir", None))
    raise SystemExit("unknown --benchmark %r; choose swebench or deepswe" % name)


# --- reusing a recorded parent -----------------------------------------------
#
# Branching is about the FAILED trajectory: its snapshots and its verdict are
# what the analysts fork from. Re-running the parent first (what branch134 did,
# because its prompt had changed) is a blind retry that costs a full attempt
# per task and tells the branches nothing. With the prompt unchanged, the
# recorded single-pass journal IS the parent: copy it into the run directory,
# grade its last snapshot, and go straight to round 1.

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
            if not pairs:
                print("   %-30s no snapshot" % journal.stem)
                continue
            grade = bench.grade(pairs[-1].snapshot_id, instance,
                                backend_for(args, bench))
            print("   %-30s %s" % (journal.stem, grade.summary()))
            verdicts.append({"name": journal.stem, "resolved": grade.resolved,
                             "f2p_pass": grade.f2p_pass,
                             "p2p_ran": grade.p2p_ran,
                             "p2p_pass": grade.p2p_pass,
                             "skipped_ids": len(grade.skipped_ids),
                             "broken": grade.broken,
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
             "instances": results}, indent=2, ensure_ascii=False),
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
                        help="branches per round: one number for every round, or "
                             "a comma list to vary it (e.g. 4,3 -- four "
                             "directions first, three in the round after). A "
                             "round only happens if the one before it produced "
                             "no resolved attempt.")
    parser.add_argument("--analyst-tokens", type=int, default=100_000,
                        help="transcript budget handed to the analyst")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
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
                        choices=["swebench", "deepswe"],
                        help="which benchmark supplies tasks, prompts and the "
                             "grader (default: swebench, unchanged behaviour)")
    parser.add_argument("--tasks-dir", default=None,
                        help="deepswe: the dataset's tasks/ directory")
    parser.add_argument("--parent-from", default=None,
                        help="reuse each instance's recorded parent journal from "
                             "this batch dir or aggregate .json instead of running "
                             "a fresh parent: grade its last snapshot, then branch. "
                             "Refuses an instance that has none.")
    args = parser.parse_args(argv)
    bench = select_benchmark(args)
    try:
        schedule = [int(x) for x in str(args.branches).split(",") if x.strip()]
    except ValueError:
        raise SystemExit("--branches wants numbers, got %r" % args.branches)
    if not schedule:
        raise SystemExit("--branches cannot be empty")

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
                          "patch_lines": a.grade.patch.count("\n"),
                          "journal": str(a.outcome.journal_path)}
                         for a in attempts],
        })
        # Written after every instance, not at the end: a run of eight that dies
        # on the sixth should still report the five it finished.
        (out_dir / "summary.json").write_text(json.dumps(
            {"slot": args.slot, "model": args.model, "subset": args.subset,
             "benchmark": bench.name, "tasks_dir": args.tasks_dir,
             "parent_from": getattr(args, "parent_from", None),
             "timeout": args.timeout, "no_network": bench.no_network,
             "branch_schedule": schedule, "rounds": args.rounds,
             "instances": results}, indent=2, ensure_ascii=False),
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
    return 0 if len(solved) == len(results) else 1


def run_one(orch: Orchestrator, args, raw, schedule: List[int],
            out_dir: Path, bench: "Optional[Benchmark]" = None) -> List["Attempt"]:
    """One instance: attempt, grade, and branch until resolved or out of rounds."""
    bench = bench or SweBench()
    out_dir.mkdir(parents=True, exist_ok=True)
    instance = bench.instance(raw)
    resources = bench.resources(instance)
    print("== %s (%s) ==" % (instance["instance_id"], instance["repo"]))
    print("   image %s" % instance["image"])
    print("   F2P %d · P2P %d · slot %s · model %s%s%s"
          % (len(instance["f2p"]), len(instance["p2p"]), args.slot, args.model,
             " · offline" if bench.no_network else "",
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
        outcome = run_attempt(orch, args, instance, name="parent",
                              prompt=bench.prompt(instance),
                              image=instance["image"], out_dir=out_dir,
                              resources=resources, bench=bench)
    parent = Attempt("parent", outcome, grade_attempt(outcome, instance, args, bench))
    attempts.append(parent)
    report(parent)
    print("   wall       %.0fs" % (time.time() - started))

    case_reports: dict = {}
    by_name: dict = {"parent": parent}
    for round_no in range(1, args.rounds + 1):
        if any(a.grade.resolved for a in attempts):
            print("\n== resolved; no further rounds ==")
            break
        width = schedule[min(round_no - 1, len(schedule) - 1)]
        print("\n== round %d (%d branches) ==" % (round_no, width))

        # -- map: analyse every failed attempt not yet analysed ----------------
        for attempt in attempts:
            if attempt.name in case_reports:
                continue
            transcript, lo, hi = render_transcript(
                attempt.outcome.journal_path, token_budget=args.analyst_tokens)
            if hi < 1:
                case_reports[attempt.name] = {
                    "failure_reason": "no tool steps recorded",
                    "lesson": "", "salvage": "nothing", "branch_candidates": []}
                continue
            print("   analysing %s (%d steps)..." % (attempt.name, hi))
            try:
                case = extract_json(ask_analyst(
                    args.analyst_model, _CASE_PROMPT.format(
                        problem=instance["problem"][:20000],
                        hint=attempt.hint or "(none -- original attempt)",
                        verdict=attempt.verdict_text(), transcript=transcript,
                        lo=lo, hi=hi)))
            except Exception as exc:  # noqa: BLE001
                case = {"failure_reason": "analysis failed: %s" % exc,
                        "lesson": "", "salvage": "unknown",
                        "branch_candidates": []}
            case["steps"] = hi
            case_reports[attempt.name] = case
            print("      %s" % str(case.get("failure_reason"))[:150])

        # -- reduce: one reviewer over all reports ------------------------------
        reports_text = json.dumps(
            [{"name": a.name, "round": a.round_no, "hint_given": a.hint or None,
              "grade": a.grade.summary(), **case_reports.get(a.name, {})}
             for a in attempts], indent=1, ensure_ascii=False)
        try:
            plan = extract_json(ask_analyst(
                args.analyst_model, _REVIEW_PROMPT.format(
                    problem=instance["problem"][:20000],
                    reports=reports_text, branches=width)))
        except Exception as exc:  # noqa: BLE001
            print("   reviewer failed: %s -- stopping" % exc)
            break
        base = by_name.get(str(plan.get("base")))
        if base is None:
            base = max(attempts, key=lambda a: a.score)
            print("   reviewer named unknown base %r; falling back to %s"
                  % (plan.get("base"), base.name))
        hi = case_reports.get(base.name, {}).get("steps", 1)
        step = max(1, min(hi, int(plan.get("branch_step") or hi)))
        print("   base %s @ step %d — %s" % (base.name, step,
                                             str(plan.get("why"))[:160]))
        print("   synthesis   %s" % str(plan.get("synthesis"))[:200])
        (out_dir / ("plan-round%d.json" % round_no)).write_text(
            json.dumps({"reports": case_reports, "review": plan}, indent=2,
                       ensure_ascii=False), encoding="utf-8")

        try:
            pair = fork_plan(base.outcome.journal_path, step)
        except ValueError as exc:
            print("   %s -- stopping" % exc)
            break

        round_attempts: List[Attempt] = []
        for index, branch in enumerate((plan.get("branches") or [])[:width], 1):
            name = "r%db%d-%s" % (round_no, index,
                                  re.sub(r"[^a-z0-9]+", "-",
                                         str(branch.get("name") or "b").lower())[:24])
            print("\n== attempt: %s ==" % name)
            print("   hint  %s" % str(branch.get("hint"))[:200].replace("\n", " "))
            started = time.time()
            outcome = run_attempt(
                orch, args, instance, name=name, out_dir=out_dir,
                prompt=bench.branch_prompt(
                    instance, verdict=base.verdict_text()[:20000],
                    hint=str(branch.get("hint") or "")),
                image=pair["snapshot_id"],
                resume=pair.get("session_ckpt"), fork=True,
                origin={"parent_run_id": base.name, "branch_step": step,
                        "snapshot_id": pair["snapshot_id"],
                        "round": round_no, "direction": branch.get("name")},
                bench=bench)
            attempt = Attempt(name, outcome,
                              grade_attempt(outcome, instance, args, bench), plan,
                              hint=str(branch.get("hint") or ""),
                              round_no=round_no)
            round_attempts.append(attempt)
            attempts.append(attempt)
            by_name[name] = attempt
            report(attempt)
            print("   wall       %.0fs" % (time.time() - started))

        if round_attempts:
            winner = max(round_attempts, key=lambda a: a.score)
            print("\n   round %d best: %s (score %d)"
                  % (round_no, winner.name, winner.score))

    print("\n== %s: %s ==" % (instance["instance_id"],
                               "RESOLVED" if any(a.grade.resolved for a in attempts)
                               else "unresolved"))
    for attempt in attempts:
        print("  %-30s score=%d  %s" % (attempt.name, attempt.score,
                                        attempt.grade.summary()))
    return attempts


if __name__ == "__main__":
    raise SystemExit(main())
