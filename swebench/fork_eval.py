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
3. **Branch on failure.** An analyst model reads the journal as a step-by-step
   transcript plus the grading verdict, picks the step to branch from and K
   diverse directions; each direction becomes another attempt whose sandbox image
   IS that step's snapshot and whose conversation forks the parent's. Grade each.
4. Repeat from the best-scoring attempt, up to ``--rounds``.

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

from harness.core.journal import read_journal
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
PROMPT = """\
You are fixing a bug in the {repo} repository, checked out at /testbed inside \
your sandbox. Use ONLY your ash MCP tools (shell, text_editor) -- your own local
shell cannot see the repository.

## Problem statement
{problem}

## What counts as done
The project's own test suite must accept your change. Find the tests that cover \
this behaviour, run them, and iterate until they pass. Do not edit tests to make \
them pass. Do not create new files unless the fix genuinely needs one.

Work efficiently: read before you edit, and prefer a minimal, targeted change \
over a sweeping refactor.
"""

BRANCH_PROMPT = """\
You are continuing work on a bug in the {repo} repository at /testbed in your \
sandbox. The filesystem already holds an earlier attempt's work -- ITS EDITS ARE \
ON DISK. You do not have that attempt's conversation, so trust the files, not \
your memory.

## Problem statement
{problem}

## What the earlier attempt produced, and how it was graded
{verdict}

## Your direction for this attempt
{hint}

Use ONLY your ash MCP tools. Verify with the project's own tests, and do not \
edit tests to make them pass.
"""


# --- analyst ---------------------------------------------------------------
_ANALYSIS_PROMPT = """\
You are deciding where to BRANCH a fresh attempt at a software bug fix, from \
per-step environment snapshots of an earlier attempt that FAILED its grading.

## The problem the attempt was solving
{problem}

## How it was graded (this is ground truth the attempt itself could not see)
{verdict}

## The attempt, one line per tool step ("[N] tool(args) -> result")
{transcript}

## Rules
- Valid branch steps: integers {lo}..{hi} that appear in the transcript. A branch
  at step N resumes from the environment AS IT STOOD AFTER step N.
- LATER IS BETTER: every step kept preserves work. But the step must be strictly
  BEFORE the decisive wrong turn -- branching after it inherits the mistake.
  Weigh both; do not reflexively pick the middle or the end.
- The resumed agent gets the ENVIRONMENT ONLY -- files as they were after that
  step. It has none of this conversation. Every hint must be self-contained: say
  what is already on disk and trustworthy, what this attempt learned (including
  from its failures), and what to do differently.
- Produce {branches} genuinely DIVERSE directions: different hypotheses about the
  failure, or different strategies (fix-forward, revert-and-redo the flawed part,
  narrower change, different verification discipline). Not paraphrases. They run
  independently, so it is fine for them to disagree.
{notes}

Return ONLY a JSON object, no prose:
{{"branch_step": <int>, "why_here": "<why this step, and why not later>",
  "what_went_wrong": "<diagnosis>",
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
            session.execute("text_editor", {
                "command": "write", "path": "/tmp/.swebench_test.patch",
                "file_text": test_patch})
            applied = session.execute("shell", {
                "command": "cd /testbed && git apply /tmp/.swebench_test.patch",
                "timeout": 120})
            if not _exit_ok(applied):
                grade.error = ("test_patch did not apply -- the attempt's edits "
                               "collided with the graded tests")
                return grade

        f2p = parse_test_list(instance.get("f2p") or instance.get("FAIL_TO_PASS"))
        p2p = parse_test_list(instance.get("p2p") or instance.get("PASS_TO_PASS"))
        repo = instance["repo"]

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
                fork: bool = False, origin: Optional[dict] = None) -> RunOutcome:
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
        backend=backend_for(args), runtime_bin=runtime_bin,
        sandbox_image=image,
        resume_session_id=resume, fork=fork, origin=origin, extra=extra,
    )
    return orch.run(spec)


def backend_for(args) -> dict:
    return {"backend": "microvm",
            "microvm": {"from_image": True,
                        "runtime_bin": str(Path(args.runtime_bin).resolve())}}


def grade_attempt(outcome: RunOutcome, instance: dict, args) -> Grade:
    checkpoints = load_checkpoints(outcome.journal_path)
    usable = [c for c in checkpoints if c.snapshot_id]
    if not usable:
        return Grade(error="no snapshot recorded -- nothing to grade")
    return grade_snapshot(usable[-1].snapshot_id, instance, backend_for(args))


def report(attempt: Attempt) -> None:
    print("   status     %s%s" % (attempt.outcome.status,
                                  " (%s)" % attempt.outcome.error
                                  if attempt.outcome.error else ""))
    print("   pairs      %d" % attempt.outcome.checkpoints)
    print("   grade      %s" % attempt.grade.summary())
    print("   patch      %d 行" % attempt.grade.patch.count("\n"))


def regrade(args, out_dir: Path) -> int:
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
    catalogue = {i["instance_id"]: i for i in load_instances(args.subset)}
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
        instance = {
            "instance_id": instance_id, "repo": raw["repo"],
            "f2p": parse_test_list(raw["FAIL_TO_PASS"]),
            "p2p": parse_test_list(raw["PASS_TO_PASS"]),
            "test_files": test_files_of(raw),
            "test_patch": raw.get("test_patch") or "",
        }
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
            grade = grade_snapshot(pairs[-1].snapshot_id, instance,
                                  backend_for(args))
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
    parser.add_argument("--regrade", action="store_true",
                        help="re-grade a finished run in -o with the current "
                             "grader, spending no agent time: every attempt left "
                             "a snapshot, so a grader fix can be applied to "
                             "results that already exist")
    args = parser.parse_args(argv)
    try:
        schedule = [int(x) for x in str(args.branches).split(",") if x.strip()]
    except ValueError:
        raise SystemExit("--branches wants numbers, got %r" % args.branches)
    if not schedule:
        raise SystemExit("--branches cannot be empty")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.regrade:
        return regrade(args, out_dir)
    if not args.instance:
        raise SystemExit("--instance is required (or use --regrade)")

    wanted = [x.strip() for x in str(args.instance).split(",") if x.strip()]
    catalogue = {i["instance_id"]: i for i in load_instances(args.subset)}
    missing = [x for x in wanted if x not in catalogue]
    if missing:
        raise SystemExit("not in %s: %s" % (args.subset, ", ".join(missing)))

    orch = Orchestrator(out_dir=out_dir)
    results = []
    for position, instance_id in enumerate(wanted, 1):
        print("\n" + "=" * 72)
        print("INSTANCE %d/%d  %s" % (position, len(wanted), instance_id))
        print("=" * 72)
        attempts = run_one(orch, args, catalogue[instance_id], schedule,
                           out_dir / instance_id)
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


def run_one(orch: Orchestrator, args, raw: dict, schedule: List[int],
            out_dir: Path) -> List["Attempt"]:
    """One instance: attempt, grade, and branch until resolved or out of rounds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    instance = {
        "instance_id": raw["instance_id"], "repo": raw["repo"],
        "image": resolve_image(raw), "problem": raw["problem_statement"],
        "f2p": parse_test_list(raw["FAIL_TO_PASS"]),
        "p2p": parse_test_list(raw["PASS_TO_PASS"]),
        "test_files": test_files_of(raw),
        "test_patch": raw.get("test_patch") or "",
    }
    print("== %s (%s) ==" % (instance["instance_id"], instance["repo"]))
    print("   image %s" % instance["image"])
    print("   F2P %d · P2P %d · slot %s · model %s"
          % (len(instance["f2p"]), len(instance["p2p"]), args.slot, args.model))
    attempts: List[Attempt] = []

    print("\n== attempt: parent ==")
    started = time.time()
    outcome = run_attempt(orch, args, instance, name="parent",
                          prompt=PROMPT.format(repo=instance["repo"],
                                               problem=instance["problem"]),
                          image=instance["image"], out_dir=out_dir)
    parent = Attempt("parent", outcome, grade_attempt(outcome, instance, args))
    attempts.append(parent)
    report(parent)
    print("   wall       %.0fs" % (time.time() - started))

    best = parent
    notes: List[str] = []
    for round_no in range(1, args.rounds + 1):
        if best.grade.resolved:
            print("\n== resolved; no further rounds ==")
            break
        # A round's width comes from the schedule; the last entry repeats if
        # there are more rounds than numbers.
        width = schedule[min(round_no - 1, len(schedule) - 1)]
        print("\n== round %d (%d branches): analysing %s =="
              % (round_no, width, best.name))
        transcript, lo, hi = render_transcript(best.outcome.journal_path,
                                              token_budget=args.analyst_tokens)
        if hi < 1:
            print("   no tool steps to branch from; stopping")
            break
        broken = ""
        if best.grade.broken:
            broken = ("\n\nTests this attempt BROKE (they passed before it): %s"
                      % ", ".join(best.grade.broken))
        verdict = "%s%s\n\nPatch (%d lines):\n%s\n\nTest output:\n%s" % (
            best.grade.summary(), broken, best.grade.patch.count("\n"),
            best.grade.patch[:40000], best.grade.detail[:20000])
        prompt = _ANALYSIS_PROMPT.format(
            problem=instance["problem"][:20000], verdict=verdict,
            transcript=transcript, lo=lo, hi=hi, branches=width,
            notes=("\n## What earlier rounds already tried (do not repeat)\n"
                   + "\n".join(notes)) if notes else "")
        try:
            plan = extract_json(ask_analyst(args.analyst_model, prompt))
        except Exception as exc:  # noqa: BLE001
            print("   analyst failed: %s -- stopping" % exc)
            break
        step = int(plan.get("branch_step") or hi)
        step = max(lo, min(hi, step))
        print("   branch_step %d — %s" % (step, str(plan.get("why_here"))[:160]))
        print("   diagnosis   %s" % str(plan.get("what_went_wrong"))[:200])
        (out_dir / ("plan-round%d.json" % round_no)).write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

        try:
            pair = fork_plan(best.outcome.journal_path, step)
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
                prompt=BRANCH_PROMPT.format(
                    repo=instance["repo"], problem=instance["problem"],
                    verdict=verdict[:20000], hint=branch.get("hint")),
                image=pair["snapshot_id"],
                resume=pair.get("session_ckpt"), fork=True,
                origin={"parent_run_id": best.name, "branch_step": step,
                        "snapshot_id": pair["snapshot_id"],
                        "round": round_no, "direction": branch.get("name")})
            attempt = Attempt(name, outcome,
                              grade_attempt(outcome, instance, args), plan)
            round_attempts.append(attempt)
            attempts.append(attempt)
            report(attempt)
            print("   wall       %.0fs" % (time.time() - started))
            notes.append("- round %d %s: %s -> %s" % (
                round_no, branch.get("name"),
                str(branch.get("hint"))[:120].replace("\n", " "),
                attempt.grade.summary()))

        if round_attempts:
            winner = max(round_attempts, key=lambda a: a.score)
            if winner.score >= best.score:
                best = winner
            print("\n   round %d best: %s (score %d); carrying %s forward"
                  % (round_no, winner.name, winner.score, best.name))

    print("\n== %s: %s ==" % (instance["instance_id"],
                               "RESOLVED" if any(a.grade.resolved for a in attempts)
                               else "unresolved"))
    for attempt in attempts:
        print("  %-30s score=%d  %s" % (attempt.name, attempt.score,
                                        attempt.grade.summary()))
    return attempts


if __name__ == "__main__":
    raise SystemExit(main())
