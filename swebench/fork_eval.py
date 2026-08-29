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
                              load_instances, needs_file_runner, parse_test_list,
                              resolve_image, sympy_runner_spec, test_files_of)

#: Mantle serves the OpenAI catalogue on Bedrock; the analyst talks to it
#: directly rather than through a translator, because one JSON call needs no
#: agent scaffolding.
MANTLE = "https://bedrock-mantle.%s.api.aws/openai/v1/responses"


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


def ask_analyst(model: str, prompt: str, region: str = "us-west-2",
                timeout: float = 300.0) -> str:
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not key:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK is required for the analyst")
    body = json.dumps({"model": model, "input": prompt,
                       "max_output_tokens": 4096}).encode()
    request = urllib.request.Request(
        MANTLE % region, data=body,
        headers={"Authorization": "Bearer %s" % key,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    chunks = []
    for item in payload.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") in ("output_text", "text"):
                chunks.append(part.get("text") or "")
    return "\n".join(chunks)


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
def render_transcript(journal_path, char_budget: int = 24000) -> "tuple[str, int, int]":
    """One line per tool step, numbered by the checkpoint step they map to.

    Numbered from the tool calls rather than from the checkpoints, because the
    analyst must name a step the *snapshot map* has -- and both are counted the
    same way (one per exec call, in order).
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
                args[:400]))
        elif kind == "tool.finished" and lines:
            out = str(record.get("output") or "")[:300].replace("\n", " ")
            lines[-1] += "  -> %s" % (out or record.get("status") or "")
        elif kind == "agent.message":
            text = str(record.get("text") or "").replace("\n", " ")
            if text.strip():
                lines.append("    (agent said: %s)" % text[:200])
    body = "\n".join(lines)
    if len(body) > char_budget:
        # Keep the head and the tail: the early steps establish what was
        # understood, the late ones contain the failure.
        half = char_budget // 2
        body = body[:half] + "\n...[middle elided]...\n" + body[-half:]
    return body, 1, step


# --- grading ---------------------------------------------------------------
@dataclass
class Grade:
    resolved: bool = False
    f2p_pass: bool = False
    p2p_pass: bool = False
    patch: str = ""
    detail: str = ""
    error: Optional[str] = None

    #: False until the regression sweep actually ran -- it is skipped when the
    #: target tests fail, and reporting that skip as "FAIL" reads as a
    #: regression that was never measured.
    p2p_ran: bool = False

    def summary(self) -> str:
        if self.error:
            return "GRADING ERROR: %s" % self.error
        regression = ("PASS" if self.p2p_pass else "FAIL") if self.p2p_ran \
            else "not run"
        return ("resolved=%s (target tests %s, regressions %s)"
                % (self.resolved, "PASS" if self.f2p_pass else "FAIL",
                   regression))


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

        files = instance.get("test_files") or None
        runner = _install_runner(session, repo, f2p)

        result = _run_tests(session, repo, f2p, files, runner, timeout)
        grade.f2p_pass = _exit_ok(result)
        grade.detail = "target: %s" % (result.output or result.error or "")[-1200:]

        if grade.f2p_pass and p2p:
            grade.p2p_ran = True
            result = _run_tests(session, repo, p2p, files, runner, timeout)
            grade.p2p_pass = _exit_ok(result)
            grade.detail += "\n\nregressions: %s" % (
                result.output or result.error or "")[-1200:]
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
                prompt: str, image: str, resume: Optional[str] = None,
                fork: bool = False, origin: Optional[dict] = None) -> RunOutcome:
    out_dir = Path(args.out)
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--slot", default="codex")
    parser.add_argument("--model", default="openai.gpt-5.6-luna")
    parser.add_argument("--analyst-model", default="openai.gpt-5.6-luna")
    parser.add_argument("--rounds", type=int, default=2,
                       help="branching rounds after the first attempt")
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    parser.add_argument("-o", "--out", default="runs/fork-eval")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = [i for i in load_instances(args.subset)
               if i["instance_id"] == args.instance]
    if not matches:
        raise SystemExit("instance %s not in %s" % (args.instance, args.subset))
    raw = matches[0]
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

    orch = Orchestrator(out_dir=out_dir)
    attempts: List[Attempt] = []

    print("\n== attempt: parent ==")
    started = time.time()
    outcome = run_attempt(orch, args, instance, name="parent",
                          prompt=PROMPT.format(repo=instance["repo"],
                                               problem=instance["problem"]),
                          image=instance["image"])
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
        print("\n== round %d: analysing %s ==" % (round_no, best.name))
        transcript, lo, hi = render_transcript(best.outcome.journal_path)
        if hi < 1:
            print("   no tool steps to branch from; stopping")
            break
        verdict = "%s\n\nPatch (%d lines):\n%s\n\nTest output:\n%s" % (
            best.grade.summary(), best.grade.patch.count("\n"),
            best.grade.patch[:3000], best.grade.detail[:3000])
        prompt = _ANALYSIS_PROMPT.format(
            problem=instance["problem"][:6000], verdict=verdict,
            transcript=transcript, lo=lo, hi=hi, branches=args.branches,
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
        for index, branch in enumerate((plan.get("branches") or [])[:args.branches], 1):
            name = "r%db%d-%s" % (round_no, index,
                                  re.sub(r"[^a-z0-9]+", "-",
                                         str(branch.get("name") or "b").lower())[:24])
            print("\n== attempt: %s ==" % name)
            print("   hint  %s" % str(branch.get("hint"))[:200].replace("\n", " "))
            started = time.time()
            outcome = run_attempt(
                orch, args, instance, name=name,
                prompt=BRANCH_PROMPT.format(
                    repo=instance["repo"], problem=instance["problem"],
                    verdict=verdict[:4000], hint=branch.get("hint")),
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

    print("\n== summary ==")
    for attempt in attempts:
        print("  %-28s score=%d  %s" % (attempt.name, attempt.score,
                                        attempt.grade.summary()))
    resolved = [a for a in attempts if a.grade.resolved]
    print("\nRESOLVED by: %s" % (", ".join(a.name for a in resolved) or "nobody"))

    summary = {
        "instance": instance["instance_id"], "slot": args.slot,
        "model": args.model,
        "attempts": [{"name": a.name, "status": a.outcome.status,
                      "pairs": a.outcome.checkpoints, "score": a.score,
                      "resolved": a.grade.resolved,
                      "f2p_pass": a.grade.f2p_pass, "p2p_pass": a.grade.p2p_pass,
                      "grading_error": a.grade.error,
                      "patch_lines": a.grade.patch.count("\n"),
                      "journal": str(a.outcome.journal_path)} for a in attempts],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote %s" % (out_dir / "summary.json"))
    return 0 if resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
