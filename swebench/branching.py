"""Branching an unfinished marathon attempt from its own checkpoints.

A failed attempt at a 10-hour task is not worthless: it carries hours of
built environment and a transcript of what was tried. Per-step snapshots
(``agent/checkpoints.py``) make any step resumable; what they cannot say is
*which* step is worth resuming from. This module asks a model.

The analysis reads the trajectory (either harness's shape), the step→snapshot
map, and the verifier's verdict, and returns a :class:`BranchPlan`:

- **one branch step**, chosen as late as possible — every step kept is work
  preserved — but strictly before the decisive wrong turn, because resuming
  after the mistake inherits it. Steps whose capture had live background
  processes are flagged to the model: a disk-only resume loses them.
- **several diverse directions**, each a self-contained hint for the resumed
  agent (which has the environment but not the conversation): different
  hypotheses about what went wrong, or different strategies forward — not
  paraphrases of one idea. Running them in parallel from one snapshot is how
  a single failed rollout fans out into several informed ones.

Usage::

    python -m swebench.branching \
        --trajectory /tmp/marathon-cc-sonnet/trajectories/rust-java-lsp.json \
        --task-dir /tmp/swe-marathon/tasks/rust-java-lsp \
        -c swebench/configs/marathon-cc-sonnet.yaml \
        --branches 3 --launch

Analysis and rollout are separate concerns: the analyzer model is this
module's own (``--model``, default Sonnet 5 on Bedrock); the rollout model is
whatever the run config says.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "bedrock/us.anthropic.claude-sonnet-5"

#: How much of the rendered transcript the analyst sees. Tail-biased when
#: over budget: the decisive mistake is usually late, and so are the branch
#: points worth considering.
CHAR_BUDGET = 120_000


class BranchingError(RuntimeError):
    """The trajectory cannot be analyzed or the plan cannot be built."""


# --------------------------------------------------------------------------
# Rendering a trajectory for the analyst
# --------------------------------------------------------------------------

def _one_line(value, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]} … {text[-half:]}"


def _result_text(result_event: dict) -> str:
    content = result_event.get("content")
    if isinstance(content, list):
        content = " ".join(str(block.get("text", "")) for block in content
                           if isinstance(block, dict))
    text = _one_line(content or "", 240)
    return f"[ERROR] {text}" if result_event.get("is_error") else text


def render_trajectory(data: dict, char_budget: int = CHAR_BUDGET) -> str:
    """One line per tool step, agent prose indented — both harness shapes.

    The claude-code harness records an event stream (``trajectory``); the
    AshAgent harness records chat ``messages``. Either way the analyst gets
    ``[step] tool(args) -> result`` lines it can name a branch step from.
    """
    lines: list[str] = []
    events = data.get("trajectory")
    if events:
        results = {e.get("tool_use_id"): e for e in events
                   if e.get("type") == "tool_result"}
        for event in events:
            kind = event.get("type")
            if kind == "tool_use":
                outcome = results.get(event.get("id"))
                lines.append(
                    f"[{event.get('step')}] {event.get('name')}"
                    f"({_one_line(event.get('input'), 200)})"
                    + (f" -> {_result_text(outcome)}" if outcome else ""))
            elif kind == "text" and str(event.get("text", "")).strip():
                lines.append(f"    agent: {_one_line(event['text'], 500)}")
    else:
        for message in data.get("messages", []):
            if isinstance(message, str):
                lines.append(f"    agent: {_one_line(message, 500)}")
                continue
            role = message.get("role", "?")
            calls = message.get("tool_calls") or []
            names = ",".join(
                str((c.get("function") or {}).get("name", "?")) for c in calls
                if isinstance(c, dict))
            suffix = f" [calls: {names}]" if names else ""
            lines.append(
                f"{role}: {_one_line(message.get('content') or '', 300)}{suffix}")

    if not lines:
        raise BranchingError("trajectory has no events or messages to analyze")

    rendered = "\n".join(lines)
    if len(rendered) <= char_budget:
        return rendered

    # Keep the opening (what the agent set out to do) and as much of the tail
    # as fits; elide the middle and SAY SO, so the analyst never mistakes a
    # cut transcript for a short attempt.
    head = lines[:max(1, len(lines) // 20)]
    spent = sum(len(line) + 1 for line in head)
    tail: list[str] = []
    for line in reversed(lines[len(head):]):
        if spent + len(line) + 1 > char_budget:
            break
        tail.append(line)
        spent += len(line) + 1
    elided = len(lines) - len(head) - len(tail)
    return "\n".join(head) + f"\n… [{elided} lines elided] …\n" + \
        "\n".join(reversed(tail))


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------

@dataclass
class BranchDirection:
    name: str
    hint: str


@dataclass
class BranchPlan:
    step: int
    snapshot_id: str
    why_here: str
    what_went_wrong: str
    directions: list[BranchDirection] = field(default_factory=list)
    #: The chosen step had live background processes at capture; the resumed
    #: environment will not have them.
    live_background: bool = False

    def as_dict(self) -> dict:
        return {"step": self.step, "snapshot_id": self.snapshot_id,
                "why_here": self.why_here,
                "what_went_wrong": self.what_went_wrong,
                "live_background": self.live_background,
                "directions": [vars(d) for d in self.directions]}


def branchable_steps(data: dict) -> tuple[dict[int, str], set[int]]:
    """The step→snapshot map, and which steps are risky to resume.

    Every step in the map is resumable (clean steps map to the previous
    capture); risky ones had live background processes, which a disk-only
    snapshot does not carry.
    """
    checkpoints = (data.get("info") or {}).get("checkpoints") or {}
    snapshots = {int(step): snap for step, snap
                 in (checkpoints.get("step_snapshots") or {}).items()}
    risky = {int(r["turn"]) for r in checkpoints.get("records", [])
             if r.get("live_background")}
    return snapshots, risky


_ANALYSIS_PROMPT = """\
You are deciding where to BRANCH a fresh attempt of a long-horizon software \
task from per-step environment snapshots of an earlier, unfinished attempt.

## The task (instruction, possibly excerpted)
{instruction}

## How the attempt ended
exit_status: {exit_status}
reward: {reward}   partial_score: {partial}
metrics: {metrics}
grading_error: {grading_error}

## The attempt, one line per tool step ("[N] tool(args) -> result")
{transcript}

## Rules
- Valid branch steps: integers {lo}..{hi} that appear in the transcript. A
  branch at step N resumes from the environment AS IT STOOD AFTER step N.
- LATER IS BETTER: every step kept preserves hours of work. But the step must
  be strictly BEFORE the decisive wrong turn — branching after it inherits it.
  Weigh both; do not reflexively pick the middle.
- These steps had live background processes which will NOT survive the resume
  (the filesystem does; processes do not): {risky}. Prefer other steps, or
  make the hint say what to restart.
- The resumed agent gets the ENVIRONMENT ONLY — files, builds, installed
  packages. It has none of this conversation. Every hint must therefore be
  self-contained: say what already exists on disk and is trustworthy, what
  the earlier attempt learned (including from its failures), and what to do
  differently this time.
- Produce {branches} genuinely DIVERSE directions: different hypotheses about
  what went wrong, or different strategies forward (fix-forward, partial
  rewrite of the flawed component, different algorithm/dependency, different
  verification discipline, ...). Not paraphrases of one idea. It is fine for
  directions to disagree with each other — they run in parallel.
{notes_section}

Return ONLY a JSON object, no prose around it:
{{"branch_step": <int>, "why_here": "<why this step, and why not later>",
  "what_went_wrong": "<your diagnosis of the attempt>",
  "branches": [{{"name": "<slug>", "hint": "<4-10 sentences for the resumed agent>"}}]}}
"""


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise BranchingError(f"analyst returned no JSON: {text[:200]!r}")
        candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Long hint strings tend to arrive with literal newlines/tabs inside
        # them — invalid JSON, harmless intent. strict=False accepts control
        # characters in strings instead of failing the whole round.
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            raise BranchingError(
                f"analyst JSON does not parse: {exc}") from exc


def analyze(trajectory_path: Path | str, instruction: str,
            model: str = DEFAULT_MODEL, branches: int = 3,
            char_budget: int = CHAR_BUDGET,
            notes: str = "",
            completion=None) -> BranchPlan:
    """Ask ``model`` where to branch and in which directions.

    ``completion`` is litellm's, injectable for tests. The returned plan's
    step is validated against the trajectory's own map; an analyst answer
    naming a step without a snapshot is clamped to the nearest EARLIER
    mapped step — later would keep work the snapshot does not hold.

    ``notes`` is how branching becomes ITERATIVE rather than one blind
    fan-out: when a task's failures are invisible to the agent (a hidden
    holdout set), each finished branch is an information probe — its change
    plus its verifier delta is a bit the next round can condition on. Pass a
    summary of earlier branches' (direction, change, score) here and the
    analyst plans round N+1 in the narrowed hypothesis space instead of
    re-guessing from zero.
    """
    data = json.loads(Path(trajectory_path).read_text())
    snapshots, risky = branchable_steps(data)
    if not snapshots:
        raise BranchingError(
            f"{trajectory_path} has no step->snapshot map; was the run "
            "started with checkpoints enabled?")

    info = data.get("info") or {}
    marathon = info.get("marathon") or {}
    notes_section = ""
    if notes.strip():
        notes_section = (
            "\n## Results of earlier branch attempts (condition on these — "
            "each is a probe whose verifier delta is information the agent "
            "itself never saw)\n" + notes.strip() + "\n"
            "\nDo NOT re-propose a direction an earlier branch already tried "
            "unless its result suggests it was right but incomplete.\n")
    prompt = _ANALYSIS_PROMPT.format(
        instruction=_one_line(instruction, 6000),
        exit_status=info.get("exit_status") or data.get("exit_status"),
        reward=marathon.get("reward"), partial=marathon.get("partial_score"),
        metrics=_one_line(marathon.get("metrics") or {}, 1500),
        grading_error=marathon.get("grading_error"),
        transcript=render_trajectory(data, char_budget),
        lo=min(snapshots), hi=max(snapshots),
        risky=sorted(risky) or "none",
        branches=branches,
        notes_section=notes_section,
    )

    if completion is None:
        import litellm
        completion = litellm.completion
    # An empty completion is not the model declining (AGENTS.md: extended
    # thinking can consume the whole output budget and return an empty text
    # block). Retry with a growing budget rather than failing on the first.
    text, last_finish = "", None
    max_tokens = 8000
    for _ in range(3):
        response = completion(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              max_tokens=max_tokens)
        text = response.choices[0].message.content or ""
        last_finish = getattr(response.choices[0], "finish_reason", None)
        if text.strip():
            break
        max_tokens = min(max_tokens * 2, 32000)
    if not text.strip():
        raise BranchingError(
            f"analyst returned empty content three times "
            f"(last finish_reason={last_finish!r}); the output budget may be "
            "going entirely to thinking")
    verdict = _extract_json(text)

    asked = int(verdict.get("branch_step", -1))
    valid = [step for step in sorted(snapshots) if step <= asked]
    if not valid:
        raise BranchingError(
            f"analyst chose step {asked}, before the first snapshot "
            f"({min(snapshots)})")
    step = valid[-1]

    directions = [BranchDirection(name=str(b.get("name", f"branch-{i}")),
                                  hint=str(b.get("hint", "")).strip())
                  for i, b in enumerate(verdict.get("branches") or [], 1)
                  if str(b.get("hint", "")).strip()]
    if not directions:
        raise BranchingError("analyst produced no usable branch directions")

    return BranchPlan(step=step, snapshot_id=snapshots[step],
                      why_here=str(verdict.get("why_here", "")),
                      what_went_wrong=str(verdict.get("what_went_wrong", "")),
                      directions=directions,
                      live_background=step in risky)


# --------------------------------------------------------------------------
# Launching the branches
# --------------------------------------------------------------------------

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "branch"


def branch_commands(plan: BranchPlan, *, task_dir: str, config: str,
                    output_base: str, harness: str = "marathon-claude-code",
                    python: str = sys.executable) -> list[tuple[Path, list[str]]]:
    """One (output_dir, argv) per direction. Separate from launching so a
    dry run prints exactly what a real one executes."""
    commands = []
    for i, direction in enumerate(plan.directions, 1):
        output_dir = Path(output_base) / f"branch-{plan.step}-{i}-{_slug(direction.name)}"
        commands.append((output_dir, [
            python, "-m", "swebench", "--harness", harness, "-c", config,
            "--task-dir", task_dir,
            "--resume-from", plan.snapshot_id,
            "--resume-hint", direction.hint,
            "-o", str(output_dir),
        ]))
    return commands


def launch(plan: BranchPlan, *, task_dir: str, config: str, output_base: str,
           harness: str = "marathon-claude-code",
           python: str = sys.executable) -> list[subprocess.Popen]:
    """Start every branch in parallel, each logging beside its output dir.

    Parallel from one snapshot is safe by construction: each run spawns its
    own sandbox from the content-addressed snapshot, and snapshot names carry
    each run's own id.
    """
    processes = []
    for output_dir, argv in branch_commands(
            plan, task_dir=task_dir, config=config, output_base=output_base,
            harness=harness, python=python):
        output_dir.mkdir(parents=True, exist_ok=True)
        log = open(output_dir.with_suffix(".log"), "w")
        processes.append(subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent)))
    return processes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Choose a branch point in an unfinished marathon attempt "
                    "and fan out resumed rollouts from it.")
    parser.add_argument("--trajectory", required=True,
                        help="Trajectory JSON of the unfinished attempt "
                             "(final, or a killed run's in-progress file)")
    parser.add_argument("--task-dir", required=True,
                        help="The SWE-Marathon task directory")
    parser.add_argument("--config", "-c", required=True,
                        help="Run config for the branched rollouts")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Analyst model (default: {DEFAULT_MODEL})")
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--notes", default=None,
                        help="Path to a text file summarizing earlier branch "
                             "rounds' (direction -> verifier delta); makes "
                             "the analysis iterative instead of a blind "
                             "fan-out")
    parser.add_argument("--output-base", default=None,
                        help="Parent directory for branch outputs "
                             "(default: <trajectory's run dir>/branches)")
    parser.add_argument("--harness", default="marathon-claude-code")
    parser.add_argument("--launch", action="store_true",
                        help="Actually start the branch runs (default: "
                             "print the plan and the commands)")
    parser.add_argument("--plan-out", default=None,
                        help="Also write the plan JSON here")
    parser.add_argument("--plan-in", default=None,
                        help="Skip analysis; launch exactly this plan JSON "
                             "(a prior --plan-out). Analysis is stochastic, "
                             "so review-then-launch needs the reviewed plan, "
                             "not a fresh one.")
    args = parser.parse_args(argv)

    from .marathon import load_task
    task = load_task(args.task_dir)
    output_base = args.output_base or str(
        Path(args.trajectory).resolve().parent.parent / "branches")

    if args.plan_in:
        p = json.loads(Path(args.plan_in).read_text())
        plan = BranchPlan(step=p["step"], snapshot_id=p["snapshot_id"],
                          why_here=p.get("why_here", ""),
                          what_went_wrong=p.get("what_went_wrong", ""),
                          directions=[BranchDirection(**d)
                                      for d in p["directions"]],
                          live_background=p.get("live_background", False))
    else:
        plan = analyze(args.trajectory, task.instruction,
                       model=args.model, branches=args.branches,
                       notes=Path(args.notes).read_text() if args.notes else "")

    print(f"branch step   : {plan.step}"
          + ("  (had live background processes!)" if plan.live_background else ""))
    print(f"snapshot      : {plan.snapshot_id}")
    print(f"diagnosis     : {plan.what_went_wrong}")
    print(f"why this step : {plan.why_here}")
    for i, direction in enumerate(plan.directions, 1):
        print(f"  [{i}] {direction.name}: {direction.hint}")

    if args.plan_out:
        Path(args.plan_out).write_text(json.dumps(plan.as_dict(), indent=2))

    commands = branch_commands(plan, task_dir=args.task_dir,
                               config=args.config, output_base=output_base,
                               harness=args.harness)
    if not args.launch:
        print("\ndry run — commands that --launch would start in parallel:")
        for _, argv_ in commands:
            print("  " + shlex.join(argv_))
        return 0

    processes = launch(plan, task_dir=args.task_dir, config=args.config,
                       output_base=output_base, harness=args.harness)
    for (output_dir, _), proc in zip(commands, processes):
        print(f"launched pid {proc.pid}: {output_dir} "
              f"(log: {output_dir.with_suffix('.log')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
