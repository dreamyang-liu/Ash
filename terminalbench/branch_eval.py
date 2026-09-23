"""One official Harbor task with mini assistant-turn branches from full snapshots."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from harness.execution.templates import find_regctl
from swebench import fork_eval, structured_review
from swebench.assistant_branch import ASSISTANT_REVIEW_PROMPT, reviewer_context
from swebench.branching import branch_count_rule
from terminalbench.review_transport import ReviewTransport


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@dataclass
class HarborGrade(fork_eval.Grade):
    reward: float | None = None

    def summary(self) -> str:
        return (f"official Harbor grading error: {self.error}" if self.error
                else f"official Harbor reward={self.reward}; resolved={self.resolved}")


def _read_grade(path: Path) -> HarborGrade:
    if not path.is_file():
        raise RuntimeError(f"Official Harbor trial has no result: {path}")
    body = json.loads(path.read_text())
    if not body.get("finished_at") or body.get("exception_info"):
        raise RuntimeError(f"Official Harbor trial failed: {body.get('exception_info')}")
    reward = ((body.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if type(reward) not in (int, float) or not 0 <= reward <= 1:
        raise RuntimeError(f"Official Harbor reward is missing or invalid: {reward!r}")
    return HarborGrade(resolved=reward == 1, f2p_pass=reward == 1,
                       p2p_pass=True, p2p_ran=True, reward=float(reward),
                       detail=json.dumps(body.get("verifier_result"), ensure_ascii=False)[:20000],
                       verifier_artifacts=str(path.parent))


def _source_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("Three-benchmark source worktree must be clean before a real trial")
    return head


def _task_image_config(image: str) -> dict:
    if "@sha256:" not in image:
        raise ValueError("Official task image is not digest-pinned")
    regctl = find_regctl()
    if regctl is None:
        raise RuntimeError("regctl is required for snapshot restore")
    output = subprocess.check_output(
        [str(regctl), "image", "config", image, "--format", "{{json .Config}}"],
        text=True, timeout=120)
    config = json.loads(output)
    if not isinstance(config, dict):
        raise ValueError("Pinned task image has no OCI config")
    return config


def _verify_prepared_task(args, task: Task, task_hash: str) -> None:
    preparation = json.loads(args.preparation_manifest.read_text())
    if preparation.get("dataset_digest") != args.dataset_digest:
        raise ValueError("Prepared dataset digest differs from the selected cohort")
    rows = [row for row in preparation.get("tasks", [])
            if Path(row.get("path", "")).resolve() == args.task_dir.resolve()
            and row.get("name") == task.name]
    if len(rows) != 1 or rows[0].get("task_toml_sha256") != task_hash:
        raise ValueError("Task is absent from the pinned preparation or its definition changed")


def _trial_config(args, task: Task, *, name: str, branch_context: Path | None) -> TrialConfig:
    agent_kwargs = {"inference_endpoint": args.model_endpoint,
                    "api_key_env": args.model_key_env,
                    "actor_timeout_s": task.config.agent.timeout_sec,
                    "max_output_tokens": args.max_output_tokens,
                    "max_turns": args.max_turns}
    env_kwargs = {"runtime_bin": str(args.runtime_bin.resolve()),
                  "checkpoint_mode": "full", "sandbox_ttl": args.sandbox_ttl,
                  "server_url": args.server_url,
                  "api_key_file": str(args.api_key_file.resolve()) if args.api_key_file else None,
                  "image_registry": args.image_registry}
    if branch_context is not None:
        agent_kwargs["branch_context"] = str(branch_context)
        env_kwargs["branch_context"] = str(branch_context)
    prefix = task.name.split("/")[-1][:36]
    return TrialConfig.model_validate({
        "task": {"path": str(args.task_dir.resolve()),
                 "source": "terminal-bench/terminal-bench-2-1@" + args.dataset_digest},
        "trial_name": f"{prefix}-{name}", "trials_dir": str((args.output / "trials").resolve()),
        "agent": {"import_path": ("terminalbench.branching:BranchMini" if branch_context
                                  else "terminalbench.agentenv_mini:AgentENVMini"),
                  "model_name": args.model, "kwargs": agent_kwargs},
        "environment": {"import_path": ("terminalbench.branching:SnapshotEnvironment"
                                        if branch_context else "terminalbench.agentenv:AgentENVEnvironment"),
                        "delete": True, "kwargs": env_kwargs},
    })


async def _trial(args, task: Task, *, name: str, branch_context: Path | None) -> tuple[fork_eval.Attempt, dict]:
    config = _trial_config(args, task, name=name, branch_context=branch_context)
    directory = config.trials_dir / config.trial_name
    result_path = directory / "result.json"
    if not result_path.exists():
        if directory.exists():
            raise RuntimeError(f"Interrupted Harbor trial requires review before retry: {directory}")
        trial = await Trial.create(config)
        await trial.run()
    grade = _read_grade(result_path)
    journal = directory / "agent" / "trajectory.jsonl"
    if not journal.is_file():
        raise RuntimeError(f"Graded attempt has no Ash journal: {journal}")
    outcome = fork_eval.outcome_from_journal(journal, run_id=name)
    if outcome.status != "completed":
        raise RuntimeError(f"Graded attempt did not complete its actor: {outcome.status}")
    snapshot = json.loads((directory / "agent" / "snapshot.json").read_text())
    if not snapshot.get("final_snapshot_id") or not snapshot.get("sandbox_id"):
        raise RuntimeError("Attempt is missing its final snapshot or sandbox identity")
    receipts = [json.loads(p.read_text()) for p in directory.glob("*.agentenv.json")]
    owned = [row for row in receipts if row.get("sandbox_id") == snapshot["sandbox_id"]]
    if len(owned) != 1 or owned[0].get("checkpoint_mode") != "full":
        raise RuntimeError("Cannot identify the full-snapshot agent environment receipt")
    if branch_context is not None:
        restored = json.loads((directory / "restored-state.json").read_text())
        if restored.get("snapshot_id") != json.loads(branch_context.read_text())["snapshot_id"]:
            raise RuntimeError("Official trial restored a different snapshot")
    return fork_eval.Attempt(name, outcome, grade), owned[0]


def _analysis(transport: ReviewTransport, model: str, task: Task,
              attempt: fork_eval.Attempt, points: dict, width: int,
              token_budget: int) -> dict:
    transcript, lo, hi = fork_eval.render_transcript(attempt.outcome.journal_path,
                                                     token_budget=token_budget)
    if hi < 1:
        return {"failure_reason": "no tool steps recorded", "steps": 0,
                "lesson": "", "salvage": "nothing", "branch_candidates": []}
    report = fork_eval.extract_json(transport(model, fork_eval._CASE_PROMPT.format(
        problem=task.instruction[:20000], verdict=attempt.verdict_text(),
        transcript=transcript,
        checkpoint_steps=json.dumps(sorted(s for s in points if s <= hi)),
        candidate_limit=width, lo=lo, hi=hi)))
    report["steps"] = hi
    return report


def _review(transport: ReviewTransport, model: str, task: Task, attempts: list,
            cases: dict, round_no: int, width: int, output: Path,
            reviewer_max_attempts: int):
    points = {a.name: fork_eval.available_branch_points(a.outcome.journal_path) for a in attempts}
    points = {name: {step: pair for step, pair in pairs.items()
                     if step <= cases.get(name, {}).get("steps", 0)}
              for name, pairs in points.items()}
    plan_path = output / f"plan-round{round_no}.json"
    record = {"available_steps": {name: sorted(pairs) for name, pairs in points.items()},
              "branch_limit": width, "branch_guidance": "assistant-turn",
              "branch_count_mode": "adaptive"}
    if not any(points.values()):
        record["validation_error"] = "no exact snapshot/native-prefix pairs"
        _write(plan_path, record)
        return []
    reports = []
    for attempt in attempts:
        reports.append({**cases[attempt.name], "name": attempt.name,
                        "round": attempt.round_no, "grade": attempt.grade.summary(),
                        "available_steps": sorted(points[attempt.name]),
                        "assistant_turn_given": attempt.assistant_turn,
                        "native_history": reviewer_context(attempt.outcome.journal_path,
                                                            points[attempt.name])})
    prompt = ASSISTANT_REVIEW_PROMPT.format(
        problem=task.instruction[:20000],
        reports=json.dumps(reports, indent=1, ensure_ascii=False),
        count_rule=branch_count_rule("adaptive", width))
    result = fork_eval.review_with_feedback(
        lambda text: transport(model, text), prompt,
        structured_review.extract_branch_plan,
        lambda plan: fork_eval.prepare_branches(
            plan, limit=width, round_no=round_no,
            attempts={a.name: a for a in attempts}, checkpoints=points,
            count_mode="adaptive", guidance_mode="assistant-turn"),
        max_attempts=reviewer_max_attempts, record=record,
        persist=lambda value: _write(plan_path, value),
    )
    if result is None:
        return []
    plan, choices = result
    record["selected_branches"] = [{"name": choice.run_name, "base": choice.base.name,
                                    "step": choice.checkpoint.step,
                                    "snapshot_id": choice.checkpoint.snapshot_id,
                                    "conversation_cut": choice.cut}
                                   for choice in choices]
    _write(plan_path, record)
    return choices


async def evaluate(args) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory for each one-task evaluation")
    if not args.runtime_bin.is_file():
        raise FileNotFoundError(args.runtime_bin)
    task = Task(args.task_dir)
    task_hash = hashlib.sha256((args.task_dir / "task.toml").read_bytes()).hexdigest()
    _verify_prepared_task(args, task, task_hash)
    commit = _source_commit()
    args.output.mkdir(parents=True)
    state = {"source_commit": commit, "task": task.name, "task_toml_sha256": task_hash,
             "dataset_digest": args.dataset_digest, "model": args.model,
             "branch_guidance": "assistant-turn", "branch_caps": args.branches,
             "rounds": args.rounds, "attempts": [], "complete": False}
    path = args.output / "summary.json"
    _write(path, state)
    transport = ReviewTransport(args.output / "reviews", model=args.model,
                                model_endpoint=args.model_endpoint,
                                api_key_env=args.model_key_env,
                                max_output_tokens=args.review_output_tokens,
                                reasoning_effort=args.reasoning_effort)
    attempts = []
    root, receipt = await _trial(args, task, name="parent", branch_context=None)
    attempts.append(root)
    base_image = receipt["image"]
    config = None
    cases = {}

    def save():
        state["attempts"] = [{"name": a.name, "round": a.round_no,
                              "resolved": a.grade.resolved,
                              "reward": a.grade.reward,
                              "journal": str(a.outcome.journal_path)} for a in attempts]
        state["resolved"] = any(a.grade.resolved for a in attempts)
        _write(path, state)

    save()
    for round_no in range(1, args.rounds + 1):
        if any(a.grade.resolved for a in attempts):
            break
        width = args.branches[min(round_no - 1, len(args.branches) - 1)]
        for attempt in attempts:
            if attempt.name not in cases:
                points = fork_eval.available_branch_points(attempt.outcome.journal_path)
                cases[attempt.name] = _analysis(
                    transport, args.model, task, attempt, points, width, args.analyst_tokens)
        choices = _review(transport, args.model, task, attempts, cases, round_no,
                          width, args.output, args.reviewer_max_attempts)
        if not choices:
            plan_path = args.output / f"plan-round{round_no}.json"
            plan = json.loads(plan_path.read_text())
            state["stop_reason"] = plan.get("validation_error") or "reviewer selected no branches"
            state["branch_schedule_complete"] = False
            break
        if config is None:
            config = _task_image_config(base_image)
        for choice in choices:
            parent = choice.base.outcome.journal_path
            context = {"branch_guidance": "assistant-turn", "checkpoint_mode": "full",
                       "snapshot_id": choice.checkpoint.snapshot_id,
                       "base_image": base_image, "image_config": config,
                       "parent_journal": str(parent),
                       "parent_journal_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
                       "step": choice.checkpoint.step,
                       "native_session_id": choice.checkpoint.session_ckpt,
                       "conversation_cut": choice.cut,
                       "assistant_turn": choice.assistant_turn}
            context_path = args.output / "contexts" / f"{choice.run_name}.json"
            _write(context_path, context)
            attempt, _ = await _trial(args, task, name=choice.run_name,
                                      branch_context=context_path)
            attempt.round_no = round_no
            attempt.assistant_turn = choice.assistant_turn
            attempts.append(attempt)
            save()
    if hashlib.sha256((args.task_dir / "task.toml").read_bytes()).hexdigest() != task_hash:
        raise RuntimeError("Official task definition changed during evaluation")
    state["complete"] = True
    state.setdefault("branch_schedule_complete", True)
    save()
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--dataset-digest", required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--model-key-env", required=True)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--server-url")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--image-registry")
    parser.add_argument("--sandbox-ttl", type=int, default=36000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--branches", type=int, nargs="+", default=[4, 3])
    parser.add_argument("--max-output-tokens", type=int, default=64000)
    parser.add_argument("--review-output-tokens", type=int, default=64000)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--reviewer-max-attempts", type=int, default=3)
    parser.add_argument("--analyst-tokens", type=int, default=100000)
    parser.add_argument("--reasoning-effort")
    args = parser.parse_args(argv)
    if (args.rounds < 0 or not args.branches or any(n < 1 for n in args.branches)
            or args.sandbox_ttl < 30000 or args.reviewer_max_attempts < 1):
        parser.error("Invalid branch schedule, snapshot TTL or reviewer attempts")
    return args


def main(argv=None) -> int:
    state = asyncio.run(evaluate(parse_args(argv)))
    print(json.dumps({key: value for key, value in state.items() if key != "attempts"}, indent=2))
    return 0 if state["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
