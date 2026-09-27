"""Run BPO or Shepherd from one recorded mini-swe-agent baseline parent."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import httpx

from harness.orchestrator.run import Orchestrator
from harness.rollout import endpoint
from harness.slots.mini_history import load_prefix
from runstore.mini_native import reference_at
from swebench import fork_eval
from swebench.assistant_branch import actor_tools_at
from taskwise.policies import (SHEPHERD_SYSTEM, bpo_select, entropy_record,
                               shepherd_select)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def source_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    import subprocess
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()
    if not head:
        raise RuntimeError("Taskwise source has no Git identity")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                    text=True).strip()
    if dirty:
        raise RuntimeError("Taskwise source must be clean before a model run")
    return head


class ModelClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 1800.0):
        base = endpoint(base_url).rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1")
                           else "/v1/chat/completions")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, payload: dict, audit: Path) -> dict:
        record = {"request": payload, "started_at": time.time(), "attempts": []}
        write_json(audit, record)
        max_attempts = 8
        retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        for attempt in range(1, max_attempts + 1):
            record.pop("response", None)
            attempt_record = {"attempt": attempt, "started_at": time.time()}
            record["attempts"].append(attempt_record)
            try:
                response = httpx.post(
                    self.url, json=payload,
                    headers={"Authorization": "Bearer " + self.api_key},
                    timeout=self.timeout,
                )
                attempt_record["http_status"] = response.status_code
                if response.status_code != 200:
                    attempt_record["response_text"] = response.text[:4000]
                    error = RuntimeError(
                        f"model endpoint returned HTTP {response.status_code}")
                    if response.status_code not in retry_statuses:
                        raise error
                    raise httpx.TransportError(str(error))
                body = response.json()
                record["response"] = body
                attempt_record["usage"] = body.get("usage")
                if body.get("error") or len(body.get("choices") or []) != 1:
                    raise httpx.TransportError(
                        "model endpoint returned no unique completion")
                record["http_status"] = response.status_code
                record["response"] = body
                return body
            except (httpx.TimeoutException, httpx.TransportError) as error:
                attempt_record["error_type"] = type(error).__name__
                attempt_record["error"] = str(error)
                if attempt == max_attempts:
                    record["error_type"] = type(error).__name__
                    record["error"] = str(error)
                    raise
                delay = min(30, 2 ** (attempt - 1))
                attempt_record["retry_delay_seconds"] = delay
                time.sleep(delay)
            except Exception as error:
                attempt_record["error_type"] = type(error).__name__
                attempt_record["error"] = str(error)
                record["error_type"] = type(error).__name__
                record["error"] = str(error)
                raise
            finally:
                attempt_record["finished_at"] = time.time()
                attempt_record["seconds"] = (attempt_record["finished_at"]
                                             - attempt_record["started_at"])
                record["finished_at"] = time.time()
                record["seconds"] = record["finished_at"] - record["started_at"]
                write_json(audit, record)


def parent_journal(parent_output: Path) -> Path:
    root = parent_output.resolve()
    hits = sorted(root.glob("**/parent.jsonl"))
    if len(hits) == 1:
        return hits[0]
    # Terminal-Bench stores the canonical parent journal as agent/trajectory.jsonl
    # and records its exact path in summary.json.  Use that signed output pointer
    # while requiring it to stay under the parent output directory.
    try:
        summary = json.loads((root / "summary.json").read_text())
        attempts = summary.get("attempts") or []
        candidate = Path(attempts[0]["journal"]).resolve() if len(attempts) == 1 else None
        if candidate is not None and candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    raise ValueError(f"Expected one baseline parent journal under {parent_output}, got {len(hits)}")


def request_positions(journal: Path, points: dict[int, object]) -> dict[int, int]:
    """Map state-after-turn checkpoints to the next request's token position."""
    positions = {}
    cumulative = 0
    previous_turn_step = None
    for row in fork_eval.read_journal(journal):
        if row.get("type") == "raw.mini-swe-agent":
            if previous_turn_step in points and previous_turn_step not in positions:
                positions[previous_turn_step] = cumulative
            usage = (row.get("response") or {}).get("usage") or {}
            count = usage.get("completion_tokens")
            if type(count) is not int or count < 0:
                raise ValueError("BPO requires original completion-token accounting")
            cumulative += count
        elif row.get("type") == "model.turn.completed":
            step = row.get("step")
            if type(step) is int:
                previous_turn_step = step
    return positions


def score_point(args, client: ModelClient, journal: Path, point, token_position: int) -> dict:
    reference = reference_at(journal, point.step, point.session_ckpt)
    if reference is None:
        raise ValueError(f"No exact mini prefix for step {point.step}")
    prefix = load_prefix(reference)
    messages = [row["message"] for row in prefix if row["type"] == "mini.message"]
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": actor_tools_at(journal, point.step),
        "reasoning_effort": args.reasoning_effort,
        "stream": False,
        "logprobs": True,
        "top_logprobs": args.bpo_top_k,
        # High-reasoning Qwen spends internal reasoning tokens before the first
        # visible token.  A one-token cap therefore returns no content/logprobs.
        # We still score only the first visible continuation token below.
        "max_tokens": 64,
    }
    response = client.complete(payload, args.output / "entropy" / f"step-{point.step}.request.json")
    tokens = ((response["choices"][0].get("logprobs") or {}).get("content") or [])
    if not tokens:
        raise ValueError("BPO endpoint omitted requested first-token logprobs")
    result = {"step": point.step, "token_position": token_position,
              "snapshot_id": point.snapshot_id,
              "scoring_usage": response.get("usage"), "first_token": tokens[0],
              "entropy_kind": "first-content-token-top-k-plus-tail-lower-bound"}
    result.update(entropy_record(tokens[0]))
    write_json(args.output / "entropy" / f"step-{point.step}.json", result)
    return result


def select(args, problem: str, journal: Path, points: dict[int, object], client: ModelClient) -> list[dict]:
    if args.method == "bpo":
        positions = request_positions(journal, points)
        eligible = [point for step, point in sorted(points.items()) if step in positions]
        if not eligible:
            raise ValueError("No exact checkpoint maps to a following provider request")
        with ThreadPoolExecutor(max_workers=args.bpo_scoring_workers) as pool:
            scores = list(pool.map(
                lambda point: score_point(args, client, journal, point, positions[point.step]),
                eligible,
            ))
        selected = bpo_select(scores, args.max_rollouts - 1,
                              args.bpo_min_spacing, args.bpo_max_points)
        write_json(args.output / "scores.json", scores)
        return selected
    transcript, lo, hi = fork_eval.render_transcript(
        journal, token_budget=args.meta_transcript_tokens)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SHEPHERD_SYSTEM},
            {"role": "user", "content": json.dumps({
                "task": problem,
                "reward": 0,
                "eligible_checkpoint_steps": sorted(points),
                "transcript_visible_step_range": [lo, hi],
                "trajectory": transcript,
            }, ensure_ascii=False)},
        ],
        "reasoning_effort": args.reasoning_effort,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": False,
        "max_tokens": args.meta_max_tokens,
    }
    response = client.complete(payload, args.output / "meta-request.json")
    if response["choices"][0].get("finish_reason") != "stop":
        raise ValueError("Incomplete Shepherd selector response")
    proposal = fork_eval.extract_json(response["choices"][0]["message"].get("content") or "")
    write_json(args.output / "proposal.json", proposal)
    return shepherd_select(proposal, sorted(points), args.max_rollouts - 1)


class ControlledOrchestrator(Orchestrator):
    model_endpoint: str = ""
    model_key_env: str = ""
    model: str = ""
    max_output_tokens: int = 64000
    max_turns: int = 300

    def run(self, spec):
        extra = {**spec.extra,
                 "mini": {"model": {"model_kwargs": {
                     "max_tokens": self.max_output_tokens,
                     "reasoning_effort": "high"}},
                     "environment": {"timeout": 60}},
                 "rollout_contract": {
                     "message_export": True,
                     "model_endpoint": self.model_endpoint,
                     "api_key_env": self.model_key_env,
                     "model": self.model,
                     "deadline_at": time.time() + spec.timeout_s,
                     "max_turns": self.max_turns,
                     "sampling_params": {"max_new_tokens": self.max_output_tokens,
                                         "reasoning_effort": "high"}}}
        return super().run(replace(spec, extra=extra, checkpoint_enabled=False,
                                   grading_snapshot=True))


def deep_args(args, timeout: float):
    return SimpleNamespace(
        slot="mini-swe-agent", model=args.model, runtime_bin=str(args.runtime_bin),
        timeout=timeout, agent_network=None, verifier_network=None,
        setting_sources=[],
    )


def run_deep(args, choices: list[dict], journal: Path, points: dict[int, object]) -> dict:
    if args.benchmark == "deepswe":
        from deepswe.bench import DeepSWE
        bench = DeepSWE(args.tasks_dir)
        task = bench.catalogue(None)[args.instance]
        timeout = task.agent_timeout_s
    else:
        from swebench_pro.bench import SWEbenchPro
        bench = SWEbenchPro(args)
        task = bench.catalogue(None)[args.instance]
        timeout = 3600
    instance = bench.instance(task)
    agent_network = "deny" if args.benchmark == "deepswe" else "allow"
    instance.update(slot="mini-swe-agent", agent_network=agent_network)
    run_args = deep_args(args, timeout)
    orch = ControlledOrchestrator(out_dir=args.output)
    orch.model_endpoint = args.model_endpoint
    orch.model_key_env = args.model_key_env
    orch.model = args.model
    orch.max_output_tokens = args.max_output_tokens
    orch.max_turns = args.max_turns
    attempts = []
    for index, choice in enumerate(choices, 1):
        point = points[choice["step"]]
        name = f"b{index:02d}-step{point.step}"
        started = time.time()
        outcome = fork_eval.run_attempt(
            orch, run_args, instance, name=name, prompt="",
            image=point.snapshot_id, out_dir=args.output,
            resume=point.session_ckpt, fork=True,
            origin={"parent_run_id": "parent", "parent_journal": str(journal),
                    "branch_step": point.step, "snapshot_id": point.snapshot_id,
                    "conversation_cut": reference_at(journal, point.step, point.session_ckpt)["cut"],
                    "branch_policy": args.method, "selection": choice,
                    "branch_guidance": "none", "hint_delivery": "fixed-neutral",
                    "max_total_rollouts": args.max_rollouts},
            resources=bench.resources(instance), bench=bench,
            resume_without_hint=True,
        )
        grade = fork_eval.grade_attempt(outcome, instance, run_args, bench)
        if grade.error or grade.verifier_artifact_error:
            raise RuntimeError(f"branch grading infrastructure error: {grade.error or grade.verifier_artifact_error}")
        record = {"name": name, "step": point.step, "status": outcome.status,
                  "resolved": grade.resolved, "grade": asdict(grade),
                  "journal": str(outcome.journal_path),
                  "started_at": started, "validated_at": time.time()}
        record["seconds"] = record["validated_at"] - started
        attempts.append(record)
        write_json(args.output / "summary.json", {
            "task": args.instance, "method": args.method,
            "shared_parent": str(journal), "attempts": attempts,
            "resolved": any(row["resolved"] for row in attempts), "complete": False})
        if grade.resolved:
            break
    return {"task": args.instance, "method": args.method,
            "shared_parent": str(journal), "attempts": attempts,
            "resolved": any(row["resolved"] for row in attempts), "complete": True}


async def run_terminal(args, choices: list[dict], journal: Path,
                       points: dict[int, object]) -> dict:
    from harbor.models.task.task import Task
    from terminalbench import branch_eval

    task = Task(args.task_dir)
    terminal_args = SimpleNamespace(
        task_dir=args.task_dir, dataset_digest=args.dataset_digest,
        reasoning_effort=args.reasoning_effort,
        output=args.output, model=args.model, model_endpoint=args.model_endpoint,
        model_key_env=args.model_key_env, max_output_tokens=args.max_output_tokens,
        max_turns=args.max_turns, runtime_bin=args.runtime_bin,
        server_url=args.server_url, api_key_file=args.api_key_file,
        image_registry=args.image_registry, sandbox_ttl=args.sandbox_ttl,
        checkpoint_mode="disk_only",
    )
    parent_args = SimpleNamespace(parent_output=args.parent_output,
                                  dataset_digest=args.dataset_digest)
    parent, receipt = branch_eval._recorded_parent(parent_args, task)
    if parent.grade.resolved:
        raise ValueError("Policy evaluation cannot run on a resolved shared parent")
    base_image = receipt["image"]
    image_config = branch_eval._task_image_config(base_image)
    attempts = []
    for index, choice in enumerate(choices, 1):
        point = points[choice["step"]]
        name = f"b{index:02d}-step{point.step}"
        context = {"branch_guidance": "none", "checkpoint_mode": receipt["checkpoint_mode"],
                   "capture_checkpoints": False,
                   "snapshot_id": point.snapshot_id, "base_image": base_image,
                   "image_config": image_config, "parent_journal": str(journal),
                   "parent_journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                   "step": point.step, "native_session_id": point.session_ckpt,
                   "conversation_cut": reference_at(journal, point.step, point.session_ckpt)["cut"],
                   "branch_policy": args.method, "selection": choice,
                   "hint_delivery": "fixed-neutral"}
        context_path = args.output / "contexts" / f"{name}.json"
        write_json(context_path, context)
        started = time.time()
        attempt, _ = await branch_eval._trial(
            terminal_args, task, name=name, branch_context=context_path)
        if attempt.grade.error:
            raise RuntimeError(f"TerminalBench grading infrastructure error: {attempt.grade.error}")
        record = {"name": name, "step": point.step, "status": attempt.outcome.status,
                  "resolved": attempt.grade.resolved, "reward": attempt.grade.reward,
                  "journal": str(attempt.outcome.journal_path),
                  "started_at": started,
                  "validated_at": attempt.validated_at or time.time()}
        record["seconds"] = record["validated_at"] - started
        attempts.append(record)
        write_json(args.output / "summary.json", {
            "task": task.name, "method": args.method,
            "shared_parent": str(journal), "attempts": attempts,
            "resolved": any(row["resolved"] for row in attempts), "complete": False})
        if attempt.grade.resolved:
            break
    return {"task": task.name, "method": args.method,
            "shared_parent": str(journal), "attempts": attempts,
            "resolved": any(row["resolved"] for row in attempts), "complete": True}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["deepswe", "swebench-pro", "terminalbench"], required=True)
    parser.add_argument("--method", choices=["bpo", "shepherd"], required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--pro-repo", type=Path)
    parser.add_argument("--pro-data", type=Path)
    parser.add_argument("--pro-cpus", type=int, default=4)
    parser.add_argument("--pro-memory-mb", type=int, default=4096)
    parser.add_argument("--pro-verifier-timeout", type=int, default=3600)
    parser.add_argument("--dataset-digest")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--model-key-env", required=True)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--image-registry")
    parser.add_argument("--sandbox-ttl", type=int, default=36000)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-output-tokens", type=int, default=64000)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--max-rollouts", type=int, default=8)
    parser.add_argument("--bpo-top-k", type=int, default=5)
    parser.add_argument("--bpo-min-spacing", type=int, default=64)
    parser.add_argument("--bpo-max-points", type=int, default=7)
    parser.add_argument("--bpo-scoring-workers", type=int, default=2)
    parser.add_argument("--meta-max-tokens", type=int, default=64000)
    parser.add_argument("--meta-transcript-tokens", type=int, default=60000)
    args = parser.parse_args(argv)
    if args.benchmark == "deepswe" and not args.tasks_dir:
        parser.error("DeepSWE requires --tasks-dir")
    if args.benchmark == "swebench-pro" and not args.pro_repo:
        parser.error("SWE-bench Pro requires --pro-repo")
    if args.benchmark == "terminalbench" and (not args.task_dir or not args.dataset_digest):
        parser.error("TerminalBench requires --task-dir and --dataset-digest")
    if not 1 <= args.max_rollouts <= 8 or not 1 <= args.bpo_top_k <= 20:
        parser.error("Invalid rollout or BPO top-k limit")
    if not args.model_key_env.isidentifier() or not os.environ.get(args.model_key_env):
        parser.error("Model key environment variable is missing")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh policy output directory")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    journal = parent_journal(args.parent_output)
    points = fork_eval.available_branch_points(journal)
    if not points:
        raise RuntimeError("Shared parent exposes no exact mini checkpoint/session pairs")
    if args.benchmark == "deepswe":
        from deepswe.bench import DeepSWE
        problem = DeepSWE(args.tasks_dir).catalogue(None)[args.instance].instruction
    elif args.benchmark == "swebench-pro":
        from swebench_pro.bench import SWEbenchPro
        problem = SWEbenchPro(args).catalogue(None)[args.instance].problem
    else:
        from harbor.models.task.task import Task
        problem = Task(args.task_dir).instruction
    client = ModelClient(args.model_endpoint, os.environ[args.model_key_env])
    choices = select(args, problem, journal, points, client)
    write_json(args.output / "plan.json", {
        "source_commit": source_identity(), "task": args.instance,
        "method": args.method, "shared_parent": str(journal),
        "max_total_rollouts": args.max_rollouts,
        "budget_includes_shared_initial": True,
        "eligible_steps": sorted(points), "selected": choices,
        "selection_only": True, "training": False,
        "branch_guidance": "none", "hint_delivery": "fixed-neutral"})
    state = (run_deep(args, choices, journal, points) if args.benchmark != "terminalbench"
             else asyncio.run(run_terminal(args, choices, journal, points)))
    write_json(args.output / "summary.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
