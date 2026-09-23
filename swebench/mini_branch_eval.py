"""One pinned mini assistant-turn 4→3 evaluation on Pro or DeepSWE."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from harness.execution.pipeline import ToolPipeline
from harness.orchestrator.run import Orchestrator
from harness.rollout import endpoint
from swebench import fork_eval, structured_review
from swebench.review_transport import ReviewTransport
from swebench_pro.limits import OfficialToolBudget


class ProMiniBudget(OfficialToolBudget):
    def __init__(self):
        super().__init__(command_timeout=450, total_seconds=1800,
                         consecutive_timeouts=3)
        self.control = None

    def before(self, context):
        reason = self.exhausted
        if reason:
            if self.control is not None:
                self.control.request_stop(reason)
            raise RuntimeError(reason)
        return super().before(context)

    def after(self, context, result):
        answer = super().after(context, result)
        reason = self.exhausted
        if reason and self.control is not None:
            self.control.request_stop(reason)
        return answer


def _source_identity() -> tuple[str, dict]:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("Shared evaluation source must be clean before a model run")
    hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
              for path in ("swebench/mini_branch_eval.py", "swebench/fork_eval.py",
                           "swebench/structured_review.py", "swebench_pro/bench.py",
                           "deepswe/bench.py")}
    return head, hashes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["swebench-pro", "deepswe"], required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True, help="exact clean branch commit")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--model-key-env", required=True)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--pro-repo", type=Path)
    parser.add_argument("--pro-data", type=Path)
    parser.add_argument("--pro-runtime-port", type=int)
    parser.add_argument("--pro-collector-runtime-port", type=int)
    parser.add_argument("--pro-cpus", type=int, default=4)
    parser.add_argument("--pro-memory-mb", type=int, default=16384)
    parser.add_argument("--pro-verifier-timeout", type=int, default=3600)
    parser.add_argument("--agent-network", choices=["allow", "deny"])
    parser.add_argument("--verifier-network", choices=["allow", "deny"])
    parser.add_argument("--max-output-tokens", type=int, default=64000)
    parser.add_argument("--review-output-tokens", type=int, default=64000)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--reviewer-max-attempts", type=int, default=3)
    parser.add_argument("--reasoning-effort")
    args = parser.parse_args(argv)
    if args.benchmark == "deepswe" and not args.tasks_dir:
        parser.error("DeepSWE requires --tasks-dir")
    if args.benchmark == "swebench-pro" and not args.pro_repo:
        parser.error("SWE-bench Pro requires --pro-repo")
    if (args.max_output_tokens < 1 or args.review_output_tokens < 1
            or args.max_turns < 1 or args.reviewer_max_attempts < 1):
        parser.error("Model output, turns and reviewer attempts must be positive")
    endpoint(args.model_endpoint)
    if not args.model_key_env.isidentifier():
        parser.error("--model-key-env must name one environment variable")
    return args


def _fork_args(args) -> list[str]:
    timeout = 3600 if args.benchmark == "swebench-pro" else 10800
    values = ["--benchmark", args.benchmark, "--instance", args.instance,
              "--slot", "mini-swe-agent", "--model", args.model,
              "--analyst-model", args.model,
              "--branch-guidance", "assistant-turn", "--branches", "4,3",
              "--branch-count-mode", "adaptive", "--rounds", "2",
              "--reviewer-max-attempts", str(args.reviewer_max_attempts),
              "--timeout", str(timeout), "--runtime-bin", str(args.runtime_bin.resolve()),
              "-o", str(args.out.resolve())]
    if args.benchmark == "deepswe":
        values += ["--tasks-dir", str(args.tasks_dir.resolve())]
    else:
        values += ["--pro-repo", str(args.pro_repo.resolve()),
                   "--pro-cpus", str(args.pro_cpus),
                   "--pro-memory-mb", str(args.pro_memory_mb),
                   "--pro-verifier-timeout", str(args.pro_verifier_timeout)]
        if args.pro_data:
            values += ["--pro-data", str(args.pro_data.resolve())]
        if args.pro_runtime_port:
            values += ["--pro-runtime-port", str(args.pro_runtime_port)]
        if args.pro_collector_runtime_port:
            values += ["--pro-collector-runtime-port", str(args.pro_collector_runtime_port)]
    for phase in ("agent", "verifier"):
        policy = getattr(args, phase + "_network")
        if policy:
            values += ["--" + phase + "-network", policy]
    return values


def main(argv=None) -> int:
    args = parse_args(argv)
    head, hashes = _source_identity()
    if head != args.source_commit:
        raise ValueError(f"Requested source {args.source_commit} differs from clean HEAD {head}")
    if not args.runtime_bin.is_file():
        raise FileNotFoundError(args.runtime_bin)
    if not os.environ.get(args.model_key_env):
        raise ValueError(f"Model key variable {args.model_key_env} is unset")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Use a fresh evaluation directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"source_commit": head, "source_hashes": hashes,
                "benchmark": args.benchmark, "instance": args.instance,
                "slot": "mini-swe-agent", "model": args.model,
                "branch_guidance": "assistant-turn", "branch_caps": [4, 3],
                "branch_count_mode": "adaptive", "rounds": 2,
                "actor_timeout_s": 3600 if args.benchmark == "swebench-pro" else 10800,
                "reviewer_max_attempts": args.reviewer_max_attempts,
                "max_output_tokens": args.max_output_tokens,
                "max_turns": args.max_turns}
    (args.out / "shared-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    transport = ReviewTransport(
        args.out / "reviews", model=args.model,
        model_endpoint=args.model_endpoint, api_key_env=args.model_key_env,
        max_output_tokens=args.review_output_tokens,
        reasoning_effort=args.reasoning_effort)

    class ControlledOrchestrator(Orchestrator):
        budget = None

        def run(self, spec):
            extra = {**spec.extra,
                     "mini": {"model": {"model_kwargs": {
                         "max_tokens": args.max_output_tokens}},
                         "environment": {"timeout": 450 if args.benchmark == "swebench-pro" else 60}},
                     "rollout_contract": {
                         "message_export": True,
                         "model_endpoint": args.model_endpoint,
                         "api_key_env": args.model_key_env,
                         "model": args.model,
                         "deadline_at": time.time() + spec.timeout_s,
                         "max_turns": args.max_turns,
                         "sampling_params": {"max_new_tokens": args.max_output_tokens}}}
            return super().run(replace(spec, extra=extra))

        def _wire_sandbox(self, spec, claim):
            owned, wiring = super()._wire_sandbox(spec, claim)
            if args.benchmark == "swebench-pro":
                self.budget = ProMiniBudget()
                owned.tracker = self.budget
                owned.server.pipeline = ToolPipeline([self.budget])
            return owned, wiring

        def _wire_checkpoints(self, spec, journal, owned=None):
            bridge = super()._wire_checkpoints(spec, journal, owned)
            if args.benchmark == "swebench-pro":
                if not bridge or not bridge.exact_mode or owned.tracker is not self.budget:
                    raise RuntimeError("Pro mini budget/checkpoint wiring is missing")
                self.budget.emit = lambda **values: journal.emit("pro.tool_budget", **values)
            return bridge

        def _wire_gateway(self, spec, journal, task, run_id):
            if self.budget is not None:
                self.budget.control = task.control
            return super()._wire_gateway(spec, journal, task, run_id)

    original_orchestrator = fork_eval.Orchestrator
    original_analyst = fork_eval.ask_analyst
    original_parser = fork_eval.extract_branch_plan
    try:
        fork_eval.Orchestrator = ControlledOrchestrator
        fork_eval.ask_analyst = transport
        fork_eval.extract_branch_plan = structured_review.extract_branch_plan
        return fork_eval.main(_fork_args(args))
    finally:
        fork_eval.Orchestrator = original_orchestrator
        fork_eval.ask_analyst = original_analyst
        fork_eval.extract_branch_plan = original_parser


if __name__ == "__main__":
    raise SystemExit(main())
