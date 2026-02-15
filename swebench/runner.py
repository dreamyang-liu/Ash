#!/usr/bin/env python3
"""
SWE-bench runner for ash agent.

Usage:
    # Single instance (interactive/debug)
    python -m swebench.runner --instance sympy__sympy-15599 --model claude-sonnet-4-5-20250929

    # Batch mode
    python -m swebench.runner --subset verified --split test --workers 4 -o results/

    # Evaluate with sb-cli
    sb-cli submit swe-bench_verified test --predictions_path results/preds.json --run_id ash-run-1
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    sys.exit(1)

from . import AgentConfig, Trajectory
from .agent import AshAgent
from .docker_env import DockerConfig, DockerEnvironment, create_docker_executor


def load_swebench_instances(subset: str = "lite", split: str = "dev", slice_spec: str = "", filter_regex: str = ""):
    """Load SWE-bench instances."""
    import re
    
    # Map subset names
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }
    dataset_name = dataset_map.get(subset, subset)
    
    # Load
    dataset = load_dataset(dataset_name, split=split)
    instances = list(dataset)
    
    # Apply filters
    if filter_regex:
        pattern = re.compile(filter_regex)
        instances = [i for i in instances if pattern.search(i.get("instance_id", ""))]
    
    if slice_spec:
        # Parse slice like "0:10" or ":5" or "10:"
        parts = slice_spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(instances)
        instances = instances[start:end]
    
    return instances


def format_task_prompt(instance: dict) -> str:
    """Format a SWE-bench instance into a task prompt."""
    return f"""<pr_description>
{instance.get('problem_statement', '')}
</pr_description>

Repository: {instance.get('repo', '')}
Base commit: {instance.get('base_commit', '')}

Your task is to fix the issue described above by modifying the source files in /testbed.
Do NOT modify tests or configuration files.

When done:
1. Create patch: git diff -- <modified_files> > patch.txt
2. Verify patch.txt
3. Submit: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
"""


def run_single_instance(
    instance: dict,
    config: AgentConfig,
    output_dir: Path,
    use_docker: bool = True,
) -> dict:
    """Run agent on a single instance."""
    instance_id = instance.get("instance_id", "unknown")
    print(f"\n{'='*60}")
    print(f"Running: {instance_id}")
    print(f"{'='*60}")
    
    # Set up environment
    if use_docker:
        docker_config = DockerConfig(
            ash_binary=config.ash_binary,  # Will be copied into container
        )
        env = DockerEnvironment(docker_config)
        
        if not env.start(instance):
            return {
                "instance_id": instance_id,
                "exit_status": "EnvironmentError",
                "submission": "",
            }
        
        executor = create_docker_executor(env)
    else:
        # Local execution (for testing)
        from . import call_ash_tool
        executor = lambda name, args: call_ash_tool(name, args, config.ash_binary)
    
    try:
        # Create agent
        agent = AshAgent(config, executor=executor)
        
        # Run
        task = format_task_prompt(instance)
        result = agent.run(task, instance_id=instance_id)
        
        # Save trajectory
        traj_path = output_dir / "trajectories" / f"{instance_id}.json"
        agent.trajectory.save(traj_path)
        
        print(f"\nResult: {result.get('exit_status', 'Unknown')}")
        if result.get("submission"):
            print(f"Patch length: {len(result['submission'])} chars")
        
        return {
            "instance_id": instance_id,
            "model_patch": result.get("submission", ""),
            "model_name_or_path": config.model,
            "exit_status": result.get("exit_status", ""),
        }
        
    finally:
        if use_docker:
            env.stop()


def run_batch(
    instances: list,
    config: AgentConfig,
    output_dir: Path,
    workers: int = 1,
    use_docker: bool = True,
):
    """Run agent on multiple instances."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    preds_path = output_dir / "preds.json"
    
    # Load existing predictions if resuming
    if preds_path.exists():
        with open(preds_path) as f:
            existing = json.load(f)
            existing_ids = {p["instance_id"] for p in existing}
            predictions = existing
            instances = [i for i in instances if i["instance_id"] not in existing_ids]
            print(f"Resuming: {len(existing)} done, {len(instances)} remaining")
    
    def save_predictions():
        with open(preds_path, "w") as f:
            json.dump(predictions, f, indent=2)
    
    if workers <= 1:
        # Sequential
        for i, instance in enumerate(instances):
            print(f"\n[{i+1}/{len(instances)}]")
            try:
                result = run_single_instance(instance, config, output_dir, use_docker)
                predictions.append(result)
                save_predictions()
            except KeyboardInterrupt:
                print("\nInterrupted! Saving progress...")
                save_predictions()
                return
            except Exception as e:
                print(f"Error: {e}")
                predictions.append({
                    "instance_id": instance["instance_id"],
                    "model_patch": "",
                    "model_name_or_path": config.model,
                    "exit_status": f"Error: {e}",
                })
                save_predictions()
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_single_instance, inst, config, output_dir, use_docker): inst
                for inst in instances
            }
            
            for future in as_completed(futures):
                instance = futures[future]
                try:
                    result = future.result()
                    predictions.append(result)
                except Exception as e:
                    print(f"Error on {instance['instance_id']}: {e}")
                    predictions.append({
                        "instance_id": instance["instance_id"],
                        "model_patch": "",
                        "model_name_or_path": config.model,
                        "exit_status": f"Error: {e}",
                    })
                save_predictions()
    
    print(f"\n{'='*60}")
    print(f"Completed! Results saved to {preds_path}")
    print(f"Total: {len(predictions)}")
    submitted = sum(1 for p in predictions if p.get("model_patch"))
    print(f"Submitted patches: {submitted}")


def main():
    parser = argparse.ArgumentParser(description="Run ash agent on SWE-bench")
    
    # Data selection
    parser.add_argument("--subset", default="lite", help="SWE-bench subset (lite/verified/full)")
    parser.add_argument("--split", default="dev", help="Dataset split (dev/test)")
    parser.add_argument("--instance", "-i", help="Single instance ID or index")
    parser.add_argument("--slice", help="Slice spec (e.g., '0:10')")
    parser.add_argument("--filter", help="Filter instance IDs by regex")
    
    # Model config
    parser.add_argument("--model", "-m", default="anthropic/claude-sonnet-4-5-20250929")
    parser.add_argument("--step-limit", type=int, default=250)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    
    # Execution
    parser.add_argument("--output", "-o", type=Path, default=Path("swebench_results"))
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--no-docker", action="store_true", help="Run locally without Docker")
    parser.add_argument("--ash-binary", default="ash", help="Path to ash binary")
    
    args = parser.parse_args()
    
    # Load instances
    instances = load_swebench_instances(
        subset=args.subset,
        split=args.split,
        slice_spec=args.slice or "",
        filter_regex=args.filter or "",
    )
    
    if args.instance:
        # Single instance mode
        if args.instance.isdigit():
            idx = int(args.instance)
            instances = [instances[idx]]
        else:
            instances = [i for i in instances if i["instance_id"] == args.instance]
            if not instances:
                print(f"Instance not found: {args.instance}")
                sys.exit(1)
    
    print(f"Loaded {len(instances)} instances from {args.subset}/{args.split}")
    
    # Create config
    config = AgentConfig(
        model=args.model,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        temperature=args.temperature,
        ash_binary=args.ash_binary,
    )
    
    # Run
    if len(instances) == 1:
        result = run_single_instance(
            instances[0],
            config,
            args.output,
            use_docker=not args.no_docker,
        )
        print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        run_batch(
            instances,
            config,
            args.output,
            workers=args.workers,
            use_docker=not args.no_docker,
        )


if __name__ == "__main__":
    main()
