"""SWE-bench / SWE-Gym runner for ash agent.

Usage:
    # SWE-bench (single instance)
    python -m swebench --instance sympy__sympy-15599

    # SWE-bench (batch)
    python -m swebench --subset verified --workers 4 -o results/

    # SWE-Gym
    python -m swebench --subset gym-lite -o results/
    python -m swebench --subset gym --slice 0:50 -o results/
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    sys.exit(1)

from .agent import AshAgent
from .ash_cli import AshSession
from . import style as S
from .tools import TOOLS_SCHEMA
from .models import AgentConfig


def load_swebench_instances(
    subset: str = "lite",
    split: str = "test",
    slice_spec: str = "",
    filter_regex: str = "",
) -> list[dict]:
    """Load SWE-bench instances from HuggingFace."""
    import re

    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
        "gym": "SWE-Gym/SWE-Gym",
        "gym-lite": "SWE-Gym/SWE-Gym-Lite",
    }
    dataset_name = dataset_map.get(subset, subset)

    # SWE-Gym uses 'train' split by default
    if not split:
        split = "train" if subset.startswith("gym") else "test"

    dataset = load_dataset(dataset_name, split=split)
    instances = list(dataset)

    if filter_regex:
        pattern = re.compile(filter_regex)
        instances = [i for i in instances if pattern.search(i.get("instance_id", ""))]

    if slice_spec:
        parts = slice_spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(instances)
        instances = instances[start:end]

    return instances


_IMAGE_REGISTRIES = {
    "swebench": ("swebench", "_1776_"),    # SWE-bench: swebench/sweb.eval.x86_64.{id}
    "xingyaoww": ("xingyaoww", "_s_"),     # SWE-Gym:   xingyaoww/sweb.eval.x86_64.{id}
}


def resolve_image(instance: dict, template: str = "", registry: str = "swebench") -> str:
    """Resolve Docker image name for a SWE-bench/SWE-Gym instance.

    Priority:
    1. instance["image_name"] — dataset-provided image name
    2. template — user-provided format string with {instance_id}, {repo}, {commit}
    3. Default: {registry}/sweb.eval.x86_64.{id}:latest
    """
    # Check dataset-provided image name first
    image_name = instance.get("image_name") or instance.get("env_image_key")
    if image_name:
        return image_name

    instance_id = instance.get("instance_id", "")

    if template:
        repo = instance.get("repo", "").replace("/", "__").lower()
        commit = instance.get("base_commit", "")[:12]
        return template.format(instance_id=instance_id, repo=repo, commit=commit)

    prefix, separator = _IMAGE_REGISTRIES.get(registry, ("swebench", "_1776_"))
    id_docker = instance_id.replace("__", separator)
    return f"{prefix}/sweb.eval.x86_64.{id_docker}:latest".lower()


def format_task_prompt(instance: dict) -> str:
    """Format a SWE-bench instance into a task prompt."""
    return f"""<issue>
{instance.get("problem_statement", "")}
</issue>

Repository: {instance.get("repo", "")}
You are working in /testbed which contains the repository at commit {instance.get("base_commit", "")}.

Fix the issue described above. Make minimal changes to the source code.
Do NOT modify test files. After making your changes, verify them by running relevant tests.
"""


def run_single_instance(
    instance: dict,
    config: AgentConfig,
    ash_binary: str,
    output_dir: Path,
    image_template: str = "",
    image_registry: str = "swebench",
) -> dict:
    """Run agent on a single SWE-bench/SWE-Gym instance."""
    instance_id = instance.get("instance_id", "unknown")
    print(S.header(instance_id))

    image = resolve_image(instance, template=image_template, registry=image_registry)
    print(S.kv("image   ", S.dim(image)))

    session = AshSession(ash_binary=ash_binary)

    try:
        if not session.create(image):
            return {
                "instance_id": instance_id,
                "model_patch": "",
                "model_name_or_path": config.model,
                "exit_status": "session_failed",
            }

        # Create and configure agent
        def _on_step(n: int, kind: str, text: str):
            print(S.step(n, kind, text), flush=True)

        trace_dir = output_dir / "traces"
        agent = AshAgent(config, executor=session.execute, on_step=_on_step, trace_dir=trace_dir)
        agent.set_tools_schema(TOOLS_SCHEMA)

        # Run agent loop
        task = format_task_prompt(instance)
        exit_status = agent.run(task, instance_id=instance_id)

        # Extract patch via ash CLI (automatic, no magic markers)
        patch = session.get_patch()

        # Save trajectory
        agent.trajectory.info = {
            "exit_status": exit_status,
            "submission": patch,
            "model": config.model,
        }
        agent.trajectory.cost = agent.cost
        traj_path = output_dir / "trajectories" / f"{instance_id}.json"
        agent.trajectory.save(traj_path)

        # Styled result output
        exit_color = S.green if exit_status == "completed" else S.yellow
        print(S.kv("exit    ", exit_color(exit_status)))
        print(S.kv("cost    ", S.cost(agent.cost.total_cost, agent.cost.api_calls)))
        print(S.kv("patch   ", S.patch_info(patch)))

        return {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": config.model,
            "exit_status": exit_status,
        }

    except Exception as e:
        print(S.kv("error   ", S.bright_red(str(e))))
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": config.model,
            "exit_status": f"error: {e}",
        }

    finally:
        session.destroy()


def run_batch(
    instances: list[dict],
    config: AgentConfig,
    ash_binary: str,
    output_dir: Path,
    workers: int = 1,
    image_template: str = "",
    image_registry: str = "swebench",
):
    """Run agent on multiple instances with resume support."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.json"

    # Resume: load existing predictions
    predictions = []
    if preds_path.exists():
        predictions = json.loads(preds_path.read_text())
        done_ids = {p["instance_id"] for p in predictions}
        instances = [i for i in instances if i["instance_id"] not in done_ids]
        print(f"  {S.green(S.CHECK)} Resuming: {S.bold(str(len(done_ids)))} done, "
              f"{S.bold(str(len(instances)))} remaining")

    def save():
        preds_path.write_text(json.dumps(predictions, indent=2))

    if workers <= 1:
        for i, inst in enumerate(instances):
            print(f"\n{S.progress(i + 1, len(instances))}")
            try:
                result = run_single_instance(inst, config, ash_binary, output_dir, image_template, image_registry)
                predictions.append(result)
                save()
            except KeyboardInterrupt:
                print(f"\n  {S.yellow('!')} Interrupted. Saving progress...")
                save()
                return
            except Exception as e:
                print(S.kv("error", S.bright_red(str(e))))
                predictions.append({
                    "instance_id": inst["instance_id"],
                    "model_patch": "",
                    "model_name_or_path": config.model,
                    "exit_status": f"error: {e}",
                })
                save()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_single_instance, inst, config, ash_binary, output_dir, image_template, image_registry): inst
                for inst in instances
            }
            for future in as_completed(futures):
                inst = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "instance_id": inst["instance_id"],
                        "model_patch": "",
                        "model_name_or_path": config.model,
                        "exit_status": f"error: {e}",
                    }
                predictions.append(result)
                save()

    # Summary
    submitted = sum(1 for p in predictions if p.get("model_patch"))
    print(S.summary(len(predictions), submitted, str(preds_path)))


def main():
    parser = argparse.ArgumentParser(description="Run ash agent on SWE-bench / SWE-Gym")

    # Dataset
    parser.add_argument("--subset", default="lite",
                        choices=["lite", "verified", "full", "gym", "gym-lite"],
                        help="Dataset subset (default: lite)")
    parser.add_argument("--split", default="",
                        help="Dataset split (default: 'train' for gym*, 'test' otherwise)")
    parser.add_argument("--instance", "-i", help="Single instance ID or index")
    parser.add_argument("--slice", help="Slice spec (e.g., '0:10')")
    parser.add_argument("--filter", help="Regex filter on instance IDs")

    # Model
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--api-base", default=None,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:30000/v1 for local SGLang)")
    parser.add_argument("--api-key", default=None,
                        help="API key / bearer token (default: auto-detect from env)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens per model call (caps thinking + response)")
    parser.add_argument("--step-limit", type=int, default=250)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Ash
    parser.add_argument("--ash-binary", default="ash",
                        help="Path to ash binary (default: ash)")
    parser.add_argument("--image-template", default="",
                        help="Docker image template with {instance_id}, {repo}, {commit} placeholders "
                             "(default: swebench/sweb.eval.x86_64.{instance_id}:latest)")

    # Execution
    parser.add_argument("--output", "-o", type=Path, default=Path("swebench_results"))
    parser.add_argument("--workers", "-w", type=int, default=1)

    args = parser.parse_args()

    # Banner
    print(S.banner())

    # Load instances
    instances = load_swebench_instances(
        subset=args.subset,
        split=args.split,
        slice_spec=args.slice or "",
        filter_regex=args.filter or "",
    )

    if args.instance:
        if args.instance.isdigit():
            idx = int(args.instance)
            if idx >= len(instances):
                print(f"  {S.bright_red('!')} Index {idx} out of range (max {len(instances) - 1})")
                sys.exit(1)
            instances = [instances[idx]]
        else:
            instances = [i for i in instances if i["instance_id"] == args.instance]
            if not instances:
                print(f"  {S.bright_red('!')} Instance not found: {args.instance}")
                sys.exit(1)

    # Resolve split for display (load_swebench_instances picks the default)
    display_split = args.split or ("train" if args.subset.startswith("gym") else "test")

    # Image registry: xingyaoww for SWE-Gym, swebench for SWE-bench
    image_registry = "xingyaoww" if args.subset.startswith("gym") else "swebench"

    print(f"  {S.green(S.CHECK)} Loaded {S.bold(str(len(instances)))} instances "
          f"from {S.cyan(f'{args.subset}/{display_split}')}")
    print(S.kv("model   ", S.dim(args.model)))

    # Build config — resolve provider, api_base, api_key
    model = args.model
    api_base = args.api_base
    api_key = args.api_key

    if model.startswith("bedrock/"):
        # AWS Bedrock: litellm handles endpoint, auth via bearer token
        api_key = api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            print(f"  {S.yellow('!')} Warning: no API key. Set AWS_BEARER_TOKEN_BEDROCK or --api-key")
    elif model.startswith(("anthropic/", "gemini/")):
        # Native providers: litellm handles routing, no api_base needed
        pass
    elif api_base:
        # Custom endpoint (local SGLang, vLLM, etc.): prefix with openai/
        if not model.startswith("openai/"):
            model = f"openai/{model}"
        api_key = api_key or "unused"
    else:
        # No api_base and no known prefix — assume local SGLang
        api_base = "http://localhost:30000/v1"
        if not model.startswith("openai/"):
            model = f"openai/{model}"
        api_key = api_key or "unused"

    agent_config = AgentConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        temperature=args.temperature,
    )

    # Run
    if len(instances) == 1:
        result = run_single_instance(instances[0], agent_config, args.ash_binary, args.output, args.image_template, image_registry)
        print(f"\n{S.section('Result')}")
        print(f"  {json.dumps(result, indent=2)}")
    else:
        run_batch(instances, agent_config, args.ash_binary, args.output, workers=args.workers, image_template=args.image_template, image_registry=image_registry)


if __name__ == "__main__":
    main()
