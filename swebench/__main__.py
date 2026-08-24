"""Unified CLI entry point for SWE-bench evaluation.

Usage:
    # LiteLLM harness (any model, custom agent loop)
    python -m swebench -c swebench/configs/bedrock-opus46.yaml

    # Claude Code harness
    python -m swebench -c swebench/configs/claude-opus.yaml --harness claude-code

    # Quick single instance
    python -m swebench --harness claude-code --model opus -i django__django-15732
"""

import argparse
import sys
from pathlib import Path

from . import style as S
from .backends import BackendError, backend_config, build_pool
from .dataset import load_instances
from .batch import run_batch
from .harnesses import HARNESSES, get_harness


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config(path: str, _seen: set | None = None) -> dict:
    import yaml
    from pathlib import Path

    if _seen is None:
        _seen = set()
    resolved = str(Path(path).resolve())
    if resolved in _seen:
        raise ValueError(f"Circular extends: {resolved}")
    _seen.add(resolved)

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    extends = config.pop("extends", None)
    if extends:
        configs_dir = Path(path).parent
        parent_path = configs_dir / f"{extends}.yaml"
        if not parent_path.exists():
            raise FileNotFoundError(f"extends: '{extends}' not found at {parent_path}")
        parent = _load_config(str(parent_path), _seen)
        config = _deep_merge(parent, config)

    return config


def _flatten_config(config: dict) -> dict:
    """Flatten nested YAML config into a flat dict."""
    mapping = {
        # model
        ("model", "name"): "model",
        ("model", "provider"): "provider",
        ("model", "api_base"): "api_base",
        ("model", "api_key"): "api_key",
        # generation (litellm harness)
        ("generation", "max_tokens"): "max_tokens",
        ("generation", "temperature"): "temperature",
        ("generation", "reasoning_effort"): "reasoning_effort",
        ("generation", "prompt_cache"): "prompt_cache",
        # agent prompts
        ("agent", "system_template"): "system_template",
        ("agent", "instance_template"): "instance_template",
        ("agent", "tools"): "tools",
        # limits (litellm harness)
        ("limits", "step_limit"): "step_limit",
        ("limits", "cost_limit"): "cost_limit",
        # claude (claude-code harness)
        ("claude", "max_budget"): "max_budget",
        ("claude", "timeout"): "timeout",
        ("claude", "permission_mode"): "permission_mode",
        # dataset
        ("dataset", "subset"): "subset",
        ("dataset", "split"): "split",
        ("dataset", "instance"): "instance",
        ("dataset", "slice"): "slice",
        ("dataset", "filter"): "filter",
        # A task directory on disk (SWE-Marathon), instead of a dataset.
        ("dataset", "task_dir"): "task_dir",
        # execution
        ("execution", "workers"): "workers",
        ("execution", "output"): "output",
        ("execution", "runtime_bin"): "runtime_bin",
        ("execution", "backend"): "backend",
        # Per-backend settings sections, kept whole so backends.py validates
        # their keys rather than every one becoming a flat entry here.
        ("execution", "docker"): "docker",
        ("execution", "microvm"): "microvm",
        ("execution", "k8s"): "k8s",
        ("execution", "image_template"): "image_template",
        ("execution", "harness"): "harness",
        # A Python file of your own interceptors; see agent/interceptors/.
        ("execution", "interceptors"): "interceptors",
        # Per-step environment checkpoints; kept whole like the backend
        # sections (enabled / mode / trigger / reboard are its keys).
        ("execution", "checkpoints"): "checkpoints",
        # Fractions of the model's own context window: elide above the first,
        # down to the second. 0 disables the guard.
        # elide (free) or summarize (one model call, keeps conclusions).
        ("execution", "context_strategy"): "context_strategy",
        ("execution", "context_budget_fraction"): "context_budget_fraction",
        ("execution", "context_target_fraction"): "context_target_fraction",
    }

    flat = {}

    # Extract nested values
    for (section, key), flat_key in mapping.items():
        if section in config and isinstance(config[section], dict):
            if key in config[section]:
                flat[flat_key] = config[section][key]

    # Top-level keys override. Mapped keys whose value is itself a settings
    # mapping (the per-backend sections) are included: skipping every dict here
    # would silently drop a top-level `microvm:` block, which reads as "my
    # settings were ignored" with nothing to point at.
    top_level = {v for v in mapping.values()}
    nested_ok = {"docker", "microvm", "k8s", "env"}
    for key in top_level:
        if key not in config:
            continue
        if isinstance(config[key], dict) and key not in nested_ok:
            continue
        flat[key] = config[key]

    # env section (passed through as-is)
    if "env" in config:
        flat["env"] = config["env"]

    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation with pluggable agent harnesses"
    )

    # Config
    parser.add_argument("--config", "-c", default=None, help="YAML config file")
    # Choices come from the registry: a hardcoded list silently omits a
    # harness the moment one is added, which is how `marathon` was rejected by
    # the CLI while being importable and registered.
    parser.add_argument("--harness", default=None,
                        choices=sorted(HARNESSES),
                        help="Agent harness (default: litellm)")

    # Dataset
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--instance", "-i", default=None)
    parser.add_argument("--slice", default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--task-dir", default=None,
                        help="SWE-Marathon task directory, or a checkout whose "
                             "tasks/ holds several (--harness marathon). Tasks "
                             "come from disk rather than a dataset, so this "
                             "replaces --subset/--instance for that harness.")

    # Model
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)

    # Execution
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--workers", "-w", type=int, default=None)
    parser.add_argument("--runtime-bin", default=None)
    parser.add_argument("--backend", default=None,
                        choices=["docker", "microvm", "k8s"],
                        help="Where sandboxes come from (default: docker). "
                             "Per-backend settings go under execution.<backend> "
                             "in the config; 'microvm' also reads "
                             "AENV_SERVER_URL / AENV_API_KEY.")
    parser.add_argument("--custom-tools-dir", default=None,
                        help="Directory of custom tool manifests (*.yaml/*.json). "
                             "Defaults to configs/custom_tools if present.")

    # LiteLLM-specific
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--step-limit", type=int, default=None)
    parser.add_argument("--cost-limit", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning-effort", default=None,
                        help="Adaptive thinking effort: low, medium, high, none")

    # Claude-code-specific
    parser.add_argument("--max-budget", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)

    args = parser.parse_args()

    # Load config file
    file_config = {}
    if args.config:
        file_config = _flatten_config(_load_config(args.config))

    # Merge: CLI > config > defaults
    def get(key, default=None):
        cli_val = getattr(args, key.replace("-", "_"), None)
        if cli_val is not None:
            return cli_val
        return file_config.get(key, default)

    harness_name = get("harness", "litellm")
    subset = get("subset", "verified")
    split = get("split", "")
    instance = get("instance")
    slice_spec = get("slice", "")
    filter_regex = get("filter", "")
    output = get("output", f"results/{harness_name}")
    workers = get("workers", 1)
    if isinstance(output, str):
        output = Path(output)
    if isinstance(workers, str):
        workers = int(workers)

    # Build harness config (pass everything through)
    harness_config = dict(file_config)
    # Override with CLI values where set
    for key in ["model", "provider", "api_base", "api_key", "max_tokens",
                "step_limit", "cost_limit", "temperature", "reasoning_effort",
                "prompt_cache", "max_budget", "timeout", "runtime_bin",
                "image_template", "subset", "workers", "custom_tools_dir",
                "backend"]:
        cli_val = getattr(args, key.replace("-", "_"), None)
        if cli_val is not None:
            harness_config[key] = cli_val
    harness_config.setdefault("subset", subset)
    harness_config.setdefault("workers", workers)

    # Validate the sandbox backend before anything expensive. A bad setting
    # would otherwise surface once per instance, as N identical failures with
    # the real cause buried in each.
    try:
        build_pool(backend_config(harness_config))
    except BackendError as exc:
        print(f"  {S.bright_red('!')} {exc}")
        return 2
    except Exception:
        pass          # reachability is the run's problem, not the config's

    # Banner
    print(S.banner())

    # Load instances. Marathon tasks live on disk, not in a dataset, so the
    # task directory is the source; everything downstream (batch, resume,
    # dashboard) then treats them like any other instance.
    task_dir = get("task_dir")
    if task_dir:
        from .marathon import discover_tasks, load_task
        directory = Path(task_dir)
        tasks = ([load_task(directory)] if (directory / "task.toml").exists()
                 else discover_tasks(directory))
        if not tasks:
            print(f"  {S.bright_red('!')} no SWE-Marathon tasks under {directory}")
            sys.exit(1)
        instances = [{"instance_id": t.instance_id,
                      "task_dir": str(t.directory)} for t in tasks]
        harness_config.setdefault("task_dir", str(directory))
        if filter_regex:
            import re
            pattern = re.compile(filter_regex)
            instances = [i for i in instances if pattern.search(i["instance_id"])]
    else:
        instances = load_instances(
            subset=subset,
            split=split,
            slice_spec=slice_spec,
            filter_regex=filter_regex,
        )

    if instance:
        if instance.isdigit():
            idx = int(instance)
            if idx >= len(instances):
                print(f"  {S.bright_red('!')} Index {idx} out of range")
                sys.exit(1)
            instances = [instances[idx]]
        else:
            instances = [i for i in instances if i["instance_id"] == instance]
            if not instances:
                print(f"  {S.bright_red('!')} Instance not found: {instance}")
                sys.exit(1)

    # Info
    model_display = harness_config.get("model", "default")
    print(f"  {S.green(S.CHECK)} Loaded {S.bold(str(len(instances)))} instances "
          f"from {S.cyan(subset)}")
    print(S.kv("harness ", S.dim(harness_name)))
    print(S.kv("model   ", S.dim(model_display)))

    # Instantiate harness
    HarnessClass = get_harness(harness_name)
    harness = HarnessClass(harness_config)

    # Run — pass harness so batch can inject dashboard
    def run_fn(inst):
        return harness.run_instance(inst, output)

    run_batch(instances, run_fn, output, workers=workers, harness=harness)


if __name__ == "__main__":
    raise SystemExit(main())
