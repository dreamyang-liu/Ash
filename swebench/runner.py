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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    sys.exit(1)

from .agent import AshAgent
from .sandbox import AshSession
from . import style as S
from .agent.tools import TOOLS_SCHEMA, BASH_ONLY_SCHEMA
from .models import AgentConfig


def load_swebench_instances(
    subset: str = "lite",
    split: str = "",
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
    """Resolve Docker image name for a SWE-bench/SWE-Gym instance."""
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


def _cleanup_containers():
    """Kill all ash-managed containers (label: ash.managed=1)."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=ash.managed=1"],
            capture_output=True, text=True, timeout=10,
        )
        cids = result.stdout.strip().split()
        if cids:
            subprocess.run(
                ["docker", "rm", "-f", *cids],
                capture_output=True, timeout=30,
            )
            print(S.kv("cleanup ", S.dim(f"removed {len(cids)} containers")))
    except Exception:
        pass


class _ContainerJanitor:
    """Background thread that periodically cleans up leaked containers."""

    def __init__(self, interval: int = 60):
        self._interval = interval
        self._active_cids: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        # Final sweep
        _cleanup_containers()

    def register(self, cid: str):
        """Mark a container as actively in use."""
        with self._lock:
            self._active_cids.add(cid)

    def unregister(self, cid: str):
        """Mark a container as no longer needed."""
        with self._lock:
            self._active_cids.discard(cid)

    def _run(self):
        import subprocess
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break
            try:
                result = subprocess.run(
                    ["docker", "ps", "-q", "--filter", "label=ash.managed=1"],
                    capture_output=True, text=True, timeout=10,
                )
                all_cids = set(result.stdout.strip().split()) - {""}
                with self._lock:
                    leaked = all_cids - self._active_cids
                if leaked:
                    subprocess.run(
                        ["docker", "rm", "-f", *leaked],
                        capture_output=True, timeout=30,
                    )
            except Exception:
                pass


def run_single_instance(
    instance: dict,
    config: AgentConfig,
    output_dir: Path,
    image_template: str = "",
    image_registry: str = "swebench",
    runtime_bin: str | None = None,
    workers: int = 1,
    on_step_callback=None,
    janitor: "_ContainerJanitor | None" = None,
) -> dict:
    """Run agent on a single SWE-bench/SWE-Gym instance."""
    instance_id = instance.get("instance_id", "unknown")
    quiet = workers > 1  # suppress direct prints in parallel mode

    if not quiet:
        print(S.header(instance_id))

    image = resolve_image(instance, template=image_template, registry=image_registry)
    if not quiet:
        print(S.kv("image   ", S.dim(image)))

    session = AshSession(runtime_bin=runtime_bin, quiet=quiet)

    try:
        if not session.create(image):
            return {
                "instance_id": instance_id,
                "model_patch": "",
                "model_name_or_path": config.model,
                "exit_status": "session_failed",
            }

        # Register container with janitor
        if janitor and session._sandbox and session._sandbox._container_id:
            janitor.register(session._sandbox._container_id)

        # Create and configure agent — rolling step display
        _step_lines: list[str] = []
        _max_visible = 30
        _agent_ref: list = []  # mutable container to reference agent from closure

        def _on_step(n: int, kind: str, text: str):
            line = S.step(n, kind, text)
            _step_lines.append(line)

            if quiet:
                if on_step_callback:
                    cost = _agent_ref[0].cost.total_cost if _agent_ref else 0
                    on_step_callback(n, kind, text, cost)
                return

            if not S._IS_TTY:
                print(line, flush=True)
                return

            # Single worker + TTY: rolling window
            visible = _step_lines[-_max_visible:]
            if len(_step_lines) > 1:
                clear_count = min(len(_step_lines) - 1, _max_visible)
                sys.stdout.write(f"\033[{clear_count}A")
            for l in visible:
                sys.stdout.write(f"\033[K{l}\n")
            sys.stdout.flush()

        trace_dir = output_dir / "traces"
        agent = AshAgent(config, executor=session.execute, on_step=_on_step, trace_dir=trace_dir)
        _agent_ref.append(agent)
        if quiet:
            agent.stream = False  # no streaming in parallel (avoids stdout conflicts)
        tools_mode = getattr(config, "tools", "default")
        agent.set_tools_schema(BASH_ONLY_SCHEMA if tools_mode == "bash_only" else TOOLS_SCHEMA)

        # Run agent loop
        task = format_task_prompt(instance)
        exit_status = agent.run(task, instance_id=instance_id)

        # Extract patch
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

        if not quiet:
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
        if not quiet:
            print(S.kv("error   ", S.bright_red(str(e))))
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": config.model,
            "exit_status": f"error: {e}",
        }

    finally:
        if janitor and session._sandbox and session._sandbox._container_id:
            janitor.unregister(session._sandbox._container_id)
        session.destroy()


class _Dashboard:
    """Live dashboard for parallel instance execution."""

    def __init__(self, instance_ids: list[str]):
        self._ids = instance_ids
        self._state: dict[str, dict] = {
            iid: {"status": "waiting", "step": 0, "detail": ""}
            for iid in instance_ids
        }
        self._lock = threading.Lock()
        self._rendered_lines = 0

    def update(self, instance_id: str, status: str = "", step: int = 0, detail: str = "", cost: float = 0):
        with self._lock:
            s = self._state.get(instance_id, {})
            if status:
                s["status"] = status
            if step:
                s["step"] = step
            if detail:
                s["detail"] = detail
            if cost:
                s["cost"] = cost
            self._state[instance_id] = s

    def render(self):
        if not S._IS_TTY:
            return
        with self._lock:
            total_cost = 0.0
            n_done = 0
            n_failed = 0
            n_running = 0
            running_lines = []

            n_waiting = 0
            for iid in self._ids:
                s = self._state[iid]
                status = s["status"]
                cost = s.get("cost", 0)
                total_cost += cost

                if status == "done":
                    n_done += 1
                elif status == "failed":
                    n_failed += 1
                elif status == "waiting":
                    n_waiting += 1
                elif status in ("running", "spawning"):
                    n_running += 1
                    short_id = iid.split("__")[-1][:20].ljust(20)
                    step = s["step"]
                    detail = s.get("detail", "").replace("\n", " ")[:36]
                    cost_str = f"${cost:.2f}".rjust(6) if cost else "      "

                    if status == "running":
                        tag = S.neon_cyan(f"step {step:>3}")
                    else:
                        tag = S.neon_purple("spawn ")

                    running_lines.append(f"\033[K  {S.dim(short_id)} [{tag}] {S.neon_orange(cost_str)}  {detail}")

            # Header: counts summary
            header_parts = []
            if n_done:
                header_parts.append(S.neon_green(f"◆ {n_done} resolved"))
            if n_failed:
                header_parts.append(S.neon_pink(f"✘ {n_failed} failed"))
            if n_running:
                header_parts.append(S.neon_cyan(f"⚡{n_running} running"))
            if n_waiting:
                header_parts.append(S.dim(f"… {n_waiting} queued"))
            header = "  ".join(header_parts)
            total_str = S.neon_orange(f"${total_cost:.2f}")
            summary_line = f"\033[K  {header}    total: {total_str}"

            lines = [summary_line]
            if running_lines:
                lines.append(f"\033[K  {S.neon_purple('╌' * 56)}")
                lines.extend(running_lines)

            # Build full frame and write atomically
            frame = ""
            if self._rendered_lines > 0:
                frame = f"\033[{self._rendered_lines}A"
            # Pad with empty lines if frame shrank (instances finishing)
            while len(lines) < self._rendered_lines:
                lines.append("\033[K")
            frame += "\n".join(lines) + "\n"
            sys.stdout.write(frame)
            sys.stdout.flush()
            self._rendered_lines = len(lines)

    def final_summary(self):
        """Print final non-overwritable summary after dashboard stops."""
        if S._IS_TTY and self._rendered_lines > 0:
            sys.stdout.write(f"\033[{self._rendered_lines}A")
        total_cost = 0.0
        with self._lock:
            for iid in self._ids:
                s = self._state[iid]
                short_id = iid.split("__")[-1][:20]
                status = s["status"]
                cost = s.get("cost", 0)
                total_cost += cost
                cost_str = f"${cost:.2f}" if cost else ""
                detail = s.get("detail", "").replace("\n", " ")[:40]
                if status == "done":
                    tag = S.neon_green("◆")
                    info = S.dim(detail)
                elif status == "failed":
                    tag = S.neon_pink("✘")
                    info = S.dim(detail)
                else:
                    tag = S.dim("·")
                    info = S.dim(status)
                print(f"  {tag} {short_id:<22} {S.neon_orange(cost_str):>8}  {info}")
            print(f"  {S.neon_purple('╌' * 56)}")
            print(f"  {'total':<24} {S.neon_orange(f'${total_cost:.2f}')}")


def run_batch(
    instances: list[dict],
    config: AgentConfig,
    output_dir: Path,
    workers: int = 1,
    image_template: str = "",
    image_registry: str = "swebench",
    runtime_bin: str | None = None,
):
    """Run agent on multiple instances with resume support."""
    janitor = None  # Disabled: was killing containers mid-run due to race condition
    # janitor = _ContainerJanitor(interval=30)
    # janitor.start()

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
                result = run_single_instance(inst, config, output_dir, image_template, image_registry, runtime_bin, workers=workers, janitor=janitor)
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
        # Parallel mode with live dashboard
        instance_ids = [i["instance_id"] for i in instances]
        dashboard = _Dashboard(instance_ids)

        # Render loop in background
        stop_render = threading.Event()

        def _render_loop():
            while not stop_render.is_set():
                dashboard.render()
                time.sleep(0.5)

        # Initial render before starting threads (establishes cursor position)
        dashboard.render()

        render_thread = threading.Thread(target=_render_loop, daemon=True)
        render_thread.start()

        def _run_with_dashboard(inst: dict) -> dict:
            iid = inst["instance_id"]
            dashboard.update(iid, status="spawning")

            def _cb(n, kind, text, cost=0):
                detail = text.replace("\n", " ")[:36]
                tag_map = {
                    "shell":       S.neon_cyan("$"),
                    "grep_files":  S.neon_purple("grep"),
                    "read_file":   S.electric_blue("read"),
                    "text_editor": S.neon_orange("edit"),
                    "process":     S.neon_green("proc"),
                }
                prefix = tag_map.get(kind, S.dim(kind[:4]))
                dashboard.update(iid, status="running", step=n, detail=f"{prefix} {S.dim(detail)}", cost=cost)

            try:
                result = run_single_instance(
                    inst, config, output_dir, image_template, image_registry,
                    runtime_bin, workers=workers, on_step_callback=_cb, janitor=janitor,
                )
                patch = result.get("model_patch", "")
                if patch:
                    dashboard.update(iid, status="done", detail=f"patch: {len(patch)} chars")
                else:
                    exit_st = result.get("exit_status", "unknown")
                    dashboard.update(iid, status="failed", detail=exit_st)
                return result
            except Exception as e:
                dashboard.update(iid, status="failed", detail=str(e)[:40])
                return {
                    "instance_id": iid,
                    "model_patch": "",
                    "model_name_or_path": config.model,
                    "exit_status": f"error: {e}",
                }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_with_dashboard, inst): inst for inst in instances}
            for future in as_completed(futures):
                result = future.result()
                predictions.append(result)
                save()

        stop_render.set()
        render_thread.join(timeout=1)
        dashboard.final_summary()
        print()

    # Stop janitor (does a final sweep)
    if janitor:
        janitor.stop()

    # Summary
    submitted = sum(1 for p in predictions if p.get("model_patch"))
    print(S.summary(len(predictions), submitted, str(preds_path)))


def _load_config_file(path: str) -> dict:
    """Load a YAML or JSON config file. YAML honors `extends:` inheritance via
    the unified loader so the runner path matches `python -m swebench`."""
    if path.endswith((".yml", ".yaml")):
        from .__main__ import _load_config
        return _load_config(path)
    with open(path) as f:
        return json.loads(f.read())


def _flatten_config(config: dict) -> dict:
    """Flatten a nested YAML config into a flat dict.

    Supports both flat and nested formats:
      Flat:   {model: "...", max_tokens: 16384}
      Nested: {model: {name: "..."}, generation: {max_tokens: 16384}}
    """
    # Mapping from nested paths to flat keys
    nested_map = {
        ("model", "name"): "model",
        ("model", "api_base"): "api_base",
        ("model", "api_key"): "api_key",
        ("generation", "max_tokens"): "max_tokens",
        ("generation", "temperature"): "temperature",
        ("generation", "reasoning_effort"): "reasoning_effort",
        ("generation", "prompt_cache"): "prompt_cache",
        ("limits", "step_limit"): "step_limit",
        ("limits", "cost_limit"): "cost_limit",
        ("dataset", "subset"): "subset",
        ("dataset", "split"): "split",
        ("dataset", "instance"): "instance",
        ("dataset", "slice"): "slice",
        ("dataset", "filter"): "filter",
        ("execution", "workers"): "workers",
        ("execution", "output"): "output",
        ("execution", "runtime_bin"): "runtime_bin",
        ("execution", "image_template"): "image_template",
    }

    flat = {}

    # First pass: extract nested values
    for (section, key), flat_key in nested_map.items():
        if section in config and isinstance(config[section], dict):
            if key in config[section]:
                flat[flat_key] = config[section][key]

    # Second pass: top-level flat keys override (for backwards compat)
    top_level_keys = {
        "model", "api_base", "api_key", "max_tokens", "step_limit",
        "cost_limit", "temperature", "reasoning_effort", "prompt_cache", "subset", "split",
        "instance", "slice", "filter", "runtime_bin", "image_template",
        "output", "workers",
    }
    for key in top_level_keys:
        if key in config and not isinstance(config[key], dict):
            flat[key] = config[key]

    return flat


def _merge_config(args, config: dict):
    """Merge config file values into args (CLI flags take precedence)."""
    flat = _flatten_config(config)

    for yaml_key in flat:
        attr_name = yaml_key
        current = getattr(args, attr_name, None)
        # CLI flag wins if explicitly set (not default/None/empty)
        if current is None or current == "" or (attr_name == "subset" and current == "lite"):
            value = flat[yaml_key]
            if attr_name == "output" and isinstance(value, str):
                value = Path(value)
            setattr(args, attr_name, value)


def main():
    parser = argparse.ArgumentParser(description="Run ash agent on SWE-bench / SWE-Gym")

    # Config file
    parser.add_argument("--config", "-c", default=None,
                        help="Path to YAML/JSON config file")

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
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--api-base", default=None,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:30000/v1 for local SGLang)")
    parser.add_argument("--api-key", default=None,
                        help="API key / bearer token (default: auto-detect from env)")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--step-limit", type=int, default=None)
    parser.add_argument("--cost-limit", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning-effort", default=None,
                        help="Adaptive thinking effort: low, medium, high, none")

    # Sandbox
    parser.add_argument("--runtime-bin", default=None,
                        help="Path to ash-runtime binary (default: auto-detect from PATH or download)")
    parser.add_argument("--image-template", default="",
                        help="Docker image template with {instance_id}, {repo}, {commit} placeholders "
                             "(default: swebench/sweb.eval.x86_64.{instance_id}:latest)")

    # Execution
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--workers", "-w", type=int, default=None)

    args = parser.parse_args()

    # Load config file and merge (CLI flags override)
    if args.config:
        config = _load_config_file(args.config)
        _merge_config(args, config)

    # Apply defaults for values not set by config or CLI
    if args.model is None:
        args.model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    if args.max_tokens is None:
        args.max_tokens = 8192
    if args.step_limit is None:
        args.step_limit = 250
    if args.cost_limit is None:
        args.cost_limit = 3.0
    # temperature: None means use model default (don't force 0.0)
    if args.output is None:
        args.output = Path("swebench_results")
    if args.workers is None:
        args.workers = 1

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

    # Resolve split for display
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
        api_key = api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            print(f"  {S.yellow('!')} Warning: no API key. Set AWS_BEARER_TOKEN_BEDROCK or --api-key")
    elif model.startswith(("anthropic/", "gemini/")):
        pass
    elif api_base:
        if not model.startswith("openai/"):
            model = f"openai/{model}"
        api_key = api_key or "unused"
    else:
        api_base = "http://localhost:30000/v1"
        if not model.startswith("openai/"):
            model = f"openai/{model}"
        api_key = api_key or "unused"

    agent_config = AgentConfig(
        tools=getattr(args, "tools", "default"),
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        temperature=args.temperature,
        reasoning_effort=getattr(args, "reasoning_effort", None),
        prompt_cache=getattr(args, "prompt_cache", True),
    )

    # Run
    if len(instances) == 1:
        result = run_single_instance(
            instances[0], agent_config, args.output,
            args.image_template, image_registry, args.runtime_bin,
            workers=1,
        )
        print(f"\n{S.section('Result')}")
        print(f"  {json.dumps(result, indent=2)}")
    else:
        run_batch(
            instances, agent_config, args.output,
            workers=args.workers, image_template=args.image_template,
            image_registry=image_registry, runtime_bin=args.runtime_bin,
        )


if __name__ == "__main__":
    main()
