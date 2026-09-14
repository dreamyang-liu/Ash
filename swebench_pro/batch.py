"""Freeze and launch a single-pass Pro batch with a persistent controller."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from swebench_pro.tasks import DATASET_REVISION, HARNESS_REVISION, load_tasks
from swebench_pro.worker import api, now, save


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def summary(root: Path, manifest: dict) -> dict:
    rows = [read(root / f"shard-{item['index']:03d}" / "worker.json") for item in manifest["tasks"]]
    completed = [row for row in rows if row.get("finished_at")]
    valid = [row for row in completed if row.get("evidence_valid")]
    successes = [row["task"] for row in valid if row.get("resolved")]
    total = len(rows)
    return {"updated_at": now(), "total_tasks": total, "started": sum(bool(row) for row in rows),
            "finished": len(completed), "valid_grades": len(valid), "resolved": len(successes),
            "held": [row["task"] for row in completed if not row.get("evidence_valid")],
            "resolved_ids": successes, "resolved_lower_bound": len(successes) / total,
            "final_resolved_rate": len(successes) / total if len(valid) == total else None,
            "phases": dict(Counter(row.get("phase", "queued") for row in rows if not row.get("finished_at"))),
            "cost_usd": sum((row.get("usage") or {}).get("cost_usd", 0) or 0 for row in rows),
            "rollouts_per_task": 1, "branching": False,
            "recovery_actions": dict(Counter(item.get("recovery_action", "fresh") for item in manifest["tasks"])),
            "approved_restart_ids": manifest.get("approved_restart_ids", [])}


def configure_credentials() -> None:
    if not os.environ.get("AENV_API_KEY"):
        raise RuntimeError("Set the current AENV_API_KEY before launching")
    os.environ.setdefault("AENV_SERVER_URL", "http://127.0.0.1:18000")
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        settings = Path.home() / ".claude/settings.json"
        token = read(settings).get("env", {}).get("AWS_BEARER_TOKEN_BEDROCK")
        if token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        raise RuntimeError("Missing Bedrock credentials")
    os.environ.update(CLAUDE_CODE_USE_BEDROCK="1", AWS_REGION="us-west-2")


def prepare(root: Path, upstream: Path, runtime: Path, workers: int, model: str,
            agent_network: str | None = None, verifier_network: str | None = None) -> dict:
    if any(value not in (None, "allow", "deny") for value in (agent_network, verifier_network)):
        raise ValueError("Network policies must be allow or deny")
    configure_credentials()
    existing = api("/sandboxes")
    tasks = load_tasks(upstream)
    if len(tasks) != 731:
        raise RuntimeError(f"Pinned public cohort expected 731 tasks, got {len(tasks)}")
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".venv", "*.egg-info")
    for name in ("harness", "swebench", "swebench_pro", "sdk", "deepswe", "contracts", "runtime"):
        shutil.copytree(source / name, root / "source" / name, ignore=ignore)
    shutil.copy2(runtime, root / "source/runtime/ash-runtime")
    (root / "upstream").mkdir()
    for name in (".git", "dockerfiles", "run_scripts", "helper_code"):
        shutil.copytree(upstream / name, root / "upstream" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "modules"))
    for name in ("swe_bench_pro_eval.py", "README.md", ".gitmodules"):
        shutil.copy2(upstream / name, root / "upstream" / name)
    sources = ["SWE-agent/config/tool_use.yaml", "SWE-agent/sweagent_wrapper_configs/example_config.yaml",
               "SWE-agent/swerex_patches/swerex/deployment/modal.py", "SWE-agent/sweagent/tools/tools.py"]
    for name in sources:
        destination = root / "official-config" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upstream / name, destination)
    canary_ids = [next(task.instance_id for task in tasks.values() if task.repo == repo)
                  for repo in ("ansible/ansible", "flipt-io/flipt")]
    ordered = sorted(tasks.values(), key=lambda task: (task.instance_id not in canary_ids, task.instance_id))
    items = []
    for index, task in enumerate(ordered):
        path = root / "tasks" / f"{task.instance_id}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text(task.sample_json + "\n")
        items.append({"index": index, "id": task.instance_id, "image": task.image,
                      "repo": task.repo, "canary": task.instance_id in canary_ids,
                      "data": str(path.relative_to(root))})
    hashes = {}
    for folder in ("source", "tasks", "official-config"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    import claude_agent_sdk

    binary = Path(claude_agent_sdk.__file__).parent / "_bundled/claude"
    manifest = {"created_at": now(), "request": "那你做一下rollout，先做所有task的单次rollout，然后timeout configuration之类的尝试用官方的config",
                "model": model, "slot": "claude-code", "tasks": items, "cohort_size": 731,
                "dataset_revision": DATASET_REVISION, "harness_revision": HARNESS_REVISION,
                "sweagent_revision": "402a7b8fdac8193f3f255bb53859ba274234f596",
                "actor_timeout_s": 3600, "tool_timeout_s": 450, "total_tool_seconds": 1800,
                "consecutive_timeouts": 3, "verifier_timeout_s": 3600, "mcp_timeout_ms": 3600000,
                "cost_limit": None, "model_call_limit": None, "rounds": 0, "automatic_retries": 2,
                "failure_policy": "isolated", "max_infra_retries": 2, "retry_backoff_s": [30, 60],
                "max_workers": workers, "prepare_workers": 4, "grade_workers": 4,
                "canaries": canary_ids, "cpu": 4, "memory_mb": 16384,
                "agent_network": agent_network, "verifier_network": verifier_network,
                "server_url": os.environ["AENV_SERVER_URL"], "sdk_version": version("claude-agent-sdk"),
                "bundled_cli": str(binary), "bundled_cli_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "sha256": hashes, "initial_sandboxes": [row["sandboxID"] for row in existing],
                "adaptations": ["Claude Code / Sonnet 4.6 retained instead of upstream SWE-agent/model",
                                "AgentENV microVMs; 4 CPU / 16 GiB per VM",
                                "3600s actor wall clock excludes image preparation",
                                "No debug 3-call limit; no monetary cap (upstream example disables it)",
                                "Native sampling settings retained; not claimed to match SWE-agent temperature",
                                "Outer MCP wait 3600s; inner command capped at450s, requested shorter timeouts honored",
                                "Canary admission requires live tool/checkpoint evidence, not a successful solution"]}
    save(root / "manifest.json", manifest)
    if shutil.disk_usage(root).free < 150 * 1024**3:
        raise RuntimeError("Insufficient disk for batch admission")
    save(root / "preflight.json", {"passed": True, "at": now(), "tasks": len(items),
                                   "runtime_sha256": hashes["source/runtime/ash-runtime"],
                                   "free_bytes": shutil.disk_usage(root).free, "model_calls": 0})
    return manifest


def controller(root: Path) -> int:
    manifest = read(root / "manifest.json")
    if manifest.get("failure_policy") == "isolated":
        from swebench_pro.retry_queue import controller as isolated_controller

        return isolated_controller(root)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    pending = [item for item in manifest["tasks"] if not (root / f"shard-{item['index']:03d}").exists()]
    pending.sort(key=lambda item: not item["canary"])
    active = {}
    stopping = False
    gate_open = False
    started = time.monotonic()
    failures = 0

    def stop(signum: int, frame: Any) -> None:
        save(root / "STOP_REQUEST.json", {"reason": "controller signal", "signal": signum})

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while pending or active:
        for index, entry in list(active.items()):
            code = entry["process"].poll()
            if code is not None:
                entry["log"].close()
                state_path = root / f"shard-{index:03d}/worker.json"
                state = read(state_path)
                if not state.get("finished_at"):
                    state.update(task=entry["task"], status="held", finished_at=now(), error=f"worker exited {code}", evidence_valid=False)
                    save(state_path, state)
                    save(root / "STOP_REQUEST.json", {"reason": "worker exited without cleanup evidence", "task": entry["task"]})
                del active[index]
        if not stopping:
            try:
                api("/sandboxes")
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= 2:
                    save(root / "STOP_REQUEST.json", {"reason": "AgentENV health failed", "error": str(exc)})
            if shutil.disk_usage(root).free < 80 * 1024**3:
                save(root / "STOP_REQUEST.json", {"reason": "disk below 80GiB"})
        canaries = [read(root / f"shard-{item['index']:03d}/worker.json") for item in manifest["tasks"] if item["canary"]]
        if not gate_open and all(row.get("steps", 0) >= 2 and row.get("snapshots", 0) >= 1 for row in canaries):
            gate_open = True
            print("ADMISSION_OPEN", manifest["max_workers"], flush=True)
        if not gate_open and (any(row.get("status") == "held" for row in canaries)
                              or time.monotonic() - started > 5400):
            save(root / "STOP_REQUEST.json", {"reason": "canary did not establish actor/checkpoint evidence"})
        requested = read(root / "STOP_REQUEST.json")
        if requested and not stopping:
            stopping = True
            for entry in active.values():
                entry["process"].send_signal(signal.SIGINT)
        if not stopping and not failures:
            limit = manifest["max_workers"] if gate_open else len(manifest["canaries"])
            while pending and len(active) < limit:
                item = pending.pop(0)
                log = (root / f"shard-{item['index']:03d}.log").open("x")
                process = subprocess.Popen([sys.executable, "-u", "-m", "swebench_pro.worker", str(root), str(item["index"])],
                                           cwd=root / "source", stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                active[item["index"]] = {"process": process, "log": log, "task": item["id"]}
                print("LAUNCH", item["index"], item["id"], process.pid, flush=True)
        state = summary(root, manifest)
        state.update(pid=os.getpid(), active=[{"index": index, "pid": entry["process"].pid, "task": entry["task"]}
                                             for index, entry in active.items()],
                     queued=len(pending), canary_gate_open=gate_open,
                     status="stopping" if stopping and active else "paused" if stopping else "running" if pending or active else "completed")
        save(root / "controller.json", state)
        if stopping and not active:
            return 2
        time.sleep(10)
    save(root / "cleanup.json", {"at": now(), "remaining_sandboxes": [row["sandboxID"] for row in api("/sandboxes")]})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["launch", "controller", "summarize"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pro-repo", type=Path)
    parser.add_argument("--runtime-bin", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--agent-network", choices=["allow", "deny"], default=None)
    parser.add_argument("--verifier-network", choices=["allow", "deny"], default=None)
    args = parser.parse_args()
    if args.phase != "launch" and (args.agent_network is not None or args.verifier_network is not None):
        parser.error("Network flags apply to launch; existing controllers use their frozen manifest")
    root = args.out.resolve()
    if args.phase == "summarize":
        print(json.dumps(summary(root, read(root / "manifest.json"))), flush=True)
        return 0
    if args.phase == "controller":
        return controller(root)
    if args.pro_repo is None or args.runtime_bin is None or args.workers < 1:
        parser.error("launch requires --pro-repo, --runtime-bin and positive workers")
    prepare(root, args.pro_repo.resolve(), args.runtime_bin.resolve(), args.workers, args.model,
            agent_network=args.agent_network, verifier_network=args.verifier_network)
    environment = dict(os.environ, PYTHONPATH=f"{root / 'source'}:{root / 'source/sdk'}")
    with (root / "controller.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "swebench_pro.batch", "controller", "--out", str(root)],
                                   cwd=root / "source", env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps({"root": str(root), "controller_pid": process.pid, "tasks": 731, "workers": args.workers}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
