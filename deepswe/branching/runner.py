"""Paired DeepSWE sampling through Ash's existing orchestrator and grader."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from .policies import SHEPHERD_SYSTEM, bpo_select, shepherd_select
from .provider import ChatClient, candidates_from_audit, score_candidate
from .storage import fingerprint, load, rows, save

CONTINUE = "Continue the original task from this state. Complete the implementation, validate it, and commit your work."


@dataclass(frozen=True)
class Config:
    model: str
    tasks_dir: str
    output: str
    runtime_bin: str
    bridge_url: str = "http://127.0.0.1:18187"
    methods: tuple[str, ...] = ("baseline", "bpo", "shepherd")
    tasks: tuple[str, ...] = ()
    max_rollouts: int = 8
    effort: str = "high"
    initial_root: str | None = None
    provider_audit_root: str | None = None
    initial_owner_template: str = "experiment:{task}/initial/parent"
    task_locks: str | None = None
    bpo_top_k: int = 5
    bpo_min_spacing: int = 64
    bpo_max_points: int = 7
    bpo_scoring_workers: int = 2
    meta_max_tokens: int = 65536
    meta_transcript_tokens: int = 60000
    meta_extra: dict | None = None
    timeout: float | None = None
    api_timeout_ms: int = 1860000

    def __post_init__(self):
        if not self.model or not 1 <= self.max_rollouts <= 8:
            raise ValueError("Specify a model and max_rollouts between 1 and 8")
        if not self.methods or set(self.methods) - {"baseline", "bpo", "shepherd"}:
            raise ValueError("Methods must be baseline, bpo, or shepherd")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Duplicate method")
        if (not 1 <= self.bpo_top_k <= 20 or self.bpo_max_points < 1
                or self.bpo_min_spacing < 0 or self.bpo_scoring_workers < 1):
            raise ValueError("Invalid BPO settings")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if type(self.api_timeout_ms) is not int or not 1 <= self.api_timeout_ms <= 2147483647:
            raise ValueError("API timeout must be an integer between 1 and 2147483647 milliseconds")
        if self.meta_extra and set(self.meta_extra) & {"model", "messages", "stream", "reasoning_effort"}:
            raise ValueError("Meta extras override protected settings")


def result_resolved(result: dict) -> bool:
    grade = result["grade"]
    if grade.get("error") or grade.get("verifier_artifact_error"):
        raise RuntimeError("Verifier infrastructure failure; result is not a measured failure")
    return bool(grade["resolved"])


def actor_environment(config: Config, owner: str) -> dict[str, str]:
    # Total API and semantic-event watchdogs are independent. The buffered
    # bridge emits keepalive pings, not fake model progress, while reasoning.
    return {"ANTHROPIC_API_KEY": owner, "ANTHROPIC_BASE_URL": config.bridge_url,
            "CLAUDE_CODE_USE_BEDROCK": "0", "CLAUDE_CODE_USE_VERTEX": "0",
            "API_TIMEOUT_MS": str(config.api_timeout_ms),
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": str(config.api_timeout_ms),
            "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": str(min(config.api_timeout_ms, 1800000))}


def select_plan(method: str, *, scores: list[dict] | None, proposal: dict | None,
                eligible: list[int], config: Config) -> list[dict]:
    budget = config.max_rollouts - 1
    if method == "bpo":
        return bpo_select(scores or [], budget, config.bpo_min_spacing, config.bpo_max_points)
    if method == "shepherd":
        return shepherd_select(proposal or {}, eligible, budget)
    return [{} for _ in range(budget)]


def dataset_hash(task) -> str:
    # Includes verifier and oracle hashes for provenance, never sends their
    # contents to the worker or Shepherd selector.
    digest = hashlib.sha256()
    for path in sorted(task.task_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(task.task_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def code_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=Path(__file__).resolve().parents[2], text=True).strip()


def code_fingerprint() -> str:
    """Also identifies an uncommitted validation checkout, not only its base ref."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in ("harness", "swebench", "deepswe", "sdk/ash_sandbox"):
        for path in sorted((root / directory).rglob("*.py")):
            if "tests" not in path.parts:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


class BenchmarkRunner:
    def __init__(self, config: Config):
        from deepswe.bench import DeepSWE
        from swebench import fork_eval
        self.config = config
        self.root = Path(config.output).resolve()
        self.bench = DeepSWE(config.tasks_dir)
        self.ev = fork_eval
        self.client = ChatClient(timeout=1800)

    def args(self, task) -> SimpleNamespace:
        return SimpleNamespace(slot="claude-code", model=self.config.model,
                               runtime_bin=self.config.runtime_bin,
                               timeout=self.config.timeout or task.agent_timeout_s,
                               agent_network=None, verifier_network=None,
                               setting_sources=[])

    def run_attempt(self, task, method: str, name: str, **branch) -> dict:
        from harness.orchestrator.run import Orchestrator
        folder = self.root / method / task.task_id
        folder.mkdir(parents=True, exist_ok=True)
        journal = folder / (name + ".jsonl")
        result_path = folder / (name + ".result.json")
        if result_path.exists():
            result = load(result_path)
            if result.get("journal_sha256") != hashlib.sha256(journal.read_bytes()).hexdigest():
                raise ValueError("Completed journal changed after verification")
            result_resolved(result)
            return result
        instance = self.bench.instance(task)
        instance.update(slot="claude-code", agent_network="deny")
        args = self.args(task)
        started = time.time()
        if journal.exists():
            # Recover grading of a FINISHED attempt, never silently start over.
            if not any(row.get("type") == "run.finished" for row in rows(journal)):
                raise RuntimeError("Interrupted rollout requires explicit recovery: " + str(journal))
            outcome = self.ev.outcome_from_journal(journal, run_id=name)
        else:
            os.environ.update(actor_environment(
                self.config, "branchbench:%s/%s/%s" % (task.task_id, method, name)))
            cwd = folder / "actor-workspaces" / name
            cwd.mkdir(parents=True, exist_ok=True)
            kwargs = dict(name=name, prompt=self.bench.prompt(instance), image=task.image,
                          out_dir=folder, resources=self.bench.resources(instance),
                          bench=self.bench, cwd=cwd)
            kwargs.update(branch)
            outcome = self.ev.run_attempt(Orchestrator(out_dir=folder), args, instance, **kwargs)
        if outcome.status == "error":
            raise RuntimeError("Actor infrastructure error; journal retained: " + str(journal))
        grade = self.ev.grade_attempt(outcome, instance, args, self.bench)
        result = {"task": task.task_id, "method": method, "name": name,
                  "journal": str(journal), "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                  "status": outcome.status, "grade": asdict(grade),
                  "finished_at": time.time(), "seconds_this_invocation": time.time() - started}
        # A grading outage can be retried from the finished journal. Do not cache
        # it as an ordinary measured result or turn it into a failed rollout.
        if grade.error or grade.verifier_artifact_error:
            save(folder / (name + ".grading-error.json"), result)
            result_resolved(result)
        save(result_path, result)
        return result

    def initial(self, task) -> tuple[dict, Path]:
        if self.config.initial_root:
            journal = Path(self.config.initial_root).resolve() / task.task_id / "parent.jsonl"
            record = load(journal.with_suffix(".result.json"))
            result_resolved(record)
            events = rows(journal)
            if record.get("status") == "error":
                raise ValueError("Shared parent is an actor infrastructure error")
            models = {e.get("model") for e in events if e.get("type") == "run.started"}
            if models != {self.config.model}:
                raise ValueError("Shared parent model does not match comparison model")
            if not any(e.get("type") == "run.finished" for e in events):
                raise ValueError("Shared parent has not finished")
            return record, journal
        record = self.run_attempt(task, "initial", "parent")
        return record, Path(record["journal"])

    def plan(self, task, method: str, journal: Path, initial: dict) -> tuple[dict, dict]:
        points = self.ev.available_branch_points(journal)
        folder = self.root / method / task.task_id
        identity = fingerprint({"config": asdict(self.config),
                                "journal": hashlib.sha256(journal.read_bytes()).hexdigest(),
                                "initial_grade": initial["grade"], "method": method})
        path = folder / "plan.json"
        if path.exists():
            plan = load(path)
            if plan["identity"] != identity:
                raise ValueError("Plan configuration/root differs; use a new output directory")
            if any(choice["step"] not in points for choice in plan["selected"]):
                raise ValueError("Previously selected exact checkpoint is no longer available")
            return plan, points
        if not points:
            raise ValueError("No exact snapshot/session pairs available for branching")
        scores, proposal = None, None
        if method == "bpo":
            audit_root = Path(self.config.provider_audit_root or self.root)
            owner = (self.config.initial_owner_template.format(task=task.task_id)
                     if self.config.initial_root else "branchbench:%s/initial/parent" % task.task_id)
            candidates = candidates_from_audit(audit_root, owner, points, self.config.model)
            def score(candidate):
                return score_candidate(candidate, self.client, folder / "entropy",
                                       top_k=self.config.bpo_top_k)
            with ThreadPoolExecutor(max_workers=self.config.bpo_scoring_workers) as pool:
                scores = list(pool.map(score, candidates))
        else:
            transcript, lo, hi = self.ev.render_transcript(
                journal, token_budget=self.config.meta_transcript_tokens)
            payload = {"model": self.config.model, "reasoning_effort": self.config.effort,
                       "temperature": 1., "top_p": .95, "stream": False,
                       "max_tokens": self.config.meta_max_tokens,
                       "messages": [{"role": "system", "content": SHEPHERD_SYSTEM}, {
                           "role": "user", "content": json.dumps({
                               "task": task.instruction, "reward": int(result_resolved(initial)),
                               "eligible_checkpoint_steps": sorted(points),
                               "transcript_visible_step_range": [lo, hi], "trajectory": transcript})}]}
            payload.update(self.config.meta_extra or {})
            audit = folder / "meta-request.json"
            if audit.exists() and load(audit).get("request_sha256") == fingerprint(payload) and load(audit).get("response"):
                response = load(audit)["response"]
            else:
                response = self.client.complete(payload, audit)
            if response["choices"][0]["finish_reason"] != "stop":
                raise ValueError("Incomplete Shepherd decision; refusing a partial plan")
            proposal = self.ev.extract_json(response["choices"][0]["message"]["content"])
        selected = select_plan(method, scores=scores, proposal=proposal,
                               eligible=sorted(points), config=self.config)
        plan = {"identity": identity, "method": method, "training": False,
                "shared_initial": str(journal), "eligible_steps": sorted(points),
                "selected": selected, "proposal": proposal,
                "max_total_rollouts": self.config.max_rollouts,
                "branch_index_semantics": "state after completed tool step",
                "hint_delivery": "fixed-neutral", "created_at": time.time()}
        if method == "bpo":
            plan["entropy_note"] = ("Top-k plus tail lower bound of first reported content token; "
                                    "hidden reasoning-token entropy is not observable via this API.")
        save(path, plan)
        return plan, points

    def branch(self, task, method: str, index: int, choice: dict, checkpoint, journal: Path) -> dict:
        name = "b%02d-step%d" % (index, checkpoint.step)
        folder = self.root / method / task.task_id
        if (folder / (name + ".jsonl")).exists():
            return self.run_attempt(task, method, name)
        restoration = self.ev.conversation_restore(journal, checkpoint.step, checkpoint.session_ckpt)
        if restoration is None:
            raise ValueError("Exact conversation restore is unavailable")
        cut, source = restoration
        source = source or self.ev.find_prefix_source(self.ev.CLAUDE_PROJECTS_DIR,
                                                     checkpoint.session_ckpt, cut)
        if source is None:
            raise ValueError("Cannot materialize the exact original transcript prefix")
        receipt = folder / "conversation-prefixes" / name
        if receipt.exists():
            # A prepared-but-not-launched branch may be resumed only while the
            # saved native prefix remains byte-identical to its receipt.
            prepared = load(receipt / "manifest.json")
            if (prepared["cut"] != cut or prepared["source_sha256"] != source.sha256
                    or hashlib.sha256(Path(prepared["native_path"]).read_bytes()).hexdigest()
                    != prepared["prefix_sha256"]):
                raise ValueError("Prepared prefix changed; refusing ambiguous recovery")
            prepared["manifest_path"] = str(receipt / "manifest.json")
        else:
            prepared = self.ev.prepare_prefix(source, folder / "actor-workspaces" / name,
                                               receipt, self.ev.CLAUDE_PROJECTS_DIR)
        origin = {"parent_run_id": "parent", "parent_journal": str(journal),
                  "branch_step": checkpoint.step, "snapshot_id": checkpoint.snapshot_id,
                  "conversation_cut": cut, "conversation_restore": "original-prefix",
                  "conversation_prefix_manifest": prepared["manifest_path"],
                  "branch_policy": method, "selection": choice, "actor_hint": CONTINUE,
                  "hint_delivery": "fixed-neutral"}
        return self.run_attempt(task, method, name, prompt=CONTINUE, image=checkpoint.snapshot_id,
                                resume=prepared["resume_session_id"], fork=True, resume_at=None,
                                cwd=Path(prepared["cwd"]), origin=origin)

    def run_task(self, task) -> dict:
        initial, journal = self.initial(task)
        parent_identity = {"journal": str(journal),
                           "sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                           "grade_sha256": fingerprint(initial["grade"])}
        identity_path = self.root / "parents" / (task.task_id + ".json")
        if identity_path.exists() and load(identity_path) != parent_identity:
            raise ValueError("Shared parent changed since the comparison began")
        save(identity_path, parent_identity)
        summary = {"task": task.task_id, "initial_resolved": result_resolved(initial), "methods": {}}
        for method in self.config.methods:
            if summary["initial_resolved"] or self.config.max_rollouts == 1:
                summary["methods"][method] = {"status": "skipped", "extra_rollouts": 0,
                                               "resolved": summary["initial_resolved"]}
                continue
            try:
                attempts = []
                plan, points = (None, None) if method == "baseline" else self.plan(task, method, journal, initial)
                choices = plan["selected"] if plan else [{}] * (self.config.max_rollouts - 1)
                for index, choice in enumerate(choices, 1):
                    if method == "baseline":
                        record = self.run_attempt(task, method, "b%02d" % index)
                    else:
                        record = self.branch(task, method, index, choice, points[choice["step"]], journal)
                    attempts.append(record)
                    if result_resolved(record):
                        break
                summary["methods"][method] = {
                    "status": "done", "extra_rollouts": len(attempts),
                    "resolved": any(result_resolved(a) for a in attempts),
                    "attempts": [{"name": a["name"], "resolved": result_resolved(a)} for a in attempts]}
            except Exception as exc:
                summary["methods"][method] = {"status": "blocked", "error": str(exc),
                                               "error_type": type(exc).__name__}
            save(self.root / "task-summary" / (task.task_id + ".json"), summary)
        save(self.root / "task-summary" / (task.task_id + ".json"), summary)
        return summary

    def run(self) -> dict:
        from filelock import FileLock
        import httpx
        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root / "runner.lock"), timeout=0):
            health = httpx.get(self.config.bridge_url.rstrip("/") + "/health", timeout=10).json()
            if health["model"] != self.config.model or health["effort"] != self.config.effort:
                raise ValueError("Bridge model/reasoning settings differ from benchmark config")
            if Path(health["audit_root"]).resolve() != self.root:
                raise ValueError("Bridge audit directory must equal this run's output directory")
            catalogue = self.bench.catalogue(None)
            ids = list(self.config.tasks or sorted(catalogue))
            if len(set(ids)) != len(ids) or any(task not in catalogue for task in ids):
                raise ValueError("Invalid fixed task list")
            locks = load(Path(self.config.task_locks)) if self.config.task_locks else {}
            tasks = [replace(catalogue[tid], image=locks[tid]["pinned_image"]) if locks else catalogue[tid]
                     for tid in ids]
            manifest = {"config": asdict(self.config), "tasks": ids, "ash_commit": code_revision(),
                        "source_sha256": code_fingerprint(),
                        "runtime_sha256": hashlib.sha256(Path(self.config.runtime_bin).read_bytes()).hexdigest(),
                        "dataset_sha256": {t.task_id: dataset_hash(t) for t in tasks},
                        "images": {t.task_id: t.image for t in tasks},
                        "budget_includes_shared_initial": True, "stop_rule": "first verified success",
                        "agentenv_release": "v0.1.2-ash.1", "training": False}
            path = self.root / "benchmark-manifest.json"
            if path.exists() and load(path) != manifest:
                raise ValueError("Run manifest changed; choose a new output directory")
            save(path, manifest)
            summaries = []
            for task in tasks:
                print("Sampling", task.task_id, flush=True)
                try:
                    summaries.append(self.run_task(task))
                except Exception as exc:
                    summary = {"task": task.task_id, "status": "blocked", "error": str(exc)}
                    save(self.root / "task-summary" / (task.task_id + ".json"), summary)
                    summaries.append(summary)
                save(self.root / "summary.json", {"tasks": summaries})
            return {"tasks": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    data = load(args.config)
    for key in ("tasks", "methods"):
        if key in data:
            data[key] = tuple(data[key])
    result = BenchmarkRunner(Config(**data)).run()
    print(json.dumps(result, indent=2))
    return int(any(t.get("status") == "blocked" or any(m["status"] == "blocked"
               for m in t.get("methods", {}).values()) for t in result["tasks"]))


if __name__ == "__main__":
    raise SystemExit(main())
