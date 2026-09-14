"""Durable, fenced infrastructure retries for independent Pro worker jobs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

from swebench_pro.recovery import durable_json, read_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def retry_payload(root: Path, manifest: dict, item: dict, attempt: Path, state: dict) -> dict:
    from harness.rollback import turn_branch_checkpoints
    from harness.slots.claude_history import find_prefix_source
    from swebench.fork_eval import CLAUDE_PROJECTS_DIR, conversation_restore
    from swebench_pro.resume import disk_manifest_valid

    directory = attempt / item["id"]
    journal = directory / "parent.jsonl"
    execution_path = directory / "execution.json"
    execution = read_json(execution_path) or {}
    events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    captures = [event for event in events if event.get("type") == "checkpoint.captured"
                and event.get("snapshot_id") and event.get("reason", "captured") == "captured"
                and event.get("captured") is not False]
    grading_only = state.get("retry_kind") == "grading" or state.get("phase") in ("grading", "waiting_for_grader")
    if execution and state.get("retry_kind") != "actor" and (execution.get("status") == "completed" or grading_only):
        if not captures or not disk_manifest_valid(captures[-1]["snapshot_id"]):
            raise ValueError("Completed actor has no intact final grading snapshot")
        return {"continuation": None, "recovery_action": "regrade", "regrade": {
            "source_execution": str(execution_path), "execution_sha256": digest(execution_path),
            "source_journal": str(journal), "journal_sha256": digest(journal),
            "snapshot_id": captures[-1]["snapshot_id"]}}
    if state.get("budget_exhausted") or execution.get("status") in ("timeout", "step_limit", "cost_limit"):
        raise ValueError("Actor budget exhausted; no additional solution attempt")
    previous = item.get("continuation") or {}
    started = next((event["ts"] for event in events if event.get("type") == "run.started"), None)
    finished = state.get("actor_finished_at") or state.get("finished_at")
    if not started or not finished:
        raise ValueError("Cannot verify consumed actor time")
    elapsed = max(0.0, (instant(finished) - instant(started)).total_seconds())
    charged = previous.get("actor_seconds_charged", 0) + elapsed
    remaining = manifest["actor_timeout_s"] - charged
    budgets = [event for event in events if event.get("type") == "pro.tool_budget"]
    reported = [event for event in budgets if "tool_seconds" in event]
    tool_seconds = max(previous.get("tool_seconds_charged", 0),
                       reported[-1]["tool_seconds"] if reported else 0)
    consecutive = reported[-1].get("consecutive_timeouts", 0) if reported else previous.get("consecutive_timeouts", 0)
    if budgets and "effective_timeout" in budgets[-1]:
        tool_seconds += max(0.0, (instant(finished) - instant(budgets[-1]["ts"])).total_seconds())
    if remaining <= 0 or tool_seconds >= manifest["total_tool_seconds"] or consecutive >= manifest["consecutive_timeouts"]:
        raise ValueError("Actor/tool budget exhausted; recovery cannot reset it")
    points = turn_branch_checkpoints(journal)
    for step, point in sorted(points.items(), reverse=True):
        if not disk_manifest_valid(point.snapshot_id):
            continue
        restored = conversation_restore(journal, step, point.session_ckpt)
        if not restored:
            continue
        cut, prefix = restored
        prefix = prefix or find_prefix_source(CLAUDE_PROJECTS_DIR, point.session_ckpt, cut)
        if prefix is None:
            continue
        return {"regrade": None, "recovery_action": "resume", "continuation": {
            "snapshot_id": point.snapshot_id, "step": step, "cut": cut,
            "source_session": point.session_ckpt, "source_journal": str(journal),
            "journal_sha256": digest(journal), "transcript_sha256": prefix.sha256,
            "source_transcript": str(prefix.transcript), "actor_seconds_remaining": remaining,
            "actor_seconds_charged": charged, "tool_seconds_charged": tool_seconds,
            "consecutive_timeouts": consecutive, "latest_recorded_step": max(points)}}
    if previous:
        if not disk_manifest_valid(previous["snapshot_id"]) or digest(Path(previous["source_journal"])) != previous["journal_sha256"]:
            raise ValueError("Original recovery checkpoint changed")
        prefix = find_prefix_source(CLAUDE_PROJECTS_DIR, previous["source_session"], previous["cut"])
        if prefix is not None and prefix.sha256 == previous["transcript_sha256"]:
            return {"regrade": None, "recovery_action": "resume", "continuation": {
                **previous, "actor_seconds_remaining": remaining, "actor_seconds_charged": charged,
                "tool_seconds_charged": tool_seconds, "consecutive_timeouts": consecutive}}
    raise ValueError("No verified checkpoint/native prefix; refusing a fresh parent retry")


class RetryQueue:
    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.manifest = manifest
        self.maximum = manifest.get("max_infra_retries", 2)
        self.backoffs = manifest.get("retry_backoff_s", [30, 60])
        if type(self.maximum) is not int or not 0 <= self.maximum <= 2:
            raise ValueError("Infrastructure retries must be between zero and two")
        if len(self.backoffs) < self.maximum or any(not isinstance(value, (int, float)) or
                                                  not math.isfinite(value) or value < 0 for value in self.backoffs):
            raise ValueError("Retry backoffs must be finite nonnegative delays")
        self.items = {item["index"]: item for item in manifest["tasks"]}
        self.jobs = {}
        for index, item in self.items.items():
            path = self.job_root(index) / "job.json"
            job = read_json(path)
            if job is None:
                prior = read_json(self.job_root(index) / "worker.json")
                if prior and not (prior.get("finished_at") and prior.get("evidence_valid")):
                    raise ValueError("Reconcile legacy unfinished shards before isolated recovery")
                job = {"task": item["id"], "phase": "completed" if prior else "queued",
                       "history": [], "overrides": {}, "not_before": 0, "active_attempt_id": None}
                durable_json(path, job)
            self.jobs[index] = job
            if job["task"] != item["id"]:
                raise ValueError("Persisted job does not match the frozen task")
            abandoned = self.job_root(index) / "attempts" / f"attempt-{len(job['history']):03d}"
            if job["phase"] in ("queued", "retry_wait") and abandoned.exists():
                job.update(phase="quarantined", publication={
                    "task": item["id"], "status": "quarantined", "evidence_valid": False,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": "Uncommitted attempt directory requires reconciliation"})
                self.persist(index)
            if job.get("publication"):
                self.publish(index, job["publication"])

    def job_root(self, index: int) -> Path:
        return self.root / f"shard-{index:03d}"

    def persist(self, index: int) -> None:
        durable_json(self.job_root(index) / "job.json", self.jobs[index])

    def ready(self, now: float) -> list[int]:
        return [index for index, job in self.jobs.items()
                if job["phase"] in ("queued", "retry_wait") and job["not_before"] <= now]

    def claim(self, index: int, *, now: float | None = None) -> Path:
        job = self.jobs[index]
        if job["phase"] not in ("queued", "retry_wait"):
            raise ValueError("Job already leased or terminal")
        if job["not_before"] > (time.time() if now is None else now):
            raise ValueError("Retry backoff has not elapsed")
        number = len(job["history"])
        if number > self.maximum:
            raise ValueError("Infrastructure retry cap exhausted")
        directory = self.job_root(index) / "attempts" / f"attempt-{number:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        request = {"attempt_id": uuid4().hex, "attempt_number": number, "index": index,
                   "task": self.items[index]["id"], "overrides": job["overrides"]}
        path = directory / "request.json"
        durable_json(path, request)
        job.update(phase="running", active_attempt_id=request["attempt_id"],
                   attempt_directory=str(directory), request_sha256=digest(path), pid=None, process_start=None)
        job.pop("publication", None)
        self.persist(index)
        self.publish(index, {"task": request["task"], "status": "running", "evidence_valid": False,
                             "attempt_id": request["attempt_id"], "retry_count": number})
        return path

    def publish(self, index: int, state: dict) -> None:
        job = self.jobs[index]
        state = {**state, "attempt_directory": job.get("attempt_directory"),
                 "retry_history": job["history"],
                 "retry_count": len(job["history"]) if job["phase"] == "running" else max(0, len(job["history"]) - 1)}
        if job.get("attempt_directory"):
            state["journal_path"] = str(Path(job["attempt_directory"]) / job["task"] / "parent.jsonl")
        usage = {}
        complete = True
        attempts = list(job["history"])
        if job["phase"] == "running" and job.get("attempt_directory"):
            request = read_json(Path(job["attempt_directory"]) / "request.json")
            current = {**self.items[index], **request["overrides"]}
            attempts.append({"mode": "regrade" if current.get("regrade") else "actor", "usage": state.get("usage", {})})
        for attempt in attempts:
            if attempt["mode"] == "regrade":
                continue
            values = attempt.get("usage") or {}
            complete = complete and bool(values)
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        state.update(usage=usage, reported_usage_complete=complete)
        durable_json(self.job_root(index) / "worker.json", state)

    def finish(self, index: int, attempt_id: str, state: dict, *, safe: bool,
               now: float, planner=None) -> bool:
        job = self.jobs[index]
        if job["phase"] != "running" or job["active_attempt_id"] != attempt_id:
            return False
        planner = planner or retry_payload
        if state.get("attempt_id") not in (None, attempt_id):
            safe = False
        directory = Path(job["attempt_directory"])
        request = read_json(directory / "request.json")
        item = {**self.items[index], **request["overrides"]}
        job["history"].append({"attempt_id": attempt_id, "directory": str(directory),
                               "mode": "regrade" if item.get("regrade") else "actor",
                               "evidence_valid": bool(state.get("evidence_valid")),
                               "error": state.get("error") or state.get("grading_error"),
                               "usage": state.get("usage", {}), "cleanup_confirmed": safe})
        state = {**state, "task": job["task"], "attempt_id": attempt_id,
                 "finished_at": state.get("finished_at") or datetime.now(timezone.utc).isoformat()}
        if not safe:
            job["phase"] = "quarantined"
            state.update(status="quarantined", evidence_valid=False, retry_blocked_reason="Old worker/resources not confirmed stopped")
        elif state.get("evidence_valid"):
            job["phase"] = "completed"
            state["status"] = "completed"
        elif not state.get("retryable") or len(job["history"]) > self.maximum:
            job["phase"] = "infra_failed"
            state.update(status="infra_failed", evidence_valid=False,
                         retry_blocked_reason="Retry cap exhausted" if state.get("retryable") else "Failure is not retryable")
        else:
            try:
                overrides = planner(self.root, self.manifest, item, directory, state)
            except Exception as error:
                job["phase"] = "infra_failed"
                state.update(status="infra_failed", evidence_valid=False, retry_blocked_reason=str(error))
            else:
                job.update(phase="retry_wait", overrides=overrides,
                           not_before=now + self.backoffs[len(job["history"]) - 1])
                state.update(status="retry_wait", finished_at=None, evidence_valid=False,
                             retry_not_before=job["not_before"], next_recovery_action=overrides["recovery_action"])
        job["publication"] = state
        self.persist(index)
        self.publish(index, state)
        return True


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def group_alive(pid: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pid:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def controller(root: Path, *, api=None, delete=None, popen=None,
               clock=time.time, sleep=time.sleep, process_group_alive=group_alive) -> int:
    from swebench_pro.batch import summary
    from swebench_pro.worker import api as default_api, delete_sandbox

    api = api or default_api
    delete = delete or delete_sandbox
    popen = popen or subprocess.Popen
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read_json(root / "manifest.json")
    queue = RetryQueue(root, manifest)
    active = {}
    stopping = False
    canaries = [index for index, item in queue.items.items() if item.get("canary")]

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for index, job in queue.jobs.items():
        if job["phase"] == "running":
            if job.get("pid") and job.get("process_start"):
                active[index] = {"process": None, "pid": job["pid"], "start": job["process_start"], "log": None}
            else:
                queue.finish(index, job["active_attempt_id"], {"error": "Controller stopped during worker dispatch"}, safe=False, now=clock())
    while True:
        pause = None
        try:
            inventory = api("/v2/sandboxes?limit=100")
            if len(inventory) >= 100:
                raise RuntimeError("Sandbox inventory may be paginated; cleanup cannot be proven")
        except Exception as error:
            inventory = None
            pause = f"AgentENV unavailable: {error}"
        if shutil.disk_usage(root).free < 80 * 1024**3:
            pause = "Disk below 80GiB; admission paused"
        if (root / "STOP_REQUEST.json").exists():
            stopping = True
        for index, entry in list(active.items()):
            process = entry["process"]
            running = process.poll() is None if process is not None else process_identity(entry["pid"]) == entry["start"]
            job = queue.jobs[index]
            directory = Path(job["attempt_directory"])
            state = read_json(directory / "worker.json") or {}
            if state.get("attempt_id") == job["active_attempt_id"]:
                if state.get("steps", 0) >= 2 and state.get("snapshots", 0) >= 1:
                    job["canary_ready"] = True
                    queue.persist(index)
                if running:
                    queue.publish(index, {**state, "status": "running", "evidence_valid": False,
                                          "resolved": None, "finished_at": None})
            if running or inventory is None:
                continue
            owned = (read_json(directory / "sandboxes.json") or {}).get("owned", [])
            safe = bool(state) and state.get("attempt_id") == job["active_attempt_id"]
            safe = safe and state.get("creation_inflight") == 0 and state.get("creation_uncertain") is False
            safe = safe and not process_group_alive(entry["pid"])
            if safe:
                try:
                    for sandbox in inventory:
                        if sandbox["sandboxID"] in owned:
                            delete(sandbox["sandboxID"])
                    current = api("/v2/sandboxes?limit=100")
                    safe = len(current) < 100 and not ({row["sandboxID"] for row in current} & set(owned))
                except Exception:
                    safe = False
            if not state.get("finished_at"):
                state.update(error="Worker exited without a final result", retryable=True,
                             finished_at=datetime.now(timezone.utc).isoformat())
            queue.finish(index, job["active_attempt_id"], state, safe=safe, now=clock())
            if job["phase"] == "completed":
                job["canary_ready"] = True
                queue.persist(index)
            if entry["log"] is not None:
                entry["log"].close()
            del active[index]
        quarantined_slots = sum(job["phase"] == "quarantined" for job in queue.jobs.values())
        terminal = {"completed", "infra_failed", "quarantined"}
        gate_open = not canaries or (any(queue.jobs[index].get("canary_ready") for index in canaries)
                                    and all(queue.jobs[index].get("canary_ready") or queue.jobs[index]["phase"] in terminal
                                            for index in canaries))
        gate_failed = bool(canaries) and not gate_open and all(queue.jobs[index]["phase"] in terminal for index in canaries)
        if gate_failed:
            pause = "No canary established runtime readiness; admission paused"
        if not stopping and pause is None:
            for index in queue.ready(clock()):
                if not gate_open and index not in canaries:
                    continue
                if len(active) + quarantined_slots >= manifest["max_workers"]:
                    break
                request = queue.claim(index, now=clock())
                log = (request.parent / "worker.log").open("x")
                try:
                    process = popen([sys.executable, "-u", "-m", "swebench_pro.worker", str(root), str(index), str(request)],
                                    cwd=root / "source", env=dict(os.environ, PYTHONPATH=f"{root / 'source'}:{root / 'source/sdk'}"),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except Exception as error:
                    log.close()
                    queue.finish(index, queue.jobs[index]["active_attempt_id"], {"error": str(error)}, safe=True, now=clock())
                    continue
                job = queue.jobs[index]
                job.update(pid=process.pid, process_start=process_identity(process.pid))
                queue.persist(index)
                active[index] = {"process": process, "pid": process.pid, "start": job["process_start"], "log": log}
        pending = any(job["phase"] in ("queued", "running", "retry_wait") for job in queue.jobs.values())
        result = summary(root, manifest)
        result.update(pid=os.getpid(), failure_policy="isolated", shared_pause=pause,
                      canary_gate_open=gate_open,
                      quarantined_slots=quarantined_slots,
                      active=[{"index": index, "pid": entry["pid"]} for index, entry in active.items()],
                      job_phases=dict(Counter(job["phase"] for job in queue.jobs.values())),
                      infrastructure_retries=sum(max(0, len(job["history"]) + (job["phase"] == "running") - 1)
                                                 for job in queue.jobs.values()),
                      status="draining" if stopping and active else "paused" if stopping or pause else
                             "running" if pending else "completed")
        durable_json(root / "controller.json", result)
        if stopping and not active:
            return 2
        if not pending and not active:
            return 0 if all(job["phase"] == "completed" for job in queue.jobs.values()) else 2
        if not active and quarantined_slots >= manifest["max_workers"]:
            return 2
        if gate_failed and not active:
            return 2
        sleep(5)
