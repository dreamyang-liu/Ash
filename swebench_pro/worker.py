"""One independent Pro attempt from a frozen batch manifest."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any

import httpx


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: Any) -> None:
    from swebench_pro.recovery import durable_json

    durable_json(path, value)


def api(route: str) -> Any:
    request = urllib.request.Request(os.environ["AENV_SERVER_URL"] + route,
                                     headers={"X-API-Key": os.environ["AENV_API_KEY"]})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def delete_sandbox(identifier: str) -> None:
    request = urllib.request.Request(os.environ["AENV_SERVER_URL"] + "/sandboxes/" + identifier,
                                     headers={"X-API-Key": os.environ["AENV_API_KEY"]}, method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=60):
            pass
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise


@contextmanager
def track_creations(state: dict, lock: Any, update: Any, own: Any):
    original_sync = httpx.Client.post
    original_async = httpx.AsyncClient.post

    @contextmanager
    def creation(url: Any):
        if urlparse(str(url)).path.rstrip("/") not in ("/sandboxes", "/sandboxes-cold"):
            yield None
            return
        with lock:
            update(creation_inflight=state["creation_inflight"] + 1)
        receipt = {}
        try:
            yield receipt
        finally:
            with lock:
                response = receipt.get("response")
                identifier = None
                if response is not None:
                    try:
                        identifier = response.json().get("sandboxID")
                    except (ValueError, AttributeError):
                        pass
                if identifier:
                    own(identifier)
                else:
                    state["creation_uncertain"] = True
                update(creation_inflight=state["creation_inflight"] - 1)

    def sync_post(client: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        with creation(url) as receipt:
            response = original_sync(client, url, *args, **kwargs)
            if receipt is not None:
                receipt["response"] = response
            return response

    async def async_post(client: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        with creation(url) as receipt:
            response = await original_async(client, url, *args, **kwargs)
            if receipt is not None:
                receipt["response"] = response
            return response

    httpx.Client.post = sync_post
    httpx.AsyncClient.post = async_post
    try:
        yield
    finally:
        httpx.Client.post = original_sync
        httpx.AsyncClient.post = original_async


def restore_completed_outcome(regrade: dict, journal_path: Path) -> Any:
    from harness.orchestrator.run import RunOutcome
    from swebench_pro.resume import disk_manifest_valid

    journal = Path(regrade["source_journal"]).read_bytes()
    execution = Path(regrade["source_execution"]).read_bytes()
    if hashlib.sha256(journal).hexdigest() != regrade["journal_sha256"]:
        raise RuntimeError("Original grading journal changed")
    if hashlib.sha256(execution).hexdigest() != regrade["execution_sha256"]:
        raise RuntimeError("Original actor outcome changed")
    events = [json.loads(line) for line in journal.splitlines() if line.strip()]
    captures = [event for event in events if event.get("type") == "checkpoint.captured"
                and event.get("snapshot_id") and (event.get("reason") or "captured") == "captured"
                and event.get("captured") is not False]
    if not captures or captures[-1]["snapshot_id"] != regrade["snapshot_id"]:
        raise RuntimeError("Regrade must use the original final snapshot")
    if not disk_manifest_valid(regrade["snapshot_id"]):
        raise RuntimeError("Original grading snapshot is not valid")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_bytes(journal)
    payload = json.loads(execution)
    payload["journal_path"] = journal_path
    return RunOutcome(**payload)


@contextmanager
def lane(root: Path, name: str, count: int):
    directory = root / "lanes"
    directory.mkdir(exist_ok=True)
    while True:
        for index in range(count):
            stream = (directory / f"{name}-{index}.lock").open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
                stream.close()
            return
        time.sleep(1)


def main() -> int:
    from harness.execution import interceptors
    from harness.orchestrator.run import Orchestrator
    from harness.slots import claude_code
    from swebench import fork_eval
    from swebench_pro import grade as grading
    from swebench_pro.bench import SWEbenchPro
    from swebench_pro.limits import OfficialToolBudget

    root = Path(sys.argv[1]).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    index = int(sys.argv[2])
    item = manifest["tasks"][index]
    isolated = manifest.get("failure_policy") == "isolated"
    attempt_id = None
    attempt_directory = None
    attempt_lock = None
    if isolated:
        from swebench_pro.retry_queue import digest

        if len(sys.argv) != 4:
            raise RuntimeError("Isolated workers require a controller-owned attempt request")
        request_path = Path(sys.argv[3]).resolve()
        job_root = root / f"shard-{index:03d}"
        if request_path.name != "request.json" or request_path.parent.parent != job_root / "attempts":
            raise RuntimeError("Attempt request is outside its job directory")
        request = json.loads(request_path.read_text())
        job = json.loads((job_root / "job.json").read_text())
        if (request["index"] != index or request["task"] != item["id"] or job["phase"] != "running"
                or job["active_attempt_id"] != request["attempt_id"] or job["request_sha256"] != digest(request_path)):
            raise RuntimeError("Attempt request lease is stale or modified")
        if set(request["overrides"]) - {"continuation", "regrade", "recovery_action"}:
            raise RuntimeError("Attempt request changes frozen job inputs")
        item = {**item, **request["overrides"]}
        attempt_id = request["attempt_id"]
        attempt_directory = request_path.parent
        attempt_lock = (attempt_directory / "worker.lock").open("a")
        fcntl.flock(attempt_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (attempt_directory / "worker.json").exists():
            raise RuntimeError("Attempt has already been executed")
    continuation = item.get("continuation")
    regrade = item.get("regrade")
    if continuation and regrade:
        raise RuntimeError("An attempt cannot both resume its actor and be regrade-only")
    if manifest.get("recovery"):
        from swebench_pro.resume import install_template_health, require_images_ready

        if not (root / "all-images-ready.json").exists():
            raise RuntimeError("Recovery has not passed the all-images-ready barrier")
        require_images_ready(root, manifest)
        install_template_health()
        if item.get("recovery_action") == "preserve":
            raise RuntimeError("Refusing to rerun a preserved completed task")
    shard = attempt_directory or root / f"shard-{item['index']:03d}"
    shard.mkdir(exist_ok=isolated)
    directory = shard / item["id"]
    owned = set()
    lock = threading.RLock()
    state = {"pid": os.getpid(), "task": item["id"], "status": "starting", "started_at": now(),
             "phase": "preflight", "steps": 0, "snapshots": 0, "evidence_valid": False,
             "network_requested": {phase: item.get(f"{phase}_network", manifest.get(f"{phase}_network"))
                                   for phase in ("agent", "verifier")},
             "recovery_action": item.get("recovery_action", "fresh")}
    if isolated:
        state.update(attempt_id=attempt_id, creation_inflight=0, creation_uncertain=False,
                     retryable=False, retry_kind=None)

    def update(**values: Any) -> None:
        with lock:
            state.update(values, updated_at=now())
            save(shard / "worker.json", state)

    def own(identifier: str) -> None:
        with lock:
            owned.add(identifier)
            save(shard / "sandboxes.json", {"owned": sorted(owned)})

    def stop_job(reason: str, record: dict | None = None, retryable: bool = False) -> None:
        if isolated:
            update(retryable=retryable, retry_kind="actor", local_stop_reason=reason,
                   execution_uncertain=bool(record and record.get("reason") == "execution_uncertain"))
        else:
            save(root / "STOP_REQUEST.json", {"reason": reason, "task": item["id"], "record": record})

    budget = OfficialToolBudget(manifest["tool_timeout_s"], manifest["total_tool_seconds"],
                                manifest["consecutive_timeouts"])
    if continuation:
        budget.elapsed = continuation["tool_seconds_charged"]
        budget.consecutive = continuation["consecutive_timeouts"]

    class BudgetedSlot(claude_code.ClaudeCodeSlot):
        async def _pre_tool_use(self, input_data: dict, tool_use_id: Any, context: Any = None) -> Any:
            reason = budget.exhausted
            if reason and self._control:
                self._control.request_stop(reason)
            return await super()._pre_tool_use(input_data, tool_use_id, context)

    class RecordedOrchestrator(Orchestrator):
        def _wire_checkpoints(self, spec: Any, journal: Any, provisioned: Any = None) -> Any:
            bridge = super()._wire_checkpoints(spec, journal, provisioned)
            if not bridge or not bridge.exact_mode or provisioned.tracker is not budget:
                raise RuntimeError("Pro budget/checkpoint wiring is missing")
            budget.emit = lambda **values: journal.emit("pro.tool_budget", **values)
            own(provisioned.session.sandbox_id)
            provisioned.session.on_swap.append(lambda sandbox: own(sandbox.sandbox_id))
            if continuation:
                journal.emit("checkpoint.captured", step=0, snapshot_id=continuation["snapshot_id"],
                             reason="captured", restored=True, source_step=continuation["step"])

            def observe(record: dict) -> None:
                kind = record.get("type")
                if kind == "tool.started":
                    update(steps=record.get("step", 0))
                elif kind == "checkpoint.captured" and record.get("reason") == "captured":
                    update(snapshots=state["snapshots"] + 1, last_snapshot=record.get("snapshot_id"))
                elif kind == "run.started":
                    config = record["config"]
                    if config.get("mcp_tool_timeout_ms") != "3600000":
                        stop_job("MCP timeout not propagated")
                    save(directory / "actor-config.json", config)
                elif kind == "model.turn.invalid" or (kind == "checkpoint.captured"
                                                      and record.get("reason") == "execution_uncertain"):
                    stop_job(kind, record, retryable=True)
            journal.subscribe(observe)
            return bridge

        def run(self, spec: Any) -> Any:
            if spec.run_id != "parent" or spec.resume_session_id or spec.fork or spec.origin:
                raise RuntimeError("Single-rollout batch attempted a resume or branch")
            if regrade:
                outcome = restore_completed_outcome(regrade, Path(spec.journal_path))
                save(directory / "regrade-origin.json", {**regrade, "actor_reexecuted": False})
                save(directory / "execution.json", {**asdict(outcome), "journal_path": str(outcome.journal_path)})
                update(phase="waiting_for_grader", agent_status=outcome.status,
                       agent_error=outcome.error, usage=outcome.usage)
                return outcome
            cwd = root / "actor-workspaces" / str(item["index"])
            if isolated:
                cwd = cwd / attempt_directory.name
            cwd.mkdir(parents=True, exist_ok=False)
            if manifest.get("recovery"):
                backend = {**spec.backend, "microvm": {**spec.backend["microvm"], "request_timeout": 900}}
                spec = replace(spec, backend=backend)
            if continuation:
                from harness.slots.claude_history import find_prefix_source, prepare_prefix
                from swebench.fork_eval import CLAUDE_PROJECTS_DIR

                if hashlib.sha256(Path(continuation["source_journal"]).read_bytes()).hexdigest() != continuation["journal_sha256"]:
                    raise RuntimeError("Original continuation journal changed")
                prefix = find_prefix_source(CLAUDE_PROJECTS_DIR, continuation["source_session"], continuation["cut"])
                if prefix is None or prefix.sha256 != continuation["transcript_sha256"]:
                    raise RuntimeError("Original native continuation prefix changed")
                prepared = prepare_prefix(prefix, cwd, directory / "conversation-prefix", CLAUDE_PROJECTS_DIR)
                origin = {"recovery": True, "source_journal": continuation["source_journal"],
                          "snapshot_id": continuation["snapshot_id"], "branch_step": continuation["step"],
                          "conversation_cut": continuation["cut"], "conversation_restore": "original-prefix",
                          "source_session_id": continuation["source_session"],
                          "actor_seconds_charged": continuation["actor_seconds_charged"],
                          "tool_seconds_charged": continuation["tool_seconds_charged"],
                          "prefix_manifest": prepared["manifest_path"], "analyst_calls": 0}
                if continuation["actor_seconds_remaining"] <= 0 or budget.exhausted:
                    from harness.core.journal import JournalWriter
                    from harness.orchestrator.run import RunOutcome

                    with JournalWriter(spec.journal_path, run_id="parent") as journal:
                        journal.emit("fork.origin", **origin)
                        journal.emit("checkpoint.captured", step=continuation["step"],
                                     snapshot_id=continuation["snapshot_id"], reason="captured")
                    outcome = RunOutcome(run_id="parent", journal_path=spec.journal_path, status="timeout",
                                         error="original rollout budget exhausted before recovery")
                    save(directory / "execution.json", {**asdict(outcome), "journal_path": str(outcome.journal_path)})
                    return outcome
                spec = replace(spec, prompt="Continue the original task from this saved checkpoint. "
                               "The previous process was interrupted. Continue implementing and testing the solution.",
                               sandbox_image=continuation["snapshot_id"], resume_session_id=prepared["resume_session_id"],
                               timeout_s=continuation["actor_seconds_remaining"], origin=origin,
                               extra={**spec.extra, "resume_session_at": continuation["cut"]})
            spec = replace(spec, cwd=str(cwd), extra={**spec.extra, "setting_sources": [],
                                                    "mcp_server_timeout_ms": 3600000})
            update(status="running", phase="actor")
            outcome = super().run(spec)
            save(directory / "execution.json", {**asdict(outcome), "journal_path": str(outcome.journal_path)})
            transport_error = outcome.status == "error" and any(name in (outcome.error or "") for name in (
                "RemoteProtocolError", "ReadTimeout", "ConnectTimeout", "ConnectError", "APIConnectionError"))
            update(agent_status=outcome.status, agent_error=outcome.error, usage=outcome.usage,
                   actor_finished_at=now(), budget_exhausted=bool(budget.exhausted) or outcome.status == "timeout")
            if isolated and transport_error:
                update(retryable=True, retry_kind="actor")
            return outcome

    class OwnedSession(grading.SandboxSession):
        def __init__(self, *args: Any, **kwargs: Any):
            if manifest.get("recovery") and kwargs.get("backend"):
                backend = kwargs["backend"]
                kwargs["backend"] = {**backend, "microvm": {**backend["microvm"], "request_timeout": 900}}
            super().__init__(*args, **kwargs)

        def create(self, image: str, resources: dict | None = None) -> bool:
            result = super().create(image, resources)
            if result:
                own(self.sandbox_id)
            return result

    original_prepare = SWEbenchPro.prepare_image
    original_grade = fork_eval.grade_attempt

    def prepare(bench: Any, *args: Any) -> str:
        if manifest.get("recovery"):
            if continuation:
                return continuation["snapshot_id"]
            receipt = json.loads((root / "image-ready" / f"{item['index']:03d}.json").read_text())
            if not receipt.get("ok") or receipt.get("image") != item["image"]:
                raise RuntimeError("Prepared image receipt does not match the task")
            if receipt.get("runtime_port", 3000) != manifest.get("runtime_port", 3000):
                raise RuntimeError("Prepared image runtime port does not match the task")
            return receipt["snapshot_id"]
        update(phase="image_preparation")
        with lane(root, "prepare", manifest["prepare_workers"]):
            return original_prepare(bench, *args)

    def grade(outcome: Any, *args: Any) -> Any:
        if isolated and state.get("retry_kind") == "actor":
            return fork_eval.Grade(error=state.get("local_stop_reason") or outcome.error or "Actor execution requires recovery")
        update(phase="waiting_for_grader")
        with lane(root, "grade", manifest["grade_workers"]):
            update(phase="grading")
            result = original_grade(outcome, *args)
        valid = not result.error and not result.verifier_artifact_error and bool(result.verifier_artifacts)
        if valid:
            artifacts = Path(result.verifier_artifacts)
            valid = all((artifacts / name).is_file() for name in ("metadata.json", "grade.json", "verifier-logs.tar.gz"))
        save(directory / "grade.json", {"task": item["id"], "evidence_valid": bool(valid),
                                         "agent_status": outcome.status, "grade": asdict(result)})
        update(evidence_valid=bool(valid), resolved=bool(result.resolved) if valid else None,
               grading_error=result.error or result.verifier_artifact_error)
        if isolated and not valid:
            update(retryable=True, retry_kind="grading")
        return result

    update()
    tracking = track_creations(state, lock, update, own) if isolated else None
    code = 2
    try:
        if tracking:
            tracking.__enter__()
        if version("claude-agent-sdk") != manifest["sdk_version"]:
            raise RuntimeError("Pinned Claude Agent SDK version changed")
        if hashlib.sha256(Path(manifest["bundled_cli"]).read_bytes()).hexdigest() != manifest["bundled_cli_sha256"]:
            raise RuntimeError("Pinned Claude CLI binary changed")
        for relative, expected in manifest["sha256"].items():
            if relative.startswith("source/") or relative == item["data"]:
                if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Frozen source/data changed: {relative}")
        os.environ.update(MCP_TOOL_TIMEOUT="3600000", CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT="3600000")
        interceptors.MutationTracker = lambda: budget
        claude_code.ClaudeCodeSlot = BudgetedSlot
        fork_eval.Orchestrator = RecordedOrchestrator
        fork_eval.grade_attempt = grade
        SWEbenchPro.prepare_image = prepare
        grading.SandboxSession = OwnedSession
        backend_arguments = []
        for key, flag in (("runtime_port", "--pro-runtime-port"),
                          ("collector_runtime_port", "--pro-collector-runtime-port"),
                          ("agent_network", "--agent-network"),
                          ("verifier_network", "--verifier-network")):
            value = item.get(key, manifest.get(key))
            if value is not None:
                backend_arguments.extend([flag, str(value)])
        code = fork_eval.main(["--benchmark", "swebench-pro", "--pro-repo", str(root / "upstream"),
                               "--pro-data", str(root / item["data"]), "--instance", item["id"],
                               "--slot", manifest["slot"], "--model", manifest["model"],
                               "--rounds", "0", "--timeout", str(manifest["actor_timeout_s"]),
                               "--pro-cpus", str(manifest.get("cpu", 4)),
                               "--pro-memory-mb", str(manifest.get("memory_mb", 16384)),
                               "--pro-verifier-timeout", str(manifest["verifier_timeout_s"]),
                               *backend_arguments,
                               "--runtime-bin", str(root / "source/runtime/ash-runtime"), "-o", str(shard)])
        update(status="completed" if state["evidence_valid"] else "held", exit_code=code)
    except BaseException as exc:
        update(status="held", error=f"{type(exc).__name__}: {exc}", exit_code=2)
        if isolated:
            update(retryable=bool(state.get("retryable")) or isinstance(exc, (OSError, TimeoutError, KeyboardInterrupt, httpx.TransportError)))
        traceback.print_exc()
    finally:
        if tracking:
            tracking.__exit__(None, None, None)
        try:
            inventory = api("/v2/sandboxes?limit=100" if isolated else "/sandboxes")
            if isolated and len(inventory) >= 100:
                raise RuntimeError("Sandbox inventory may be paginated")
            leaked = {entry["sandboxID"] for entry in inventory} & owned
            update(remaining_owned_sandboxes=sorted(leaked), cleanup_confirmed=not leaked)
            if leaked:
                if isolated:
                    update(retryable=True, cleanup_confirmed=False)
                else:
                    save(root / "STOP_REQUEST.json", {"reason": "owned sandbox leak", "ids": sorted(leaked)})
        except Exception as exc:
            if isolated:
                update(cleanup_confirmed=False, cleanup_error=str(exc))
            else:
                save(root / "STOP_REQUEST.json", {"reason": "cleanup check failed", "error": str(exc)})
        update(finished_at=now())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
