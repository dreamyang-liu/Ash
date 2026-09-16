"""One isolated attempt process, with no database credentials or polling loop."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys
import threading
import time

from harness.execution.session import SandboxSession
from harness.orchestrator.resources import ResourceLedger
from harness.orchestrator.run import Orchestrator, RunSpec
from runstore.config import referenced_env, resolve
from runstore.files import read_json, write_json
from runstore.native import materialize
from runstore.payload import receive_payload


class TrackingSession(SandboxSession):
    def __init__(self, *, ledger: ResourceLedger, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ledger = ledger
        self.claim = None

    def create(self, image, resources=None):
        self.ledger._append("allocation_started", run_id="grader", image=image)
        ready = super().create(image, resources)
        if ready:
            self.ledger._append("claim", run_id="grader", kind="sandbox", id=self.sandbox_id)
        else:
            self.ledger._append("allocation_unknown", run_id="grader", image=image)
        return ready

    def destroy(self) -> None:
        identifier = self.sandbox_id
        super().destroy()
        if identifier:
            self.ledger._append("release", run_id="grader", kind="sandbox", id=identifier)


def execute(request: dict, directory: Path) -> dict:
    os.umask(0o077)
    spec = resolve(request["effective_spec"])
    profile = request["profile_config"]
    worker_env = resolve(profile.get("worker_env", {}))
    agent_env = resolve(profile.get("env", {}))
    os.environ.update(worker_env)
    os.environ.update(agent_env)
    hidden_env = (referenced_env(request) | set(worker_env)) - set(agent_env)
    cwd = directory / "cwd"
    cwd.mkdir(exist_ok=True)
    native_home = directory / "native-home"
    native_home.mkdir(exist_ok=True)
    ledger = ResourceLedger(directory / "resources.jsonl")
    if request["kind"] == "grade":
        from runstore.grading import grade

        return grade(spec, directory,
                     session_factory=lambda **kwargs: TrackingSession(ledger=ledger, **kwargs))
    slot = spec.get("slot", "claude-code")
    os.environ["CODEX_HOME" if slot == "codex" else "CLAUDE_CONFIG_DIR"] = str(native_home)
    extra = dict(spec.get("extra", {}))
    contract_overrides = (
        request.get("context", {}).get("rl_driver", {}).get(
            "rollout_contract_overrides", {}
        )
    )
    if contract_overrides:
        if not isinstance(contract_overrides, dict) or set(contract_overrides) - {
            "max_model_calls", "max_tool_calls"
        }:
            raise ValueError("Invalid RL-driver rollout contract overrides")
        contract = extra.get("rollout_contract")
        if not isinstance(contract, dict):
            raise ValueError("Rollout contract overrides require a rollout contract")
        for name, value in contract_overrides.items():
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} override must be a nonnegative integer or null")
        extra["rollout_contract"] = {**contract, **contract_overrides}
    rollout_contract = extra.get("rollout_contract", {})
    message_export = rollout_contract.get("message_export", False)
    from harness.rollout import capture_final_snapshot as should_capture_final_snapshot

    capture_final_state = should_capture_final_snapshot(rollout_contract)
    completion = {}
    if request.get("recovery"):
        recovered = request["recovery"]
        native = recovered["native"]
        if native["slot"] != slot:
            raise ValueError("Native prefix slot differs from requested slot")
        extra.update(materialize(native, cwd, directory / "restoration", native_home))
        spec["sandbox_image"] = recovered["snapshot_id"]
        spec["origin"] = {"point_id": recovered["id"], "snapshot_id": recovered["snapshot_id"],
                          "tool_depth": recovered["tool_depth"], "message_step": recovered["message_step"]}
        if extra.get("rollout_contract", {}).get("message_export"):
            from runstore.message_export import mark_hint

            spec["prompt"] = mark_hint(spec["prompt"])
        rollout_contract = extra.get("rollout_contract")
        if rollout_contract and rollout_contract.get("session_id"):
            model_position = recovered.get("model_position")
            if not isinstance(model_position, dict):
                raise ValueError(
                    "Miles-backed continuation requires a verified model position"
                )
            if model_position.get("session_id") != rollout_contract["session_id"]:
                raise ValueError(
                    "Recovery model position belongs to a different Miles session"
                )
            extra["rollout_contract"] = {
                **rollout_contract,
                "model_parent_position": model_position,
            }
    extra["exact_capture"] = True
    spec.update(cwd=str(cwd), run_id=request["attempt_id"], agent_id=request["attempt_id"],
                journal_path=directory / "trajectory.jsonl", transport="http", extra=extra)
    ledger._append("allocation_started", run_id=request["attempt_id"], image=spec.get("sandbox_image"))

    progress_lock = threading.Lock()

    def progress(kind: str, payload: dict) -> None:
        with progress_lock:
            current = read_json(directory / "progress.json") or {}
            current.update({"phase": kind, **payload})
            current["updated_at_unix_seconds"] = time.time()
            write_json(directory / "progress.json", current)

    class TrackedOrchestrator(Orchestrator):
        def _teardown(self, run_spec, gateway, provisioned, claim):
            if capture_final_state and provisioned is not None:
                from runstore.message_completion import capture_final_snapshot

                try:
                    completion.update(capture_final_snapshot(
                        provisioned, directory, ledger, request["attempt_id"]))
                except Exception as error:
                    completion["training_snapshot_error"] = str(error)
            return super()._teardown(run_spec, gateway, provisioned, claim)

        def _wire_gateway(self, run_spec, journal, task, run_id):
            task.env.update({name: "" for name in hidden_env})
            task.env.update(agent_env)
            return super()._wire_gateway(run_spec, journal, task, run_id)

        def _own_sandbox(self, run_spec, claim):
            owned = super()._own_sandbox(run_spec, claim)
            original_swap = owned.session.swap_sandbox

            def swap(snapshot):
                ledger._append("allocation_started", run_id=request["attempt_id"], image=str(snapshot))
                changed = original_swap(snapshot)
                if changed:
                    claim.sandbox(owned.session.sandbox_id)
                else:
                    ledger._append("allocation_unknown", run_id=request["attempt_id"])
                return changed

            owned.session.swap_sandbox = swap
            return owned

    outcome = TrackedOrchestrator(out_dir=directory, ledger=ledger, on_event=progress).run(RunSpec(**spec))
    result = asdict(outcome)
    result["journal_path"] = str(outcome.journal_path)
    result["native_home"] = str(native_home)
    result["failure_kind"] = "actor" if outcome.status != "completed" else None
    result.update(completion)
    if message_export:
        from runstore.message_completion import complete_message_result

        result = complete_message_result(result, directory, slot, request.get("recovery"))
    return result


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    directory = Path(sys.argv[1]).resolve()
    try:
        request = receive_payload(sys.stdin.fileno(), sys.argv[2], directory.parent.name, directory.name)
    except (OSError, ValueError):
        print("Attempt bootstrap rejected: missing, invalid or timed-out stdin payload", file=sys.stderr)
        return 2
    started = time.monotonic()

    def terminate(signum, frame):
        raise KeyboardInterrupt("Worker requested shutdown")

    signal.signal(signal.SIGTERM, terminate)
    try:
        result = execute(request, directory)
    except BaseException as error:
        result = {"status": "error", "failure_kind": "infrastructure",
                  "error": f"{type(error).__name__}: {error}"}
    result.update(attempt_id=request["attempt_id"], elapsed_s=time.monotonic() - started)
    write_json(directory / "outcome.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
