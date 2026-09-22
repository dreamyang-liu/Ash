"""One isolated attempt process, with no database credentials or polling loop."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys
import time

from harness.execution.session import SandboxSession
from harness.orchestrator.resources import ResourceLedger
from harness.orchestrator.run import Orchestrator, RunSpec
from runstore.config import referenced_env, resolve
from runstore.files import write_json
from runstore.native import materialize
from runstore.payload import receive_payload
from runstore.branch_guidance import execution_guidance


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
    if slot != "mini-swe-agent":
        os.environ["CODEX_HOME" if slot == "codex" else "CLAUDE_CONFIG_DIR"] = str(native_home)
    extra = dict(spec.get("extra", {}))
    if slot == "mini-swe-agent":
        extra["native_home"] = str(native_home)
    message_export = extra.get("rollout_contract", {}).get("message_export", False)
    completion = {}
    if request.get("recovery"):
        recovered = request["recovery"]
        native = recovered["native"]
        if native["slot"] != slot:
            raise ValueError("Native prefix slot differs from requested slot")
        extra, guidance, guidance_origin = execution_guidance(
            {**spec, "extra": extra}, recovered, request.get("job_id"))
        extra.update(materialize(native, cwd, directory / "restoration", native_home))
        spec["sandbox_image"] = recovered["snapshot_id"]
        spec["origin"] = {"point_id": recovered["id"], "snapshot_id": recovered["snapshot_id"],
                          "tool_depth": recovered["tool_depth"], "message_step": recovered["message_step"],
                          **guidance_origin}
        if guidance == "user-hint" and extra.get("rollout_contract", {}).get("message_export"):
            from runstore.message_export import mark_hint

            spec["prompt"] = mark_hint(spec["prompt"])
    extra["exact_capture"] = True
    spec.update(cwd=str(cwd), run_id=request["attempt_id"], agent_id=request["attempt_id"],
                journal_path=directory / "trajectory.jsonl", transport="http", extra=extra)
    ledger._append("allocation_started", run_id=request["attempt_id"], image=spec.get("sandbox_image"))

    def progress(kind: str, payload: dict) -> None:
        write_json(directory / "progress.json", {"phase": kind, **payload})

    class TrackedOrchestrator(Orchestrator):
        def _teardown(self, run_spec, gateway, provisioned, claim):
            if message_export and provisioned is not None:
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
    if message_export:
        from runstore.message_completion import complete_message_result

        result.update(completion)
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
