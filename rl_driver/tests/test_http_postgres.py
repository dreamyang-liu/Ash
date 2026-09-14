"""Real TCP driver -> Run Store API -> isolated PostgreSQL contract checks.

Claims/results are supplied by a controlled worker peer. These checks do not
run agents, create VMs, or establish training-token export/native restore.
"""

from contextlib import contextmanager
import socket
import threading
import time

import uvicorn

from rl_driver.backend import RunStoreClient
from rl_driver.client import ExecutionClient as Client
from rl_driver.driver import Driver
from rl_driver.ledger import Ledger
from rl_driver.server import create_app
from rl_driver.tests.test_driver import request
from runstore.api import create_app as runstore_app
from runstore.tests.test_store import store  # isolated schema; never a production database


@contextmanager
def serving(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise AssertionError("Test HTTP service did not start")
            time.sleep(0.01)
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "Test service did not stop"


def claim(store, kind):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = store.claim("controlled-worker", kinds=(kind,))
        if job:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Driver did not submit a {kind} job")


class ControlledIndex:
    """Native/snapshot validity is already tested by Run Store's own suite."""
    def __init__(self):
        self.recoveries = []

    def points(self, attempt):
        return [point for point in self.recoveries if point["attempt_id"] == attempt]

    def get_point(self, identifier):
        return next(point for point in self.recoveries if point["id"] == identifier)

    def valid(self, point):
        return point["valid"]

    def tools(self, attempt, after, limit):
        return [{"depth": n, "response": {"output": "ok"}} for n in range(1, 4) if n > after][:limit]

    def add(self, job):
        self.recoveries.append({"id": "point-" + job["id"], "job_id": job["id"],
                               "attempt_id": job["active_attempt"], "valid": True,
                               "snapshot_id": "snapshot-final", "tool_depth": 3,
                               "message_step": 2, "native": {"slot": "codex"}})


def test_real_http_queue_grade_branch_and_driver_restart(store, tmp_path):
    index = ControlledIndex()
    profiles = {"actor": {"run_defaults": {"slot": "codex"}}, "grade": {}}
    with serving(runstore_app(store, "runstore-fixture", index=index, profiles=profiles)) as backend_url:
        backend = RunStoreClient(backend_url, "runstore-fixture")
        path = tmp_path / "driver.sqlite3"
        ledger = Ledger(path)
        ledger.bind(backend_url)
        driver = Driver(backend, ledger)
        try:
            with serving(create_app(driver, "driver-fixture", poll_interval_s=0.01)) as driver_url:
                client = Client(driver_url, "driver-fixture")
                try:
                    body = request(grade=True)
                    group_id = client.submit(body)
                    actor = claim(store, "rollout")
                    assert actor["request"]["context"]["rl_driver"]["sample_slot_id"] == "slot-0"
                    index.add(actor)
                    store.append_events(actor["id"], actor["lease_token"], [
                        {"seq": 1, "type": "tool.started", "call_id": "call-1"},
                        {"seq": 2, "type": "tool.finished", "call_id": "call-1", "output": "ok"},
                    ])
                    store.finish(actor["id"], actor["lease_token"], {"status": "completed", "final_text": "patch"})
                    grade = claim(store, "grade")
                    assert grade["request"]["spec"]["snapshot_id"] == "snapshot-final"
                    assert grade["request"]["context"]["rl_driver"]["actor_attempt_id"] == actor["active_attempt"]
                    store.finish(grade["id"], grade["lease_token"], {"status": "completed", "resolved": False})
                    result = client.wait(group_id, timeout_s=5, interval_s=0.01)
                    assert result["status"] == "completed"
                    assert result["samples"][0]["grade"]["result"]["resolved"] is False
                    assert client.events(group_id, "slot-0", after=1)[0]["seq"] == 2
                    point = client.recovery_points(group_id, "slot-0")[0]
                    assert point["available"] is True
                    client.release(group_id)
                finally:
                    client.close()
            restored = Driver(backend, Ledger(path))
            with serving(create_app(restored, "driver-fixture", poll_interval_s=0.01)) as driver_url:
                client = Client(driver_url, "driver-fixture")
                try:
                    assert client.submit(body) == group_id
                    assert client.get(group_id)["acknowledged_at"] is not None
                    assert len(store.list_jobs()) == 2
                    branch_body = {"rollout_job_id": "branch-group", "prompt_group_id": "prompt", "samples": [{
                        "sample_slot_id": "child-0", "branch": {"job_id": actor["id"], "point_id": point["id"],
                                                                  "overrides": {"prompt": "Try a different fix"}},
                    }]}
                    client.submit(branch_body)
                    child = claim(store, "rollout")
                    assert child["request"]["parent_point"] == point["id"]
                    assert child["request"]["spec"]["prompt"] == "Try a different fix"
                    assert child["request"]["spec"]["sandbox_image"] == actor["request"]["spec"]["sandbox_image"]
                    store.finish(child["id"], child["lease_token"], {"status": "completed"})
                    assert client.wait("branch-group", timeout_s=5, interval_s=0.01)["status"] == "completed"
                    assert len(store.list_jobs()) == 3
                finally:
                    client.close()
        finally:
            backend.close()


def test_real_http_cancel_waits_for_running_execution(store, tmp_path):
    with serving(runstore_app(store, "fixture")) as url:
        backend = RunStoreClient(url, "fixture")
        driver = Driver(backend, Ledger(tmp_path / "driver.sqlite3"))
        try:
            with serving(create_app(driver, "fixture", poll_interval_s=0.01)) as driver_url:
                client = Client(driver_url, "fixture")
                try:
                    group_id = client.submit(request())
                    actor = claim(store, "rollout")
                    assert client.release(group_id)["status"] == "cancelling"
                    assert store.get(actor["id"])["state"] == "running"
                    assert not client.get(group_id)["ready"]
                    store.finish(actor["id"], actor["lease_token"], {"status": "completed"})
                    assert client.wait(group_id, timeout_s=5, interval_s=0.01)["status"] == "cancelled"
                finally:
                    client.close()
        finally:
            backend.close()


def test_miles_request_through_queue_orchestrator_and_recorded_session(store, tmp_path, monkeypatch):
    """Real transport and execution wiring with fixture inference and tool executor."""
    from types import SimpleNamespace
    from fastapi import FastAPI
    from harness.core.journal import read_journal
    from harness.core.result import ToolResult
    from harness.core.slot import SlotResult
    from harness.execution.pipeline import CallContext, ToolPipeline
    from harness.orchestrator.run import Orchestrator, RunSpec
    from rl_driver.client import Client as MilesClient
    from rl_driver.miles import MilesAdapter
    from rl_driver.tests.test_miles import miles_request, config, recorded_state
    import httpx

    body = miles_request()
    body["budgets"]["max_wall_time_seconds"] = 30
    states, released, executions, owned_servers = {}, [], [], []
    session_app = FastAPI()

    @session_app.post("/sessions")
    def new_session():
        sid = f"session-{len(states)}"
        states[sid] = {}
        return {"session_id": sid}

    @session_app.post("/sessions/{sid}/v1/responses")
    def generate(sid: str, payload: dict):
        assert payload["model"] == "served-model"
        states[sid] = recorded_state(body)
        return {"id": sid, "output": [], "usage": {"input_tokens": 2, "output_tokens": 1}}

    @session_app.get("/sessions/{sid}")
    def read_session(sid: str):
        return states[sid]

    @session_app.delete("/sessions/{sid}")
    def delete_session(sid: str):
        released.append(sid)
        return {}

    def provision(*args):
        server = SimpleNamespace(pipeline=ToolPipeline())
        owned_servers.append(server)
        return SimpleNamespace(server=server, sandbox_id="fixture-vm", destroy=lambda: None), None

    class FixtureSlot:
        def run(self, task, journal, mcp):
            response = httpx.post(task.env["ANTHROPIC_BASE_URL"] + "/v1/responses",
                                  headers={"Authorization": "Bearer " + task.env["ASH_GATEWAY_TOKEN"]},
                                  json={"model": "served-model", "input": task.prompt}, timeout=5)
            assert response.status_code == 200
            def execute(name, args):
                executions.append(name)
                return ToolResult(True, "fixture-output")
            result = owned_servers[-1].pipeline.execute(CallContext("agent", "vm", "shell", {}), execute)
            assert result.success
            return SlotResult(status="completed", final_text="done")

    monkeypatch.setattr(Orchestrator, "_wire_sandbox", provision)
    monkeypatch.setattr(Orchestrator, "_wire_checkpoints", lambda *args: None)
    monkeypatch.setattr("harness.slots.load_slot", lambda name: FixtureSlot)
    with serving(session_app) as session_url, serving(runstore_app(store, "fixture")) as queue_url:
        body["session_server_endpoint"] = session_url
        backend = RunStoreClient(queue_url, "fixture")
        driver = Driver(backend, Ledger(tmp_path / "miles-ledger"))
        adapter = MilesAdapter(driver, config(body))
        try:
            with serving(create_app(driver, None, miles=adapter, poll_interval_s=0.01)) as url:
                client = MilesClient(url)
                try:
                    group = client.submit(body)
                    for index in range(2):
                        job = claim(store, "rollout")
                        spec = {**job["request"]["spec"], "run_id": job["active_attempt"],
                                "journal_path": tmp_path / f"attempt-{index}.jsonl"}
                        outcome = Orchestrator().run(RunSpec(**spec))
                        assert outcome.status == "completed", outcome.error
                        store.append_events(job["id"], job["lease_token"], list(read_journal(spec["journal_path"])))
                        store.finish(job["id"], job["lease_token"], {"status": outcome.status, "final_text": outcome.final_text})
                    result = client.wait(group, timeout_s=5, interval_s=0.01)
                    assert result["status"] == "completed", result
                    assert result["actual_samples"] == 2
                    assert result["consumed_budget"] == {"model_calls": 2, "tool_calls": 2}
                    assert len(executions) == 2 and len(released) == 2
                    assert [t["sample_slot_id"] for t in result["trajectories"]] == [s["sample_slot_id"] for s in body["sample_slots"]]
                    assert result["trajectories"][0]["token_ids"] == [*body["prompt_token_ids"], 90]
                finally:
                    client.close()
        finally:
            backend.close()


def test_worker_bounds_child_startup_by_remaining_group_deadline(store, tmp_path, monkeypatch):
    from pathlib import Path
    from runstore.specs import JobSpec, digest
    from runstore.worker import Worker

    seen = []
    def before_launch(command, **kwargs):
        directory = Path(command[-2])
        payload = store.payload(directory.parent.name, directory.name)
        assert command[-1] == digest(payload)
        seen.append(payload)
        raise RuntimeError("Controlled probe stops before child execution")
    monkeypatch.setattr("runstore.worker.subprocess.Popen", before_launch)
    spec = {"prompt": "task", "slot": "codex", "sandbox_image": "fixture",
            "backend": {"backend": "microvm"}, "timeout_s": 3600,
            "extra": {"rollout_contract": {"deadline_at": time.time() + 5}}}
    first = store.submit(JobSpec("rollout", spec), "deadline")
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": {}}})
    worker.run_once()
    assert 0 < seen[0]["effective_spec"]["timeout_s"] <= 5
    assert store.get(first["id"])["request"]["spec"]["timeout_s"] == 3600
    spec["extra"]["rollout_contract"]["deadline_at"] = time.time() - 1
    second = store.submit(JobSpec("rollout", spec), "expired")
    worker.run_once()
    assert len(seen) == 1
    assert "expired before execution" in store.get(second["id"])["error"]
