from copy import deepcopy
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from rl_driver.backend import RunStoreClient
from rl_driver.driver import Driver
from rl_driver.ledger import Conflict, Ledger, canonical
from rl_driver.server import DEFAULT_PORT, create_app
from rl_driver.specs import PROTOCOL_VERSION
from runstore.specs import JobSpec


def request(group="group", *, grade=False):
    sample = {"sample_slot_id": "slot-0", "run": {"profile": "actor", "spec": {
        "prompt": "Fix the parser", "slot": "codex", "sandbox_image": "prepared-image", "timeout_s": 30,
    }}}
    if grade:
        sample["grade"] = {"profile": "grade", "spec": {
            "benchmark": "swebench-verified", "instance_id": "task", "dataset_path": "/worker/task.json",
            "dataset_sha256": "123", "grader_revision": "3.0.15", "timeout_s": 60,
        }}
    return {"rollout_job_id": group, "prompt_group_id": "prompt", "samples": [sample]}


class QueueHTTP:
    """An HTTP contract peer; only explicit finish() simulates a worker."""
    def __init__(self):
        self.jobs = {}
        self.keys = {}
        self.requests = []
        self.lose_ack = False
        self.tools = {}
        self.points = {}
        self.events = {}

    def __call__(self, req):
        body = json.loads(req.content) if req.content else None
        path = req.url.path
        self.requests.append((req.method, path, body, dict(req.url.params)))
        if req.method == "POST" and (path == "/v1/jobs" or path.endswith("/branch")):
            key = req.headers["Idempotency-Key"]
            if key in self.keys:
                job_id, previous = self.keys[key]
                assert previous == canonical(body), "Retry changed the submitted job"
            else:
                job_id = f"job-{len(self.jobs)}"
                kind = JobSpec.from_dict(body).kind if path == "/v1/jobs" else "rollout"
                self.jobs[job_id] = {"id": job_id, "kind": kind, "state": "queued", "phase": "queued",
                                     "idempotency_key": key, "active_attempt": None, "result": None, "error": None}
                self.keys[key] = (job_id, canonical(body))
            if self.lose_ack:
                self.lose_ack = False
                raise httpx.ReadError("Reply lost after durable acceptance", request=req)
            return httpx.Response(202, json={"id": job_id})
        job_id = path.split("/")[3]
        job = self.jobs[job_id]
        if req.method == "GET" and path == f"/v1/jobs/{job_id}":
            return httpx.Response(200, json=job)
        if path.endswith("/cancel"):
            if job["state"] == "queued":
                job.update(state="cancelled", phase="cancelled")
            elif job["state"] == "running":
                job.update(phase="cancelling", cancel_requested=True)
            return httpx.Response(200, json=job)
        if path.endswith("/tools"):
            after = int(req.url.params.get("after", 0))
            # Small pages ensure the real adapter must follow the cursor.
            rows = [row for row in self.tools.get(job_id, []) if row["depth"] > after][:2]
            return httpx.Response(200, json=rows)
        if path.endswith("/recovery-points"):
            return httpx.Response(200, json=self.points.get(job_id, []))
        if path.endswith("/events"):
            after = int(req.url.params.get("after", 0))
            limit = int(req.url.params.get("limit", 1000))
            event_types = tuple(req.url.params.get_list("event_type"))
            rows = [
                event for event in self.events.get(job_id, [])
                if event["seq"] > after
                and (not event_types or event.get("type") in event_types)
            ]
            if req.url.params.get("newest") == "true":
                rows.reverse()
            rows = rows[:limit]
            return httpx.Response(200, json=rows)
        raise AssertionError((req.method, path))

    def finish(self, job_id, *, resolved=None):
        self.jobs[job_id].update(state="succeeded", active_attempt=f"attempt-{job_id}",
                                 result={"status": "completed", "resolved": resolved})

    def cancel(self, job_id):
        self.jobs[job_id].update(
            state="cancelled", phase="cancelled", active_attempt=f"attempt-{job_id}",
            result={"status": "cancelled", "stop_reason": "cancelled"},
        )

    def final_point(self, job_id):
        self.events.setdefault(job_id, []).append({
            "seq": len(self.events.get(job_id, [])) + 1,
            "type": "environment.prepared",
            "workdir": "/task-repository",
            "base_commit": "1" * 40,
            "agent_workdir": "/testbed",
            "baseline_untracked": ["image-cache.txt"],
        })
        self.tools[job_id] = [{"depth": n, "response": {"output": "ok"}} for n in range(1, 4)]
        self.points[job_id] = [
            {"id": "old", "tool_depth": 2, "message_step": 1, "snapshot_id": "AB", "available": True},
            {"id": "last", "tool_depth": 3, "message_step": 2, "snapshot_id": "ABC", "available": True},
        ]


@pytest.fixture
def peer():
    peer = QueueHTTP()
    client = RunStoreClient("http://runstore", "fixture")
    client.http.close()
    client.http = httpx.Client(base_url="http://runstore", transport=httpx.MockTransport(peer))
    try:
        yield peer, client
    finally:
        client.close()


def test_two_samples_http_polling_out_of_order_and_consumption_survives_restart(tmp_path, peer):
    queue, client = peer
    path = tmp_path / "ledger.sqlite3"
    driver = Driver(client, Ledger(path))
    body = request()
    second = deepcopy(body["samples"][0])
    second["sample_slot_id"] = "slot-1"
    body["samples"].append(second)
    with TestClient(create_app(driver, "driver", background=False)) as http:
        assert http.post("/execution-groups", json=body).status_code == 401
        http.headers["Authorization"] = "Bearer driver"
        assert http.post("/execution-groups", json=body).json()["status"] == "queued"
        driver.tick()
        assert len(queue.jobs) == 2
        queue.finish("job-1")
        driver.tick()
        assert not http.get("/execution-groups/group").json()["ready"]
        queue.finish("job-0")
        queue.events["job-0"] = [{"seq": 1, "type": "tool.started"}, {"seq": 2, "type": "tool.finished"}]
        driver.tick()
        view = http.get("/execution-groups/group/result").json()
        assert view["status"] == "completed"
        assert [s["actor"]["job_id"] for s in view["samples"]] == ["job-0", "job-1"]
        assert "submission" not in view["samples"][0]["actor"]
        assert http.get("/execution-groups/group/samples/slot-0/events?after=1").json() == queue.events["job-0"][1:]
        assert queue.requests[-1][-1]["attempt_id"] == "attempt-job-0"
        assert http.delete("/execution-groups/group").status_code == 200
        changed = deepcopy(body)
        changed["samples"][0]["run"]["spec"]["prompt"] = "different"
        assert http.post("/execution-groups", json=changed).status_code == 409
    restored = Driver(client, Ledger(path))
    assert restored.submit(body)["status"] == "completed"
    assert restored.get("group")["acknowledged_at"] is not None
    restored.tick()
    assert len(queue.jobs) == 2
    assert DEFAULT_PORT == 11001


def test_all_events_uses_bounded_cursor_pages(peer):
    queue, client = peer
    queue.jobs["job"] = {
        "id": "job", "kind": "rollout", "state": "succeeded",
        "phase": "succeeded", "active_attempt": "attempt-job",
        "result": {}, "error": None,
    }
    queue.events["job"] = [
        {"seq": seq, "type": "agent.message"}
        for seq in range(1, 122)
    ]

    assert client.all_events("job", attempt_id="attempt-job") == queue.events["job"]
    requests = [request for request in queue.requests if request[1].endswith("/events")]
    assert [request[3]["after"] for request in requests] == ["0", "50", "100", "121"]
    assert {request[3]["limit"] for request in requests} == {"50"}


def test_swe_rebench_grade_selects_only_repository_baseline_event(tmp_path, peer):
    queue, client = peer
    body = request(grade=True)
    body["samples"][0]["grade"]["spec"]["benchmark"] = "swe-rebench-v2"
    body["samples"][0]["grade"]["spec"]["parser_path"] = "/worker/parser.py"
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(body)
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    # A real long rollout also has a very large SessionTree event.  The grade
    # preparation path must never request or scan it just to find the baseline.
    queue.events["job-0"].append({
        "seq": 2,
        "type": "rollout.session_state",
        "state": {"text": "large-state-placeholder"},
    })

    driver.tick()

    grade = next(
        submitted
        for method, _path, submitted, _params in queue.requests
        if method == "POST" and submitted.get("kind") == "grade"
    )
    assert grade["spec"]["baseline_untracked"] == ["image-cache.txt"]
    event_requests = [
        request for request in queue.requests if request[1].endswith("/events")
    ]
    assert event_requests[-1][3]["event_type"] == "environment.prepared"
    assert event_requests[-1][3]["limit"] == "2"


@pytest.mark.parametrize("branch", [False, True])
def test_lost_submit_reply_restarts_with_same_idempotency_key(tmp_path, peer, branch):
    queue, client = peer
    body = request()
    if branch:
        body["samples"][0].pop("run")
        body["samples"][0]["branch"] = {"job_id": "parent", "point_id": "point", "overrides": {"prompt": "retry"}}
    path = tmp_path / "ledger.sqlite3"
    driver = Driver(client, Ledger(path))
    driver.submit(body)
    queue.lose_ack = True
    driver.tick()
    assert len(queue.jobs) == 1
    assert driver.get("group")["samples"][0]["actor"]["state"] == "submitting"
    restored = Driver(client, Ledger(path))
    restored.tick()
    assert len(queue.jobs) == 1
    assert restored.get("group")["samples"][0]["actor"]["job_id"] == "job-0"


def test_cancel_reaches_running_job_and_does_not_launch_grading(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    body = request(grade=True)
    body["samples"].append({"sample_slot_id": "slot-1", "run": deepcopy(body["samples"][0]["run"])})
    driver.submit(body)
    driver.tick()
    queue.jobs["job-0"].update(state="running", active_attempt="attempt-job-0")
    assert driver.release("group")["status"] == "cancelling"
    driver.tick()
    assert queue.jobs["job-0"]["state"] == "running"
    assert queue.jobs["job-0"]["phase"] == "cancelling"
    assert queue.jobs["job-0"]["cancel_requested"] is True
    assert queue.jobs["job-1"]["state"] == "cancelled"
    assert not driver.get("group")["ready"]
    queue.cancel("job-0")
    driver.tick()
    assert driver.get("group")["status"] == "cancelled"
    assert driver.get("group")["ready"]
    assert len(queue.jobs) == 2


def test_cancel_before_dispatch_never_submits(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.release("group")
    driver.tick()
    assert driver.get("group")["status"] == "cancelled"
    assert queue.requests == []


def test_internal_deferred_slots_reuse_runstore_branch(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    body = request()
    template = body["samples"][0].pop("run")
    template["spec"]["extra"] = {"rollout_contract": {
        "session_id": "shared-session", "max_model_calls": 4,
        "max_tool_calls": 4,
    }}
    body["samples"] = [
        {"sample_slot_id": "parent", "run": deepcopy(template)},
        {"sample_slot_id": "child", "run": deepcopy(template)},
    ]
    driver.submit(body, deferred_sample_ids={"child"})
    driver.tick()
    assert len(queue.jobs) == 1
    assert driver.get("group")["status"] == "running"
    parent_job = next(iter(queue.jobs))
    queue.finish(parent_job)
    queue.points[parent_job] = [{
        "id": "joint-point", "available": True, "tool_depth": 1,
        "message_step": 1, "snapshot_id": "snapshot", "model_position": {
            "session_id": "shared-session", "response_id": "parent-response",
        },
    }]
    driver.tick()

    decision = {
        "kind": "branch", "source_sample_slot_id": "parent",
        "point_id": "joint-point", "overrides": {"prompt": "try another path"},
    }
    driver.decide_deferred("group", "child", decision)
    assert driver.decide_deferred("group", "child", decision)["samples"][1]["actor"][
        "policy_decision"
    ] == decision
    with pytest.raises(Conflict, match="different policy decision"):
        driver.decide_deferred("group", "child", {"kind": "skip"})
    driver.tick()
    child_job = next(job for job in queue.jobs if job != parent_job)
    branch_request = next(
        body for method, path, body, _ in queue.requests
        if method == "POST" and path.endswith("/branch")
    )
    assert branch_request["point_id"] == "joint-point"
    assert branch_request["context"]["rl_driver"]["sample_slot_id"] == "child"
    assert "rollout_contract" not in branch_request
    queue.finish(child_job)
    driver.tick()
    result = driver.get("group")
    assert result["ready"] and result["status"] == "completed"
    assert result["samples"][1]["actor"]["origin"] == {
        "job_id": parent_job, "point_id": "joint-point"
    }


def test_deferred_slot_rejects_foreign_or_unavailable_branch_point(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    body = request()
    template = body["samples"][0].pop("run")
    body["samples"] = [
        {"sample_slot_id": "parent", "run": deepcopy(template)},
        {"sample_slot_id": "child", "run": deepcopy(template)},
    ]
    driver.submit(body, deferred_sample_ids={"child"})
    driver.tick()
    parent_job = next(iter(queue.jobs))
    queue.finish(parent_job)
    queue.points[parent_job] = [{"id": "bad", "available": False}]
    driver.tick()
    with pytest.raises(ValueError, match="absent or unavailable"):
        driver.decide_deferred("group", "child", {
            "kind": "branch", "source_sample_slot_id": "parent",
            "point_id": "bad", "overrides": {},
        })
    assert len(queue.jobs) == 1


def test_deferred_decision_survives_driver_restart_without_duplicate_branch(
    tmp_path, peer
):
    queue, client = peer
    path = tmp_path / "ledger.sqlite3"
    body = request()
    template = body["samples"][0]["run"]
    body["samples"] = [
        {"sample_slot_id": "parent", "run": deepcopy(template)},
        {"sample_slot_id": "child", "run": deepcopy(template)},
    ]
    driver = Driver(client, Ledger(path))
    driver.submit(body, deferred_sample_ids={"child"})
    driver.tick()
    queue.finish("job-0")
    queue.points["job-0"] = [{
        "id": "point", "available": True, "snapshot_id": "snapshot",
        "tool_depth": 1, "message_step": 1,
    }]
    driver.tick()
    decision = {
        "kind": "branch", "source_sample_slot_id": "parent",
        "point_id": "point", "overrides": {},
    }
    driver.decide_deferred("group", "child", decision)

    restored = Driver(client, Ledger(path))
    assert restored.decide_deferred("group", "child", decision)["samples"][1][
        "actor"
    ]["policy_decision"] == decision
    restored.tick()
    restored.tick()
    assert len(queue.jobs) == 2
    assert len([
        request for request in queue.requests
        if request[0] == "POST" and request[1].endswith("/branch")
    ]) == 1


def test_cancellation_terminates_deferred_slots_without_launching_them(tmp_path, peer):
    queue, client = peer
    body = request()
    second = deepcopy(body["samples"][0])
    second["sample_slot_id"] = "child"
    body["samples"].append(second)
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(body, deferred_sample_ids={"child"})
    driver.tick()
    assert len(queue.jobs) == 1
    driver.release("group")
    driver.tick()
    assert len(queue.jobs) == 1
    assert driver.get("group")["samples"][1]["actor"]["state"] == "cancelled"


def test_cancel_lost_ack_does_not_repost_possibly_new_work(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request())
    queue.lose_ack = True
    driver.tick()
    driver.release("group")
    calls = len(queue.requests)
    driver.tick()
    assert len(queue.requests) == calls
    view = driver.get("group")
    assert view["status"] == "cancelling" and not view["ready"]
    assert "acknowledgement unknown" in view["samples"][0]["actor"]["last_error"]
    queue.jobs["wrong-job"] = {**queue.jobs["job-0"], "id": "wrong-job", "idempotency_key": "different"}
    with pytest.raises(ValueError, match="idempotency key"):
        driver.reconcile_submission("group", "slot-0", "actor", "wrong-job")
    driver.reconcile_submission("group", "slot-0", "actor", "job-0")
    driver.tick()
    assert queue.jobs["job-0"]["state"] == "cancelled"
    assert driver.get("group")["ready"]
    assert not [r for r in queue.requests[calls:] if r[0] == "POST" and r[1] == "/v1/jobs"]


def test_cancellation_during_grade_preparation_does_not_submit_grader(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    original = client.points

    def points(*args, **kwargs):
        driver.release("group")
        return original(*args, **kwargs)

    client.points = points
    driver.tick()
    assert driver.get("group")["status"] == "cancelled"
    assert len(queue.jobs) == 1


@pytest.mark.parametrize("resolved", [True, False])
def test_grade_uses_final_snapshot_and_unresolved_is_completed(tmp_path, peer, resolved):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    driver.tick()
    grades = [body for method, path, body, _ in queue.requests if method == "POST" and body.get("kind") == "grade"]
    assert grades[0]["spec"]["snapshot_id"] == "ABC"
    assert grades[0]["context"]["rl_driver"]["actor_attempt_id"] == "attempt-job-0"
    queue.finish("job-1", resolved=resolved)
    driver.tick()
    view = driver.get("group")
    assert view["status"] == "completed"
    assert view["samples"][0]["grade"]["result"]["resolved"] is resolved
    assert len(queue.jobs) == 2


def test_v2_grade_prefers_worker_final_snapshot_without_recovery_points(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.jobs["job-0"]["result"]["final_snapshot_id"] = "actor-final"
    driver.tick()

    grades = [
        body for method, path, body, _ in queue.requests
        if method == "POST" and body.get("kind") == "grade"
    ]
    assert grades[0]["spec"]["snapshot_id"] == "actor-final"


def test_unavailable_final_message_never_grades_an_older_snapshot(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    queue.points["job-0"][-1]["available"] = False
    driver.tick()
    view = driver.get("group")
    assert view["status"] == "failed"
    assert "older snapshot" in view["samples"][0]["grade"]["error"]
    assert len(queue.jobs) == 1


def test_grade_lost_ack_keeps_original_snapshot_on_restart(tmp_path, peer):
    queue, client = peer
    path = tmp_path / "ledger.sqlite3"
    driver = Driver(client, Ledger(path))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    queue.lose_ack = True
    driver.tick()
    queue.points["job-0"] = []
    Driver(client, Ledger(path)).tick()
    assert len(queue.jobs) == 2
    assert driver.get("group")["samples"][0]["grade"]["job_id"] == "job-1"


def test_quarantine_terminates_group_without_rerunning_from_driver(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request())
    driver.tick()
    queue.jobs["job-0"]["state"] = "quarantined"
    driver.tick()
    assert driver.get("group")["status"] == "failed"
    assert driver.get("group")["ready"]
    queue.finish("job-0")
    driver.tick()
    # Run Store may retain/reconcile the quarantined job independently, but
    # this fixed RL group is already terminal and is never silently rewritten.
    assert driver.get("group")["status"] == "failed"
    assert len(queue.jobs) == 1


def test_one_broken_group_does_not_block_another(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request("first"))
    driver.submit(request("second"))
    queue.lose_ack = True
    driver.tick()
    assert len(queue.jobs) == 2
    assert driver.get("second")["samples"][0]["actor"]["job_id"] == "job-1"


@pytest.mark.parametrize("mutate", [
    lambda body: body.update(protocol_version="ash-rollout-v2"),
    lambda body: body.update(budgets={"max_model_calls": 2}),
    lambda body: body["samples"].append(deepcopy(body["samples"][0])),
    lambda body: body["samples"][0]["run"]["spec"].update(sampling_params={"temperature": 1}),
    lambda body: body["samples"][0]["run"]["spec"].update(api_key="inline-secret"),
])
def test_unsupported_contracts_rejected_before_dispatch(tmp_path, peer, mutate):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    body = request()
    mutate(body)
    with pytest.raises((ValueError, TypeError)):
        driver.submit(body)
    driver.tick()
    assert not queue.requests


def test_ledger_refuses_second_controller_and_different_backend(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.bind("http://original")
    with pytest.raises(Conflict):
        ledger.bind("http://different")
    with ledger.owner():
        with pytest.raises(Conflict):
            with Ledger(ledger.path).owner():
                pytest.fail("Second controller acquired the ledger")


def test_cli_wires_original_port_and_separate_runstore_url(tmp_path, monkeypatch):
    from rl_driver import __main__ as command

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"ledger": "state.sqlite3"}))
    monkeypatch.setenv("ASH_RUNSTORE_TOKEN", "backend-fixture")
    monkeypatch.setenv("ASH_RL_DRIVER_TOKEN", "driver-fixture")
    monkeypatch.setattr("sys.argv", ["rl_driver", "--config", str(config)])
    seen = {}
    monkeypatch.setattr(command.uvicorn, "run", lambda app, **kwargs: seen.update(app=app, **kwargs))
    command.main()
    assert seen["port"] == 11001
    assert seen["host"] == "127.0.0.1"
    assert "/execution-groups" in {route.path for route in seen["app"].routes}
    with Ledger(tmp_path / "state.sqlite3").connection() as db:
        assert db.execute("SELECT value FROM driver_meta WHERE name='backend_url'").fetchone()[0] == "http://127.0.0.1:18110"


def test_example_is_valid_and_package_has_no_execution_calls():
    import ast
    from pathlib import Path
    from rl_driver.specs import validate_request

    directory = Path(__file__).resolve().parents[1]
    body = validate_request(json.loads((directory / "execution-group.example.json").read_text()))
    assert len(body["samples"]) == 2
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("harness.", "ash_sandbox", "swebench", "runstore.worker", "runstore.store"))
            if isinstance(node, ast.Import):
                assert not any(alias.name in {"subprocess", "psycopg2", "docker"} for alias in node.names)
