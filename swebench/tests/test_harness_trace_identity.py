from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from swebench.agent.trace import new_run_id
from swebench.agent.waggle import CoordinatedExecutor, WorkspaceCoordinator
from swebench.harnesses import best_of_n, manager_worker
from swebench.harnesses.manager_worker import _normalize_subtasks
from swebench.models import AgentConfig, CostTracker, ToolResult
from swebench.sandbox import AshSession


class _FakeTrajectory:
    def __init__(self):
        self.info = {}
        self.cost = None

    def save(self, path: Path) -> None:
        self.saved_to = path


_IDENTITY_KEYS = ("trace_dir", "run_id", "agent_id", "sandbox_id")


class _FakeAgent:
    created = []

    def __init__(self, config, executor, **kwargs):
        self.config = config
        self.executor = executor
        # Identity only — these tests are about who a call is attributed to.
        # Other kwargs (e.g. pipeline=) are recorded separately so adding one
        # does not break every identity assertion.
        self.identity = {k: v for k, v in kwargs.items() if k in _IDENTITY_KEYS}
        self.kwargs = kwargs
        self.cost = CostTracker()
        self.trajectory = _FakeTrajectory()
        self.stream = True
        self.created.append(self)

    def set_tools_schema(self, schema) -> None:
        self.schema = schema

    def run(self, task: str, instance_id: str = "") -> str:
        self.task = task
        self.instance_id = instance_id
        # Make one tool call, so tests can tell an agent's traffic apart from
        # the harness's own. A fake that never touches its executor would let
        # attribution assertions pass no matter how the wiring is broken.
        self.executor("shell", {"command": f"work {instance_id}"})
        return "completed"


class _FakeWorkerExecutor:
    created: list["_FakeWorkerExecutor"] = []

    def __init__(self, url: str, agent_id: str = ""):
        self.url = url
        self.agent_id = agent_id
        self.closed = False
        self.created.append(self)

    def __call__(self, name: str, args: dict) -> ToolResult:
        return ToolResult(success=True, output="")

    def close(self) -> None:
        self.closed = True


class _RecordingSession:
    """AshSession stand-in that records the identity behind every call.

    Mirrors the real split: ``execute`` is the harness's own channel, while
    ``executor_for`` hands out a channel with one agent's identity bound in.
    """

    def __init__(self, output: str = ""):
        self.output = output
        self.calls: list[tuple[str, str]] = []  # (agent_id, command)

    def execute(self, name, args):
        return self._run(name, args, "harness")

    def executor_for(self, agent_id: str):
        def run(name, args):
            return self._run(name, args, agent_id)
        return run

    def _run(self, name, args, agent_id):
        self.calls.append((agent_id, args.get("command", "")))
        return ToolResult(success=True, output=self.output)

    def callers(self) -> set[str]:
        return {agent_id for agent_id, _ in self.calls}


def test_run_and_sandbox_identities_are_stable():
    first = new_run_id()
    second = new_run_id()
    assert first != second
    assert len(first) == len(second) == 32

    session = AshSession()
    assert session.sandbox_id == "unknown"
    session._sandbox = SimpleNamespace(sandbox_id="sandbox-123")
    assert session.sandbox_id == "sandbox-123"


def test_manager_normalizes_subtask_ids_for_agent_and_trace_identity():
    subtasks = _normalize_subtasks({"subtasks": [
        {"id": "api/review", "description": "one"},
        {"id": "api/review", "description": "two"},
        {"id": "...", "description": "three"},
    ]})
    assert [task["id"] for task in subtasks] == ["api-review", "api-review-2", "t3"]


_PLAN_OUTPUT = '{"subtasks":[{"id":"t1","description":"one","files":[]}]}'


def _run_manager_and_workers(monkeypatch, tmp_path, n_workers=2, subtask_ids=("t1",)):
    """Drive one manager + worker round, returning the session and the agents."""
    _FakeAgent.created = []
    _FakeWorkerExecutor.created = []
    monkeypatch.setattr(manager_worker, "AshAgent", _FakeAgent)
    monkeypatch.setattr(manager_worker, "_WorkerExecutor", _FakeWorkerExecutor)

    session = _RecordingSession(output=_PLAN_OUTPUT)
    harness = manager_worker.ManagerWorkerHarness({})
    trace_dir = tmp_path / "traces"
    traj_dir = tmp_path / "trajectories"

    subtasks = harness._run_manager(
        session, {}, "problem", n_workers, 10, "iid", traj_dir,
        trace_dir=trace_dir, run_id="run-shared", sandbox_id="sandbox-shared",
    )
    harness._run_workers(
        "http://sandbox", AgentConfig(), "problem", subtasks, "iid", traj_dir,
        WorkspaceCoordinator(), trace_dir=trace_dir, run_id="run-shared",
        sandbox_id="sandbox-shared",
    )
    return session


def test_each_worker_gets_an_executor_bound_to_its_own_identity(monkeypatch, tmp_path):
    # The runtime keys each consumer's cursor over the event log by agent_id,
    # so two workers sharing one would split events between them. Binding the
    # identity to the executor is what keeps them distinct -- and means a call
    # site cannot forget it.
    _run_manager_and_workers(monkeypatch, tmp_path)

    bound = [ex.agent_id for ex in _FakeWorkerExecutor.created]
    assert bound == ["worker-t1"], "the worker's executor must carry its identity"
    assert len(set(bound)) == len(bound), "identities must be unique per worker"


def test_coordinated_workers_leave_read_before_edit_to_waggle(monkeypatch, tmp_path):
    # With Waggle mounted on the worker's executor, a guardrail seat enforcing
    # the same rule would state it to the model twice -- and Waggle's message is
    # the better one (it names the version the worker is stale against).
    _run_manager_and_workers(monkeypatch, tmp_path)   # waggle_state passed in

    manager, worker = _FakeAgent.created
    seats = {i.name: i for i in worker.kwargs["pipeline"].interceptors}
    assert seats["GuardrailInterceptor"].read_before_edit is False
    # The other guardrail still applies: turning one rule off is not turning
    # the seat off.
    assert seats["GuardrailInterceptor"].edit_streak_limit > 0
    assert "TruncateInterceptor" in seats
    # The manager is uncoordinated (read-only exploration), so it keeps the rule.
    assert manager.kwargs.get("pipeline") is None


def test_manager_bookkeeping_is_not_attributed_to_the_manager(monkeypatch, tmp_path):
    # The harness reads the plan file and resets the repo through the same
    # session. Those are its own actions: attributing them to the manager
    # would put them in the manager's event stream and consume its cursor.
    session = _run_manager_and_workers(monkeypatch, tmp_path)

    bookkeeping = [cmd for who, cmd in session.calls if who == "harness"]
    assert bookkeeping, "the harness should have made bookkeeping calls"
    assert any("ash_plan" in cmd for cmd in bookkeeping), \
        "reading the plan file is the harness's own action"

    # The manager's own work went through its bound channel, so it appears
    # under "manager" -- proving the two are actually distinguishable here.
    assert ("manager", "work iid-manager") in session.calls
    assert "manager" not in {who for who, cmd in session.calls
                             if "ash_plan" in cmd}


def test_manager_and_workers_share_run_and_sandbox_identity(monkeypatch, tmp_path):
    _FakeAgent.created = []
    monkeypatch.setattr(manager_worker, "AshAgent", _FakeAgent)
    monkeypatch.setattr(manager_worker, "_WorkerExecutor", _FakeWorkerExecutor)

    session = _RecordingSession(output=_PLAN_OUTPUT)
    harness = manager_worker.ManagerWorkerHarness({})
    run_id = "run-shared"
    sandbox_id = "sandbox-shared"
    trace_dir = tmp_path / "traces"
    traj_dir = tmp_path / "trajectories"

    subtasks = harness._run_manager(
        session, {}, "problem", 2, 10, "iid", traj_dir,
        trace_dir=trace_dir, run_id=run_id, sandbox_id=sandbox_id,
    )
    cfg = AgentConfig()
    harness._run_workers(
        "http://sandbox", cfg, "problem", subtasks, "iid", traj_dir,
        WorkspaceCoordinator(), trace_dir=trace_dir, run_id=run_id,
        sandbox_id=sandbox_id,
    )

    manager, worker = _FakeAgent.created
    assert manager.identity == {
        "trace_dir": trace_dir,
        "run_id": run_id,
        "agent_id": "manager",
        "sandbox_id": sandbox_id,
    }
    assert worker.identity == {
        "trace_dir": trace_dir,
        "run_id": run_id,
        "agent_id": "worker-t1",
        "sandbox_id": sandbox_id,
    }
    assert isinstance(worker.executor, CoordinatedExecutor)
    assert worker.executor._agent == worker.identity["agent_id"]
    assert worker.executor._sbx == sandbox_id


def test_best_of_n_candidate_uses_shared_run_and_own_sandbox(monkeypatch, tmp_path):
    _FakeAgent.created = []
    monkeypatch.setattr(best_of_n, "AshAgent", _FakeAgent)

    class CandidateSession(_RecordingSession):
        def __init__(self, **kwargs):
            super().__init__()
            self.sandbox_id = "sandbox-c2"

        def create(self, image):
            return True

        def get_patch(self):
            return ""

        def destroy(self):
            pass

    monkeypatch.setattr(best_of_n, "AshSession", CandidateSession)

    harness = best_of_n.BestOfNHarness({"selection": "heuristic"})
    trace_dir = tmp_path / "traces"
    harness._run_candidate(
        2,
        {"instance_id": "iid", "problem_statement": "problem"},
        "image",
        tmp_path / "trajectories",
        trace_dir=trace_dir,
        run_id="run-shared",
    )

    (candidate,) = _FakeAgent.created
    assert candidate.identity == {
        "trace_dir": trace_dir,
        "run_id": "run-shared",
        "agent_id": "candidate-2",
        "sandbox_id": "sandbox-c2",
    }
