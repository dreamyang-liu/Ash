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


class _FakeAgent:
    created = []

    def __init__(self, config, executor, **identity):
        self.config = config
        self.executor = executor
        self.identity = identity
        self.cost = CostTracker()
        self.trajectory = _FakeTrajectory()
        self.stream = True
        self.created.append(self)

    def set_tools_schema(self, schema) -> None:
        self.schema = schema

    def run(self, task: str, instance_id: str = "") -> str:
        self.task = task
        self.instance_id = instance_id
        return "completed"


class _FakeWorkerExecutor:
    def __init__(self, url: str):
        self.url = url
        self.closed = False

    def __call__(self, name: str, args: dict) -> ToolResult:
        return ToolResult(success=True, output="")

    def close(self) -> None:
        self.closed = True


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


def test_manager_and_workers_share_run_and_sandbox_identity(monkeypatch, tmp_path):
    _FakeAgent.created = []
    monkeypatch.setattr(manager_worker, "AshAgent", _FakeAgent)
    monkeypatch.setattr(manager_worker, "_WorkerExecutor", _FakeWorkerExecutor)

    class ManagerSession:
        def execute(self, name, args):
            return ToolResult(
                success=True,
                output='{"subtasks":[{"id":"t1","description":"one","files":[]}]}'
            )

    harness = manager_worker.ManagerWorkerHarness({})
    run_id = "run-shared"
    sandbox_id = "sandbox-shared"
    trace_dir = tmp_path / "traces"
    traj_dir = tmp_path / "trajectories"

    subtasks = harness._run_manager(
        ManagerSession(), {}, "problem", 2, 10, "iid", traj_dir,
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

    class CandidateSession:
        def __init__(self, **kwargs):
            self.sandbox_id = "sandbox-c2"

        def create(self, image):
            return True

        def execute(self, name, args):
            return ToolResult(success=True, output="")

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
