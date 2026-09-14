from pathlib import Path
from types import SimpleNamespace

from harness.core.result import CommandOutcome, ToolResult
from harness.execution.pipeline import CallContext, ToolPipeline
from swebench_pro.batch import summary
from swebench_pro.limits import OfficialToolBudget
from swebench_pro.worker import save


def test_official_command_limit_is_consumed_by_execution_pipeline():
    budget = OfficialToolBudget()
    calls = []
    pipeline = ToolPipeline([budget])
    for args in ({"command": "test"}, {"command": "test", "timeout": 900}, {"command": "test", "timeout": 60}):
        pipeline.execute(CallContext("actor", "sandbox", "shell", args, {}),
                         lambda name, effective: calls.append(effective) or ToolResult(True, ""))
    assert [call["timeout"] for call in calls] == [450, 450, 60]
    assert budget.dirty


def test_official_total_and_consecutive_timeout_limits():
    budget = OfficialToolBudget()
    pipeline = ToolPipeline([budget])
    for index in range(3):
        pipeline.execute(CallContext("actor", "sandbox", "shell", {"command": "test"}, {}),
                         lambda *args: ToolResult(False, "", outcome=CommandOutcome(exit_code=-1, timed_out=True)))
    assert budget.exhausted == "official consecutive tool timeout limit exceeded"
    budget.consecutive = 0
    budget.elapsed = 1801
    assert budget.exhausted == "official total tool execution budget exceeded"


def test_batch_summary_preserves_unmeasured_and_queued_tasks(tmp_path: Path):
    manifest = {"tasks": [{"index": index, "id": f"task-{index}"} for index in range(3)]}
    save(tmp_path / "shard-000/worker.json", {"task": "task-0", "finished_at": "now", "evidence_valid": True,
                                            "resolved": True, "usage": {"cost_usd": 1.5}})
    save(tmp_path / "shard-001/worker.json", {"task": "task-1", "finished_at": "now", "evidence_valid": False})
    report = summary(tmp_path, manifest)
    assert report["finished"] == 2 and report["valid_grades"] == 1
    assert report["resolved"] == 1 and report["resolved_lower_bound"] == 1 / 3
    assert report["held"] == ["task-1"] and report["final_resolved_rate"] is None
    assert report["phases"] == {"queued": 1}
    assert report["cost_usd"] == 1.5


def test_claude_pro_prompt_matches_shell_only_surface(tmp_path: Path):
    from swebench_pro.bench import SWEbenchPro

    bench = SWEbenchPro(SimpleNamespace(pro_repo=tmp_path))
    prompt = bench.prompt({"repo": "org/repo", "problem": "Fix it", "slot": "claude-code"})
    assert "text_editor" not in prompt and "shell" in prompt


def test_controller_prioritizes_pending_actor_canaries(tmp_path, monkeypatch):
    from swebench_pro import batch

    manifest = {"tasks": [{"index": index, "id": f"task-{index}", "canary": index != 0}
                          for index in range(3)], "max_workers": 24, "canaries": ["task-1", "task-2"]}
    save(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(batch, "api", lambda route: [])
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda root: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(batch.signal, "signal", lambda *args: None)
    launched = []

    def spawn(arguments, **kwargs):
        index = int(arguments[-1])
        launched.append(index)
        save(tmp_path / f"shard-{index:03d}/worker.json",
             {"task": f"task-{index}", "status": "completed", "finished_at": "now",
              "steps": 2, "snapshots": 1, "evidence_valid": True})
        return SimpleNamespace(pid=index + 100, poll=lambda: 0)

    monkeypatch.setattr(batch.subprocess, "Popen", spawn)
    assert batch.controller(tmp_path) == 0
    assert launched == [1, 2, 0]


def test_batch_launch_forwards_independent_network_flags(tmp_path, monkeypatch):
    import sys

    from swebench_pro import batch

    seen = {}
    root = tmp_path / "batch"

    def prepare(output, upstream, runtime, workers, model, **kwargs):
        seen.update(kwargs)
        output.mkdir()

    monkeypatch.setattr(batch, "prepare", prepare)
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=42))
    monkeypatch.setattr(sys, "argv", ["batch", "launch", "--out", str(root), "--pro-repo", str(tmp_path),
                                     "--runtime-bin", str(tmp_path / "runtime"),
                                     "--agent-network", "deny", "--verifier-network", "allow"])
    assert batch.main() == 0
    assert seen == {"agent_network": "deny", "verifier_network": "allow"}


def test_existing_batch_cannot_silently_override_frozen_network_policy(tmp_path, monkeypatch):
    import sys

    import pytest

    from swebench_pro import batch

    monkeypatch.setattr(sys, "argv", ["batch", "controller", "--out", str(tmp_path),
                                     "--agent-network", "allow"])
    monkeypatch.setattr(batch, "controller", lambda *args: pytest.fail("frozen batch was overridden"))
    with pytest.raises(SystemExit):
        batch.main()
