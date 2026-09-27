from pathlib import Path
from types import SimpleNamespace
import pytest

pytest.importorskip("harbor")
from harbor.models.task.task import Task
from terminalbench.branch_eval import _trial_config


def test_high_reaches_harbor_actor_and_official_timeout(tmp_path):
    root = Path(__file__).resolve().parents[3]
    task_dir = root / "terminalbench/tests/agentenv_fixtures/shell-task"
    task = Task(task_dir)
    args = SimpleNamespace(task_dir=task_dir, dataset_digest="sha256:test", output=tmp_path,
        model="qwen3.8-27b", model_endpoint="http://localhost:9000", model_key_env="QWEN_API_KEY",
        max_output_tokens=64000, max_turns=300, runtime_bin=tmp_path / "runtime",
        sandbox_ttl=36000, server_url="http://localhost:8000", api_key_file=None,
        image_registry=None, reasoning_effort="high")
    config = _trial_config(args, task, name="parent", branch_context=None)
    assert config.agent.kwargs["reasoning_effort"] == "high"
    assert config.agent_timeout_multiplier == 1
    pro_task = SimpleNamespace(name="swebench-pro/example", config=task.config)
    pro_config = _trial_config(args, pro_task, name="parent", branch_context=None)
    assert pro_config.task.source == "scaleapi/SWE-bench_Pro-os/v2@sha256:test"
