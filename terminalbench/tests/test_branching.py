from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from terminalbench import branching


SNAPSHOT = "01a0a960-85f0-75b1-9d54-14da424d2d61"


def context_file(tmp_path: Path) -> Path:
    parent = tmp_path / "parent.jsonl"
    parent.write_text('{"type":"run.started","slot":"mini-swe-agent"}\n')
    context = {
        "branch_guidance": "assistant-turn", "checkpoint_mode": "full",
        "snapshot_id": SNAPSHOT, "base_image": "image@sha256:digest",
        "image_config": {"User": "nobody", "WorkingDir": "/work", "Env": []},
        "parent_journal": str(parent),
        "parent_journal_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "step": 2, "native_session_id": "session", "conversation_cut": "turn-id",
        "assistant_turn": {"role": "assistant", "content": "", "tool_calls": []},
    }
    path = tmp_path / "branch-context.json"
    path.write_text(json.dumps(context))
    return path


def snapshot_environment(tmp_path: Path, context: Path):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    runtime = tmp_path / "runtime"
    runtime.touch()
    return branching.SnapshotEnvironment(
        environment_dir=environment_dir, environment_name="task", session_id="branch",
        task_env_config=EnvironmentConfig(docker_image="image@sha256:digest"),
        trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(runtime),
        branch_context=str(context), checkpoint_mode="full",
    )


def test_snapshot_restore_uses_selected_state_and_refuses_rebuild(tmp_path):
    context = context_file(tmp_path)
    environment = snapshot_environment(tmp_path, context)
    assert environment._prepare_image(False) == (
        SNAPSHOT, {"User": "nobody", "WorkingDir": "/work", "Env": []})
    with pytest.raises(ValueError, match="restore its snapshot"):
        environment._prepare_image(True)


def test_branch_context_refuses_changed_parent_and_disk_only(tmp_path):
    path = context_file(tmp_path)
    data = json.loads(path.read_text())
    data["checkpoint_mode"] = "disk_only"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="full snapshot"):
        branching.read_branch_context(path)
    data["checkpoint_mode"] = "full"
    path.write_text(json.dumps(data))
    Path(data["parent_journal"]).write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        branching.read_branch_context(path)


def test_branch_mini_rechecks_snapshot_and_injects_only_exact_prefix(tmp_path, monkeypatch):
    path = context_file(tmp_path)
    environment = snapshot_environment(tmp_path, path)
    environment.session = SimpleNamespace()
    environment.backend = {"backend": "microvm"}
    environment.image = SNAPSHOT
    agent = branching.BranchMini(
        logs_dir=tmp_path / "agent", model_name="fixture",
        inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_KEY",
        branch_context=str(path),
    )
    point = SimpleNamespace(snapshot_id=SNAPSHOT, session_ckpt="session")
    monkeypatch.setattr(branching, "available_branch_points", lambda _: {2: point})
    reference = {"slot": "mini-swe-agent", "session_id": "session", "cut": "turn-id",
                 "path": "native.jsonl", "byte_length": 10, "sha256": "digest"}
    monkeypatch.setattr(branching, "reference_at", lambda *_: reference)
    monkeypatch.setattr(branching, "load_prefix", lambda _: [{"type": "mini.session"}])
    monkeypatch.setattr(branching, "actor_tools_at", lambda *_: [])
    monkeypatch.setattr(branching, "validate_assistant_turn", lambda turn, **_: turn)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = agent._make_spec("must not repeat parent prompt", workspace, tmp_path / "child.jsonl", environment)
    assert spec.prompt == "" and spec.slot == "mini-swe-agent" and spec.fork
    assert spec.extra["native_prefix"] == reference
    assert spec.extra["assistant_turn"] == json.loads(path.read_text())["assistant_turn"]
    assert spec.origin["snapshot_id"] == SNAPSHOT

    point.snapshot_id = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        agent._make_spec("", workspace, tmp_path / "child.jsonl", environment)
