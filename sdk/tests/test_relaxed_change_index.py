from __future__ import annotations

import hashlib

from ash_sandbox.relaxed_change_index import RelaxedChangeIndex


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def tr(tool: str, args: dict, content: str = "ok", success: bool = True) -> dict:
    return {
        "role": "tool_result",
        "tool_name": tool,
        "tool_args": args,
        "content": content,
        "success": success,
    }


def test_safe_read_order_collapses_to_same_environment_target(tmp_path):
    db = tmp_path / "relaxed.sqlite"
    reads_a = [
        tr("grep_files", {"pattern": "TODO", "path": "/testbed"}, "a.py:1"),
        tr("text_editor", {"command": "view", "path": "/testbed/a.py"}, "contents"),
    ]
    reads_b = list(reversed(reads_a))

    with RelaxedChangeIndex(db) as index:
        stored = index.register(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=digest("workspace"),
            messages=reads_a,
            reference="snapshot:read-a",
            exact_prefix_hash=digest("prefix-a"),
            step_id=2,
        )
        match = index.lookup(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=digest("workspace"),
            messages=reads_b,
        )

    assert match is not None
    assert match.convergence_key == stored.convergence_key
    assert match.reference == "snapshot:read-a"
    assert match.model_prefix_reusable is False
    assert match.kv_reuse is False


def test_different_workspace_state_never_matches(tmp_path):
    with RelaxedChangeIndex(tmp_path / "relaxed.sqlite") as index:
        index.register(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=digest("workspace-a"),
            messages=[],
            reference="snapshot:a",
        )
        assert index.lookup(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=digest("workspace-b"),
            messages=[],
        ) is None


def test_structured_edit_histories_can_converge_only_via_same_final_workspace(tmp_path):
    direct = [
        tr(
            "text_editor",
            {"command": "write", "path": "/testbed/a.py", "file_text": "final"},
            "written",
        )
    ]
    two_step = [
        tr(
            "text_editor",
            {"command": "write", "path": "/testbed/a.py", "file_text": "temp"},
            "written",
        ),
        tr(
            "text_editor",
            {"command": "str_replace", "path": "/testbed/a.py", "old_str": "temp", "new_str": "final"},
            "replaced",
        ),
    ]
    final_digest = digest("same-final-workspace")

    with RelaxedChangeIndex(tmp_path / "relaxed.sqlite") as index:
        stored = index.register(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=final_digest,
            messages=direct,
            reference="snapshot:direct",
        )
        match = index.lookup(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=final_digest,
            messages=two_step,
        )

    assert match is not None
    assert match.convergence_key == stored.convergence_key


def test_external_barrier_history_prevents_false_convergence(tmp_path):
    clean = []
    shell = [tr("shell", {"command": "python -c 'print(1)'"}, "1")]
    same_workspace = digest("workspace")

    with RelaxedChangeIndex(tmp_path / "relaxed.sqlite") as index:
        index.register(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=same_workspace,
            messages=clean,
            reference="snapshot:clean",
        )
        assert index.lookup(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=same_workspace,
            messages=shell,
        ) is None


def test_task_and_environment_identity_are_isolated(tmp_path):
    workspace = digest("workspace")
    with RelaxedChangeIndex(tmp_path / "relaxed.sqlite") as index:
        index.register(
            task_id="task-a",
            env_fingerprint="env-a",
            workspace_digest=workspace,
            messages=[],
            reference="snapshot:a",
        )
        assert index.lookup(
            task_id="task-b",
            env_fingerprint="env-a",
            workspace_digest=workspace,
            messages=[],
        ) is None
        assert index.lookup(
            task_id="task-a",
            env_fingerprint="env-b",
            workspace_digest=workspace,
            messages=[],
        ) is None


def test_safe_shell_reads_can_be_relaxed_only_when_explicitly_enabled(tmp_path):
    shell_read = [
        tr(
            "shell",
            {"command": "cd /testbed && cat django/conf/global_settings.py"},
            "contents",
        )
    ]
    workspace = digest("workspace")

    with RelaxedChangeIndex(tmp_path / "relaxed.sqlite") as index:
        index.register(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=workspace,
            messages=[],
            reference="snapshot:root",
            allow_safe_shell=True,
        )
        assert index.lookup(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=workspace,
            messages=shell_read,
            allow_safe_shell=True,
        ) is not None
        assert index.lookup(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=workspace,
            messages=shell_read,
            allow_safe_shell=False,
        ) is None


def test_register_updates_same_convergence_target_in_place(tmp_path):
    with RelaxedChangeIndex(tmp_path / "relaxed.sqlite") as index:
        first = index.register(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=digest("workspace"),
            messages=[],
            reference="snapshot:old",
            step_id=1,
        )
        second = index.register(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=digest("workspace"),
            messages=[],
            reference="snapshot:new",
            step_id=3,
            metadata={"reason": "newer"},
        )
        match = index.lookup(
            task_id="task",
            env_fingerprint="env-v1",
            workspace_digest=digest("workspace"),
            messages=[],
        )
        stats = index.stats()

    assert first.convergence_key == second.convergence_key
    assert match is not None
    assert match.reference == "snapshot:new"
    assert match.step_id == 3
    assert match.metadata == {"reason": "newer"}
    assert stats["target_count"] == 1
