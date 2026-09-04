from __future__ import annotations

import hashlib

import pytest

from ash_sandbox.prefix_index import ExactPrefixIndex
from ash_sandbox.relaxed_change_index import RelaxedChangeIndex
from ash_sandbox.trajectory_cache import TrajectoryCache


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


def cache(tmp_path):
    exact = ExactPrefixIndex(tmp_path / "exact.sqlite")
    relaxed = RelaxedChangeIndex(tmp_path / "relaxed.sqlite")
    return exact, relaxed, TrajectoryCache(exact, relaxed)


def test_none_mode_never_returns_a_cache_hit(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    try:
        assert c.lookup(
            mode="none",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[],
        ) is None
    finally:
        exact.close()
        relaxed.close()


def test_exact_mode_returns_exact_registered_prefix(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    read = tr("grep_files", {"pattern": "TODO", "path": "/testbed"}, "a.py:1")
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read],
            workspace_digest=digest("workspace"),
            messages=[read],
            reference="snapshot:1",
            step_id=1,
        )
        hit = c.lookup(
            mode="exact",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read],
        )
        assert hit is not None
        assert hit.kind == "exact"
        assert hit.reference == "snapshot:1"
        assert hit.exact_history_match is True
        assert hit.kv_reuse is False
    finally:
        exact.close()
        relaxed.close()


def test_relaxed_mode_still_prefers_full_depth_exact_hit(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    read = tr("text_editor", {"command": "view", "path": "/testbed/a.py"}, "contents")
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read],
            workspace_digest=digest("workspace"),
            messages=[read],
            reference="snapshot:exact",
        )
        hit = c.lookup(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read],
            workspace_digest=digest("workspace"),
            messages=[read],
        )
        assert hit is not None and hit.kind == "exact"
    finally:
        exact.close()
        relaxed.close()


def test_relaxed_mode_matches_different_safe_read_history(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    read_a = tr("grep_files", {"pattern": "TODO", "path": "/testbed"}, "a.py:1")
    read_b = tr("text_editor", {"command": "view", "path": "/testbed/a.py"}, "contents")
    workspace = digest("same-workspace")
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read_a],
            workspace_digest=workspace,
            messages=[read_a],
            reference="snapshot:a",
            step_id=1,
        )
        assert c.lookup(
            mode="exact",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read_b],
        ) is None
        hit = c.lookup(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read_b],
            workspace_digest=workspace,
            messages=[read_b],
        )
        assert hit is not None
        assert hit.kind == "relaxed"
        assert hit.reference == "snapshot:a"
        assert hit.exact_history_match is False
        assert hit.model_prefix_reusable is False
    finally:
        exact.close()
        relaxed.close()


def test_relaxed_current_state_beats_shallower_exact_prefix(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    common = tr("grep_files", {"pattern": "foo", "path": "/testbed"}, "a.py:1")
    extra = tr("text_editor", {"command": "view", "path": "/testbed/a.py"}, "contents")
    workspace = digest("same-workspace")
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common],
            workspace_digest=workspace,
            messages=[common],
            reference="snapshot:common",
            step_id=1,
        )
        exact_hit = c.lookup(
            mode="exact",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common, extra],
        )
        assert exact_hit is not None and exact_hit.kind == "exact"
        hit = c.lookup(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common, extra],
            workspace_digest=workspace,
            messages=[common, extra],
        )
        assert hit is not None
        assert hit.kind == "relaxed"
        assert hit.reference == "snapshot:common"
    finally:
        exact.close()
        relaxed.close()


def test_barrier_mismatch_falls_back_to_shallower_exact_hit(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    common = tr("grep_files", {"pattern": "foo", "path": "/testbed"}, "a.py:1")
    barrier = tr("shell", {"command": "python -c 'print(1)'"}, "1")
    workspace = digest("workspace")
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common],
            workspace_digest=workspace,
            messages=[common],
            reference="snapshot:common",
        )
        hit = c.lookup(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common, barrier],
            workspace_digest=workspace,
            messages=[common, barrier],
        )
        assert hit is not None
        assert hit.kind == "exact"
        assert hit.reference == "snapshot:common"
    finally:
        exact.close()
        relaxed.close()


def test_relaxed_lookup_without_workspace_digest_uses_projection_tier_only(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    read_a = tr("grep_files", {"pattern": "foo", "path": "/testbed"}, "a.py:1")
    read_b = tr("grep_files", {"pattern": "bar", "path": "/testbed"}, "b.py:2")
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read_a],
            messages=[read_a],
            reference="snapshot:read-only",
        )
        hit = c.lookup(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[read_b],
            messages=[read_b],
        )
        assert hit is not None
        assert hit.kind == "relaxed"
        assert hit.relaxed_tier == "projection"
        assert hit.reference == "snapshot:read-only"

        miss = c.lookup(
            mode="relaxed",
            task_id="other-task",
            env_fingerprint="env",
            trajectory_prefix=[read_b],
            messages=[read_b],
        )
        assert miss is None
    finally:
        exact.close()
        relaxed.close()


def test_lookup_materialized_state_accepts_current_projection_but_not_shallow_exact(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    common = tr("grep_files", {"pattern": "foo", "path": "/testbed"}, "a.py:1")
    extra_read = tr("text_editor", {"command": "view", "path": "/testbed/a.py"}, "contents")
    barrier = tr("shell", {"command": "python -c 'print(1)'"}, "1")
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common],
            messages=[common],
            reference="snapshot:common",
            step_id=1,
        )

        projection_hit = c.lookup_materialized_state(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common, extra_read],
            messages=[common, extra_read],
        )
        assert projection_hit is not None
        assert projection_hit.kind == "relaxed"
        assert projection_hit.reference == "snapshot:common"

        # Normal relaxed lookup may fall back to a shallower exact checkpoint so a
        # caller can replay the suffix. Pre-snapshot coalescing must never do that.
        normal = c.lookup(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common, barrier],
            messages=[common, barrier],
        )
        assert normal is not None and normal.kind == "exact"
        assert c.lookup_materialized_state(
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=[common, barrier],
            messages=[common, barrier],
        ) is None
    finally:
        exact.close()
        relaxed.close()


def test_lookup_materialized_state_accepts_full_depth_exact_hit(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    prefix = [tr("grep_files", {"pattern": "foo", "path": "/testbed"}, "a.py:1")]
    try:
        c.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=prefix,
            messages=prefix,
            reference="snapshot:exact",
        )
        hit = c.lookup_materialized_state(
            mode="exact",
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=prefix,
        )
        assert hit is not None
        assert hit.kind == "exact"
        assert hit.reference == "snapshot:exact"
    finally:
        exact.close()
        relaxed.close()


def test_unknown_cache_mode_is_rejected(tmp_path):
    exact, relaxed, c = cache(tmp_path)
    try:
        with pytest.raises(ValueError, match="unknown cache mode"):
            c.lookup(
                mode="bogus",  # type: ignore[arg-type]
                task_id="task",
                env_fingerprint="env",
                trajectory_prefix=[],
            )
    finally:
        exact.close()
        relaxed.close()
