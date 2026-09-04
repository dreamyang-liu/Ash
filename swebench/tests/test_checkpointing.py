import asyncio
from types import SimpleNamespace

import pytest

from ash_sandbox import CheckpointStore, ExactPrefixIndex, RelaxedChangeIndex, TrajectoryCache
from swebench.checkpointing import SessionCheckpointer
from swebench.sandbox import AshSession


class _SnapshotPool:
    def __init__(self):
        self.calls = []

    def supports_snapshot(self):
        return True

    def supports_fork(self):
        return True

    async def snapshot(self, sandbox, name=None):
        self.calls.append((sandbox.sandbox_id, name))
        return f"snap-{len(self.calls)}"


class _NoSnapshotPool:
    def supports_snapshot(self):
        return False

    def supports_fork(self):
        return False


class _RestorePool:
    def __init__(self):
        self.restored = []

    def supports_restore(self):
        return True

    async def restore(self, snapshot_id, *, agent_id=""):
        self.restored.append((snapshot_id, agent_id))
        return _Sandbox(f"restored-{snapshot_id}")

    async def close(self):
        pass


class _Sandbox:
    def __init__(self, sid="sb-1"):
        self.sandbox_id = sid


def test_ash_session_exposes_snapshot_capability_without_private_pool_calls():
    session = AshSession(quiet=True)
    pool = _SnapshotPool()
    session._pool = pool
    session._sandbox = _Sandbox("sb-parent")

    assert session.supports_snapshot is True
    assert session.supports_fork is True
    assert session.snapshot("turn-3") == "snap-1"
    assert pool.calls == [("sb-parent", "turn-3")]


def test_ash_session_refuses_snapshot_on_unsupported_backend():
    session = AshSession(quiet=True)
    session._pool = _NoSnapshotPool()
    session._sandbox = _Sandbox()

    assert session.supports_snapshot is False
    with pytest.raises(NotImplementedError, match="durable snapshots"):
        session.snapshot()


def test_restore_builds_an_independent_pool_for_the_branch(monkeypatch):
    import swebench.sandbox as sandbox_module

    pools = []

    def fake_build_pool(*_args, **_kwargs):
        pool = _RestorePool()
        pools.append(pool)
        return pool

    monkeypatch.setattr(sandbox_module, "build_pool", fake_build_pool)
    session = AshSession(quiet=True, backend={"backend": "microvm"})

    assert session.restore("snap-42", agent_id="branch-a") is True
    assert session.sandbox_id == "restored-snap-42"
    assert len(pools) == 1
    assert pools[0].restored == [("snap-42", "branch-a")]


def test_session_checkpointer_pairs_exact_messages_with_snapshot(tmp_path):
    session = AshSession(quiet=True)
    pool = _SnapshotPool()
    session._pool = pool
    session._sandbox = _Sandbox("sb-prefix")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]

    with CheckpointStore(tmp_path / "cp.sqlite3") as store:
        hook = SessionCheckpointer(
            session=session,
            store=store,
            task_id="nl2repo/math-verify",
            trajectory_id="traj-1",
            env_fingerprint="ash-base@v1",
            every_n_turns=2,
            metadata={"benchmark": "nl2repo"},
        )

        hook(1, messages)
        assert hook.records == []
        hook(2, messages)

        record = hook.last_record
        assert record is not None
        assert record.snapshot_id == "snap-1"
        assert record.step_id == 2
        assert record.trajectory_prefix == messages
        assert record.metadata["sandbox_id"] == "sb-prefix"
        assert record.metadata["benchmark"] == "nl2repo"

        loaded = store.get(
            task_id="nl2repo/math-verify",
            trajectory_prefix=messages,
            env_fingerprint="ash-base@v1",
        )
        assert loaded == record

    assert pool.calls == [("sb-prefix", "checkpoint-nl2repo-math-verify-step-2")]


def test_session_checkpointer_uses_global_step_offset(tmp_path):
    session = AshSession(quiet=True)
    pool = _SnapshotPool()
    session._pool = pool
    session._sandbox = _Sandbox("sb-branch")

    with CheckpointStore(tmp_path / "branch.sqlite3") as store:
        hook = SessionCheckpointer(
            session=session,
            store=store,
            task_id="nl2repo/demo",
            every_n_turns=5,
            step_offset=10,
        )
        hook(5, [{"role": "user", "content": "continued"}])
        assert hook.last_record is not None
        assert hook.last_record.step_id == 15

    assert pool.calls == [("sb-branch", "checkpoint-nl2repo-demo-step-15")]


def test_session_checkpointer_registers_exact_and_relaxed_cache_keys(tmp_path):
    session = AshSession(quiet=True)
    pool = _SnapshotPool()
    session._pool = pool
    session._sandbox = _Sandbox("sb-cache")
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "text_editor",
                    "arguments": '{"command":"view","path":"/testbed/a.py"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
    ]
    workspace_digest = "a" * 64

    with (
        CheckpointStore(tmp_path / "cp.sqlite3") as store,
        ExactPrefixIndex(tmp_path / "exact.sqlite3") as exact,
        RelaxedChangeIndex(tmp_path / "relaxed.sqlite3") as relaxed,
    ):
        cache = TrajectoryCache(exact, relaxed)
        hook = SessionCheckpointer(
            session=session,
            store=store,
            task_id="task/cache",
            trajectory_id="traj-cache",
            env_fingerprint="env-v1",
            trajectory_cache=cache,
            trajectory_cache_mode="relaxed",
            workspace_digest_provider=lambda: workspace_digest,
        )
        hook(1, messages)

        record = hook.last_record
        assert record is not None
        assert len(hook.cache_registrations) == 1
        registration = hook.cache_registrations[0]
        assert registration.exact.reference == record.snapshot_id
        assert registration.relaxed.reference == record.snapshot_id
        assert record.metadata["workspace_digest"] == workspace_digest
        assert record.metadata["cache_exact_chain_hash"] == registration.exact.cursor.chain_hash
        assert record.metadata["cache_relaxed_convergence_key"] == registration.relaxed.convergence_key
        assert record.metadata["workspace_fingerprint_ms"] >= 0
        assert record.metadata["trajectory_cache_registration_ms"] >= 0

        exact_hit = cache.lookup(
            mode="exact",
            task_id="task/cache",
            env_fingerprint="env-v1",
            trajectory_prefix=messages,
        )
        assert exact_hit is not None and exact_hit.reference == record.snapshot_id

        alternate_read = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "grep_files",
                        "arguments": '{"pattern":"TODO","path":"/testbed"}',
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "a.py:1"},
        ]
        relaxed_hit = cache.lookup(
            mode="relaxed",
            task_id="task/cache",
            env_fingerprint="env-v1",
            trajectory_prefix=alternate_read,
            workspace_digest=workspace_digest,
            messages=alternate_read,
        )
        assert relaxed_hit is not None
        assert relaxed_hit.kind == "relaxed"
        assert relaxed_hit.reference == record.snapshot_id


def test_relaxed_projection_checkpoint_does_not_require_workspace_fingerprint(tmp_path):
    session = AshSession(quiet=True)
    pool = _SnapshotPool()
    session._pool = pool
    session._sandbox = _Sandbox("sb-relaxed-projection")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-read",
                "type": "function",
                "function": {
                    "name": "grep_files",
                    "arguments": '{"pattern":"TODO","path":"/testbed"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-read", "content": "a.py:1"},
    ]
    with (
        CheckpointStore(tmp_path / "cp.sqlite3") as store,
        ExactPrefixIndex(tmp_path / "exact.sqlite3") as exact,
        RelaxedChangeIndex(tmp_path / "relaxed.sqlite3") as relaxed,
    ):
        cache = TrajectoryCache(exact, relaxed)
        hook = SessionCheckpointer(
            session=session,
            store=store,
            task_id="task",
            env_fingerprint="env-v1",
            trajectory_cache=cache,
            trajectory_cache_mode="relaxed",
        )
        hook(1, messages)
        record = hook.last_record
        assert record is not None
        registration = hook.cache_registrations[0]
        assert registration.projection.reference == record.snapshot_id
        assert registration.relaxed is None
        assert record.metadata["workspace_digest"] is None
        assert record.metadata["workspace_fingerprint_ms"] == 0.0
        assert record.metadata["cache_relaxed_projection_state_hash"] == registration.projection.state_hash
        assert record.metadata["cache_relaxed_convergence_key"] is None


def test_relaxed_projection_coalesces_state_equivalent_physical_snapshots(tmp_path):
    session = AshSession(quiet=True)
    pool = _SnapshotPool()
    session._pool = pool
    session._sandbox = _Sandbox("sb-coalesce")
    first = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "grep_files",
                    "arguments": '{"pattern":"foo","path":"/testbed"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "a.py:1"},
    ]
    second = [
        *first,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "text_editor",
                    "arguments": '{"command":"view","path":"/testbed/a.py"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "contents"},
    ]

    with (
        CheckpointStore(tmp_path / "cp.sqlite3") as store,
        ExactPrefixIndex(tmp_path / "exact.sqlite3") as exact,
        RelaxedChangeIndex(tmp_path / "relaxed.sqlite3") as relaxed,
    ):
        cache = TrajectoryCache(exact, relaxed)
        hook = SessionCheckpointer(
            session=session,
            store=store,
            task_id="task/coalesce",
            env_fingerprint="env-v1",
            trajectory_cache=cache,
            trajectory_cache_mode="relaxed",
        )
        hook(1, first)
        first_record = hook.last_record
        assert first_record is not None
        hook(2, second)
        second_record = hook.last_record
        assert second_record is not None

        assert len(pool.calls) == 1
        assert len(hook.records) == 2
        assert first_record.snapshot_id == second_record.snapshot_id == "snap-1"
        assert first_record.metadata["snapshot_reused"] is False
        assert second_record.metadata["snapshot_reused"] is True
        assert second_record.metadata["snapshot_reuse_kind"] == "relaxed"
        assert second_record.metadata["snapshot_reuse_relaxed_tier"] == "projection"
        assert second_record.metadata["snapshot_reuse_source_step"] == 1
        assert second_record.metadata["snapshot_ms"] == 0.0
        assert exact.stats()["target_count"] == 2
        assert relaxed.stats()["projection_target_count"] == 1


def test_exact_cache_checkpoint_does_not_pay_workspace_fingerprint_cost(tmp_path):
    session = AshSession(quiet=True)
    pool = _SnapshotPool()
    session._pool = pool
    session._sandbox = _Sandbox("sb-exact")
    messages = [{"role": "user", "content": "prefix"}]

    with (
        CheckpointStore(tmp_path / "cp.sqlite3") as store,
        ExactPrefixIndex(tmp_path / "exact.sqlite3") as exact,
        RelaxedChangeIndex(tmp_path / "relaxed.sqlite3") as relaxed,
    ):
        cache = TrajectoryCache(exact, relaxed)
        hook = SessionCheckpointer(
            session=session,
            store=store,
            task_id="task/exact",
            env_fingerprint="env-v1",
            trajectory_cache=cache,
            trajectory_cache_mode="exact",
        )
        hook(1, messages)
        record = hook.last_record
        assert record is not None
        assert record.metadata["trajectory_cache_mode"] == "exact"
        assert record.metadata["workspace_digest"] is None
        assert record.metadata["workspace_fingerprint_ms"] == 0.0
        assert record.metadata["cache_exact_chain_hash"]
        assert record.metadata["cache_relaxed_convergence_key"] is None
        assert relaxed.stats()["target_count"] == 0
