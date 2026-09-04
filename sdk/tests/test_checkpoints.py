import asyncio

from ash_sandbox import (
    BranchManager,
    CheckpointStore,
    canonical_prefix,
    trajectory_prefix_chain_hashes,
    trajectory_prefix_hash,
)


class _ModelLike:
    def __init__(self, value):
        self.value = value

    def model_dump(self, mode="json"):
        assert mode == "json"
        return {"value": self.value}


class _Sandbox:
    def __init__(self, sid, agent_id=""):
        self.sandbox_id = sid
        self.agent_id = agent_id


class _SnapshotPool:
    def __init__(self):
        self.snapshots = []
        self.restores = []
        self.forks = []
        self.deleted_snapshots = []

    def supports_snapshot(self):
        return True

    def supports_restore(self):
        return True

    def supports_fork(self):
        return True

    def supports_snapshot_delete(self):
        return True

    async def snapshot(self, sandbox, name=None):
        snapshot_id = f"snap-{len(self.snapshots) + 1}"
        self.snapshots.append((sandbox.sandbox_id, name, snapshot_id))
        return snapshot_id

    async def restore(self, snapshot_id, *, agent_id=""):
        sid = f"restored-{len(self.restores) + 1}"
        self.restores.append((snapshot_id, agent_id, sid))
        return _Sandbox(sid, agent_id=agent_id)

    async def delete_snapshot(self, snapshot_id):
        self.deleted_snapshots.append(snapshot_id)

    async def fork(self, sandbox, count=1, agent_ids=None):
        ids = list(agent_ids or [])
        children = [
            _Sandbox(f"fork-{i + 1}", ids[i] if i < len(ids) else "")
            for i in range(count)
        ]
        self.forks.append((sandbox.sandbox_id, count, ids))
        return children


def test_prefix_hash_is_canonical_and_supports_model_objects():
    left = [
        {"role": "assistant", "tool_calls": [_ModelLike({"b": 2, "a": 1})]},
        {"role": "tool", "content": "ok"},
    ]
    right = [
        {"tool_calls": [_ModelLike({"a": 1, "b": 2})], "role": "assistant"},
        {"content": "ok", "role": "tool"},
    ]
    assert canonical_prefix(left) == canonical_prefix(right)
    assert trajectory_prefix_hash(left) == trajectory_prefix_hash(right)


def test_checkpoint_store_maps_task_and_prefix_to_snapshot(tmp_path):
    db = tmp_path / "checkpoints.sqlite3"
    prefix = [{"role": "user", "content": "build it"}]

    with CheckpointStore(db) as store:
        record = store.put(
            task_id="nl2repo/math-verify",
            trajectory_prefix=prefix,
            snapshot_id="snap-123",
            step_id=20,
            trajectory_id="traj-a",
            backend="MicroVMPool",
            env_fingerprint="ash-base@sha256:abc",
            metadata={"benchmark": "nl2repo"},
        )
        loaded = store.get(
            task_id="nl2repo/math-verify",
            trajectory_prefix=prefix,
            env_fingerprint="ash-base@sha256:abc",
        )
        assert loaded == record
        assert store.get(
            task_id="nl2repo/math-verify",
            trajectory_prefix=prefix,
            env_fingerprint="ash-base@sha256:different",
        ) is None

    # The index is durable across processes/connections; AgentENV owns the heavy
    # snapshot bytes, while SQLite only stores the logical lookup.
    with CheckpointStore(db) as store:
        loaded = store.get(
            task_id="nl2repo/math-verify",
            trajectory_prefix=prefix,
            env_fingerprint="ash-base@sha256:abc",
        )
        assert loaded is not None
        assert loaded.snapshot_id == "snap-123"
        assert loaded.step_id == 20


def test_same_prefix_in_different_tasks_does_not_collide(tmp_path):
    prefix = [{"role": "user", "content": "same prefix"}]
    with CheckpointStore(tmp_path / "cp.sqlite3") as store:
        a = store.put(
            task_id="task-a", trajectory_prefix=prefix,
            snapshot_id="snap-a", step_id=1,
        )
        b = store.put(
            task_id="task-b", trajectory_prefix=prefix,
            snapshot_id="snap-b", step_id=1,
        )
        assert a.prefix_hash == b.prefix_hash
        assert store.get(task_id="task-a", trajectory_prefix=prefix).snapshot_id == "snap-a"
        assert store.get(task_id="task-b", trajectory_prefix=prefix).snapshot_id == "snap-b"


def test_longest_exact_prefix_match_returns_deepest_reusable_checkpoint(tmp_path):
    steps = [{"step": i, "value": f"v{i}"} for i in range(1, 8)]
    with CheckpointStore(tmp_path / "longest.sqlite3") as store:
        for length in (1, 3, 5):
            store.put(
                task_id="task-prefix",
                trajectory_prefix=steps[:length],
                snapshot_id=f"snap-{length}",
                step_id=length,
                env_fingerprint="env-v1",
            )

        hit = store.longest_prefix_match(
            task_id="task-prefix",
            trajectory_prefix=steps,
            env_fingerprint="env-v1",
        )
        assert hit is not None
        assert hit.snapshot_id == "snap-5"
        assert hit.prefix_length == 5


def test_longest_exact_prefix_match_stops_before_divergence(tmp_path):
    shared = [{"step": i} for i in range(1, 4)]
    stored_tail = shared + [{"step": 4, "branch": "old"}, {"step": 5}]
    query = shared + [{"step": 4, "branch": "new"}, {"step": 5}]
    with CheckpointStore(tmp_path / "diverge.sqlite3") as store:
        store.put(
            task_id="task-prefix",
            trajectory_prefix=shared,
            snapshot_id="snap-3",
            step_id=3,
            env_fingerprint="env-v1",
        )
        store.put(
            task_id="task-prefix",
            trajectory_prefix=stored_tail,
            snapshot_id="snap-5",
            step_id=5,
            env_fingerprint="env-v1",
        )
        hit = store.longest_prefix_match(
            task_id="task-prefix",
            trajectory_prefix=query,
            env_fingerprint="env-v1",
        )
        assert hit is not None
        assert hit.snapshot_id == "snap-3"


def test_longest_exact_prefix_match_respects_environment_identity(tmp_path):
    prefix = [{"step": 1}, {"step": 2}]
    with CheckpointStore(tmp_path / "env.sqlite3") as store:
        store.put(
            task_id="task-prefix",
            trajectory_prefix=prefix,
            snapshot_id="snap-env-a",
            step_id=2,
            env_fingerprint="env-a",
        )
        assert store.longest_prefix_match(
            task_id="task-prefix",
            trajectory_prefix=prefix + [{"step": 3}],
            env_fingerprint="env-b",
        ) is None


def test_prefix_chain_hashes_are_incremental_and_canonical():
    a = [{"b": 2, "a": 1}, {"tool": {"y": 2, "x": 1}}]
    b = [{"a": 1, "b": 2}, {"tool": {"x": 1, "y": 2}}]
    hashes_a = trajectory_prefix_chain_hashes(a)
    hashes_b = trajectory_prefix_chain_hashes(b)
    assert hashes_a == hashes_b
    assert len(hashes_a) == 2
    assert hashes_a[0] != hashes_a[1]


def test_branch_manager_checkpoint_restore_and_live_fork(tmp_path):
    pool = _SnapshotPool()
    source = _Sandbox("source")
    prefix = [{"role": "user", "content": "task"}]

    async def scenario():
        with CheckpointStore(tmp_path / "branch.sqlite3") as store:
            manager = BranchManager(pool, store)
            record = await manager.checkpoint(
                source,
                task_id="task-1",
                trajectory_prefix=prefix,
                step_id=12,
                trajectory_id="traj-1",
                env_fingerprint="env-v1",
                name="task-1-step-12",
            )
            restored = await manager.restore(
                record, count=2, agent_ids=["branch-a", "branch-b"]
            )
            forked = await manager.fork_live(
                source, count=2, agent_ids=["live-a", "live-b"]
            )
            return record, restored, forked

    record, restored, forked = asyncio.run(scenario())
    assert record.snapshot_id == "snap-1"
    assert pool.snapshots == [("source", "task-1-step-12", "snap-1")]
    assert [sb.agent_id for sb in restored] == ["branch-a", "branch-b"]
    assert [call[:2] for call in pool.restores] == [
        ("snap-1", "branch-a"),
        ("snap-1", "branch-b"),
    ]
    assert [sb.agent_id for sb in forked] == ["live-a", "live-b"]


def test_checkpoint_gc_preserves_latest_and_active_frontier(tmp_path):
    pool = _SnapshotPool()

    async def scenario():
        with CheckpointStore(tmp_path / "gc.sqlite3") as store:
            manager = BranchManager(pool, store)
            for step in range(1, 6):
                store.put(
                    task_id="task-gc",
                    trajectory_prefix=[{"step": step}],
                    snapshot_id=f"snap-{step}",
                    step_id=step,
                )

            dry = await manager.gc_checkpoints(
                task_id="task-gc",
                retain_latest=1,
                protected_snapshot_ids=["snap-2"],
                dry_run=True,
            )
            assert [record.snapshot_id for record in dry] == ["snap-1", "snap-3", "snap-4"]
            assert pool.deleted_snapshots == []

            deleted = await manager.gc_checkpoints(
                task_id="task-gc",
                retain_latest=1,
                protected_snapshot_ids=["snap-2"],
            )
            remaining = store.list_for_task("task-gc")
            return deleted, remaining

    deleted, remaining = asyncio.run(scenario())
    assert [record.snapshot_id for record in deleted] == ["snap-1", "snap-3", "snap-4"]
    assert pool.deleted_snapshots == ["snap-1", "snap-3", "snap-4"]
    assert [record.snapshot_id for record in remaining] == ["snap-2", "snap-5"]


def test_checkpoint_gc_keeps_metadata_when_physical_delete_fails(tmp_path):
    class _FailingDeletePool(_SnapshotPool):
        async def delete_snapshot(self, snapshot_id):
            raise RuntimeError("delete failed")

    pool = _FailingDeletePool()

    async def scenario():
        with CheckpointStore(tmp_path / "gc-failure.sqlite3") as store:
            store.put(
                task_id="task-gc",
                trajectory_prefix=[{"step": 1}],
                snapshot_id="snap-1",
                step_id=1,
            )
            try:
                await BranchManager(pool, store).gc_checkpoints(
                    task_id="task-gc",
                    retain_latest=0,
                )
            except RuntimeError:
                pass
            return store.list_for_task("task-gc")

    remaining = asyncio.run(scenario())
    assert [record.snapshot_id for record in remaining] == ["snap-1"]
