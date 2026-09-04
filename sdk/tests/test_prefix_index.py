from ash_sandbox.prefix_index import ExactPrefixIndex


def test_shared_prefix_nodes_are_stored_once(tmp_path):
    common = [{"step": i} for i in range(100)]
    with ExactPrefixIndex(tmp_path / "prefix.sqlite3") as index:
        a = index.put(
            task_id="task",
            trajectory_prefix=common + [{"branch": "a"}],
            reference="snap-a",
            env_fingerprint="env",
        )
        b = index.put(
            task_id="task",
            trajectory_prefix=common + [{"branch": "b"}],
            reference="snap-b",
            env_fingerprint="env",
        )
        stats = index.stats()
        assert a.cursor.depth == 101
        assert b.cursor.depth == 101
        assert stats["node_count"] == 102
        assert stats["target_count"] == 2


def test_incremental_append_supports_dense_checkpointing(tmp_path):
    with ExactPrefixIndex(tmp_path / "dense.sqlite3") as index:
        cursor = index.root()
        for i in range(250):
            cursor = index.append(cursor, {"step": i})
            index.register(
                task_id="dense",
                cursor=cursor,
                reference=f"snap-{i}",
                env_fingerprint="env-v1",
                step_id=i + 1,
            )
        stats = index.stats()
        assert stats["node_count"] == 250
        assert stats["target_count"] == 250
        assert len(index.reconstruct(cursor)) == 250


def test_longest_match_finds_deepest_shared_prefix(tmp_path):
    shared = [{"step": 1}, {"step": 2}, {"step": 3}]
    with ExactPrefixIndex(tmp_path / "match.sqlite3") as index:
        index.put(
            task_id="task",
            trajectory_prefix=shared,
            reference="snap-3",
            env_fingerprint="env",
            step_id=3,
        )
        index.put(
            task_id="task",
            trajectory_prefix=shared + [{"branch": "old"}],
            reference="snap-old",
            env_fingerprint="env",
            step_id=4,
        )
        hit = index.longest_match(
            task_id="task",
            trajectory_prefix=shared + [{"branch": "new"}, {"step": 5}],
            env_fingerprint="env",
        )
        assert hit is not None
        assert hit.reference == "snap-3"
        assert hit.cursor.depth == 3


def test_longest_match_is_scoped_by_environment(tmp_path):
    with ExactPrefixIndex(tmp_path / "env.sqlite3") as index:
        index.put(
            task_id="task",
            trajectory_prefix=[{"step": 1}],
            reference="snap-a",
            env_fingerprint="env-a",
        )
        assert index.longest_match(
            task_id="task",
            trajectory_prefix=[{"step": 1}, {"step": 2}],
            env_fingerprint="env-b",
        ) is None
