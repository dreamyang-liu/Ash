from ash_sandbox.checkpoints import CheckpointStore
from ash_sandbox.exact_prefix_cache import ExactPrefixCache
from ash_sandbox.model_prefix_cache import HistoryOnlyPrefixBackend, ModelPrefixStore


def test_exact_prefix_cache_returns_deepest_environment_and_model_match(tmp_path):
    steps = [
        {"tool": "shell", "args": {"command": "pwd"}},
        {"tool": "shell", "args": {"command": "ls"}},
        {"tool": "text_editor", "args": {"path": "a.py"}},
        {"tool": "shell", "args": {"command": "pytest -q"}},
    ]
    with CheckpointStore(tmp_path / "checkpoints.sqlite3") as checkpoints, ModelPrefixStore(
        tmp_path / "model-prefixes.sqlite3"
    ) as model_prefixes:
        shallow = checkpoints.put(
            task_id="task-1",
            trajectory_prefix=steps[:2],
            snapshot_id="snap-2",
            step_id=2,
            env_fingerprint="env-A",
        )
        deep = checkpoints.put(
            task_id="task-1",
            trajectory_prefix=steps[:3],
            snapshot_id="snap-3",
            step_id=3,
            env_fingerprint="env-A",
        )
        backend = HistoryOnlyPrefixBackend(backend="qwen-history-jsonl")
        handle = backend.capture(
            task_id="task-1",
            trajectory_prefix=steps[:3],
            model="qwen3.8-27b",
            reference="trajectory-3.jsonl",
            prefix_units=300,
        )
        cache = ExactPrefixCache(checkpoints, model_prefixes)
        cache.bind_model_prefix(checkpoint=deep, model_prefix=handle)

        hit = cache.lookup(
            task_id="task-1",
            trajectory_prefix=steps,
            env_fingerprint="env-A",
            model="qwen3.8-27b",
            model_backend="qwen-history-jsonl",
        )

        assert shallow.snapshot_id == "snap-2"
        assert hit is not None
        assert hit.snapshot_id == "snap-3"
        assert hit.matched_length == 3
        assert hit.model_prefix is not None
        assert hit.model_prefix.reference == "trajectory-3.jsonl"
        assert hit.model_prefix.capabilities.history_reuse is True
        assert hit.model_prefix.capabilities.kv_reuse is False
        assert hit.model_prefix.estimated_prefill_units_avoided == 0


def test_exact_prefix_cache_keeps_environment_hit_when_model_prefix_is_missing(tmp_path):
    prefix = [{"tool": "shell", "args": {"command": "pwd"}}]
    query = prefix + [{"tool": "shell", "args": {"command": "ls"}}]
    with CheckpointStore(tmp_path / "checkpoints.sqlite3") as checkpoints, ModelPrefixStore(
        tmp_path / "model-prefixes.sqlite3"
    ) as model_prefixes:
        checkpoints.put(
            task_id="task-1",
            trajectory_prefix=prefix,
            snapshot_id="snap-1",
            step_id=1,
            env_fingerprint="env-A",
        )
        hit = ExactPrefixCache(checkpoints, model_prefixes).lookup(
            task_id="task-1",
            trajectory_prefix=query,
            env_fingerprint="env-A",
            model="qwen3.8-27b",
            model_backend="qwen-history-jsonl",
        )
        assert hit is not None
        assert hit.snapshot_id == "snap-1"
        assert hit.has_model_prefix is False


def test_exact_prefix_cache_respects_environment_identity(tmp_path):
    prefix = [{"tool": "shell", "args": {"command": "pwd"}}]
    with CheckpointStore(tmp_path / "checkpoints.sqlite3") as checkpoints, ModelPrefixStore(
        tmp_path / "model-prefixes.sqlite3"
    ) as model_prefixes:
        checkpoints.put(
            task_id="task-1",
            trajectory_prefix=prefix,
            snapshot_id="snap-A",
            step_id=1,
            env_fingerprint="env-A",
        )
        cache = ExactPrefixCache(checkpoints, model_prefixes)
        assert cache.lookup(
            task_id="task-1", trajectory_prefix=prefix, env_fingerprint="env-B"
        ) is None


def test_bind_model_prefix_rejects_mismatched_prefix(tmp_path):
    with CheckpointStore(tmp_path / "checkpoints.sqlite3") as checkpoints, ModelPrefixStore(
        tmp_path / "model-prefixes.sqlite3"
    ) as model_prefixes:
        checkpoint = checkpoints.put(
            task_id="task-1",
            trajectory_prefix=["A"],
            snapshot_id="snap-A",
            step_id=1,
        )
        handle = HistoryOnlyPrefixBackend(backend="qwen-history-jsonl").capture(
            task_id="task-1",
            trajectory_prefix=["B"],
            model="qwen3.8-27b",
            reference="other.jsonl",
        )
        cache = ExactPrefixCache(checkpoints, model_prefixes)
        try:
            cache.bind_model_prefix(checkpoint=checkpoint, model_prefix=handle)
        except ValueError as exc:
            assert "different prefixes" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_lookup_requires_model_and_backend_as_pair(tmp_path):
    with CheckpointStore(tmp_path / "checkpoints.sqlite3") as checkpoints, ModelPrefixStore(
        tmp_path / "model-prefixes.sqlite3"
    ) as model_prefixes:
        cache = ExactPrefixCache(checkpoints, model_prefixes)
        try:
            cache.lookup(task_id="task-1", trajectory_prefix=[], model="qwen3.8-27b")
        except ValueError as exc:
            assert "provided together" in str(exc)
        else:
            raise AssertionError("expected ValueError")
