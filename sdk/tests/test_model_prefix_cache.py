from ash_sandbox.model_prefix_cache import (
    HistoryOnlyPrefixBackend,
    ModelPrefixHandle,
    ModelPrefixStore,
    PrefixCacheCapabilities,
    prefix_reuse_accounting,
)


def test_history_only_backend_is_not_counted_as_prefill_compute_reuse():
    backend = HistoryOnlyPrefixBackend(backend="codex-session-jsonl")
    handle = backend.capture(
        task_id="task-1",
        trajectory_prefix=[{"role": "user", "content": "hello"}],
        model="gpt-test",
        reference="frozen-parent.jsonl",
        prefix_units=1234,
    )

    assert handle.capabilities.history_reuse is True
    assert handle.capabilities.kv_reuse is False
    assert handle.estimated_prefill_units_avoided == 0
    accounting = prefix_reuse_accounting(handle)
    assert accounting["history_reuse"] is True
    assert accounting["kv_reuse"] is False
    assert accounting["prefix_units"] == 1234
    assert accounting["estimated_prefill_units_avoided"] == 0


def test_true_kv_handle_counts_reusable_prefix_units():
    handle = ModelPrefixHandle(
        task_id="task-1",
        prefix_hash="abc",
        model="gpt-test",
        backend="test-kv",
        reference="opaque-ref",
        capabilities=PrefixCacheCapabilities(
            history_reuse=True,
            kv_reuse=True,
            persistent=False,
            cross_process=False,
            backend_scope="process",
        ),
        prefix_units=777,
    )
    assert handle.estimated_prefill_units_avoided == 777


def test_model_prefix_store_round_trip_and_model_backend_isolation(tmp_path):
    prefix = [{"role": "user", "content": "same prefix"}]
    backend = HistoryOnlyPrefixBackend(backend="codex-session-jsonl")
    first = backend.capture(
        task_id="task-1",
        trajectory_prefix=prefix,
        model="model-a",
        reference="parent-a.jsonl",
        prefix_units=100,
        metadata={"session": "a"},
    )
    second = backend.capture(
        task_id="task-1",
        trajectory_prefix=prefix,
        model="model-b",
        reference="parent-b.jsonl",
        prefix_units=110,
    )

    with ModelPrefixStore(tmp_path / "prefixes.sqlite3") as store:
        stored_first = store.put(first)
        store.put(second)
        assert stored_first.created_at

        loaded = store.get(
            task_id="task-1",
            trajectory_prefix=prefix,
            model="model-a",
            backend="codex-session-jsonl",
        )
        assert loaded is not None
        assert loaded.reference == "parent-a.jsonl"
        assert loaded.prefix_units == 100
        assert loaded.metadata == {"session": "a"}
        assert loaded.capabilities.history_reuse is True
        assert loaded.capabilities.kv_reuse is False

        records = store.list_for_task("task-1")
        assert len(records) == 2
        assert {record.model for record in records} == {"model-a", "model-b"}
        assert store.delete(loaded) is True
        assert store.get(
            task_id="task-1",
            trajectory_prefix=prefix,
            model="model-a",
            backend="codex-session-jsonl",
        ) is None


def test_history_backend_rejects_invalid_prefix_units():
    backend = HistoryOnlyPrefixBackend()
    try:
        backend.capture(
            task_id="task-1",
            trajectory_prefix=[],
            model="model-a",
            reference="parent.jsonl",
            prefix_units=-1,
        )
    except ValueError as exc:
        assert "prefix_units" in str(exc)
    else:
        raise AssertionError("expected ValueError")
