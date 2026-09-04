from __future__ import annotations

from copy import deepcopy

from ash_sandbox import ExactPrefixIndex, RelaxedChangeIndex, TrajectoryCache
from swebench.cache_resume import resolve_cache_resume, restore_cache_resume


def tr(name: str, args: dict, content: str = "ok") -> dict:
    return {
        "role": "tool_result",
        "tool_name": name,
        "tool_args": args,
        "content": content,
        "success": True,
    }


class _Session:
    def __init__(self):
        self.calls = []

    def restore(self, snapshot_id: str, agent_id: str = "") -> bool:
        self.calls.append((snapshot_id, agent_id))
        return True


def test_relaxed_resume_restores_source_environment_but_keeps_target_history(tmp_path):
    exact = ExactPrefixIndex(tmp_path / "exact.sqlite3")
    relaxed = RelaxedChangeIndex(tmp_path / "relaxed.sqlite3")
    cache = TrajectoryCache(exact, relaxed)
    source = [
        tr("grep_files", {"pattern": "foo", "path": "/testbed"}, "a.py:1"),
        tr("text_editor", {"command": "write", "path": "/testbed/x.txt", "file_text": "x"}, "done"),
    ]
    target = [
        tr("grep_files", {"pattern": "bar", "path": "/testbed"}, "b.py:2"),
        tr("text_editor", {"command": "write", "path": "/testbed/x.txt", "file_text": "x"}, "done"),
    ]
    target_original = deepcopy(target)
    try:
        cache.register(
            task_id="task",
            env_fingerprint="env",
            trajectory_prefix=source,
            messages=source,
            reference="snapshot:source",
            step_id=7,
        )
        decision = resolve_cache_resume(
            cache,
            mode="relaxed",
            task_id="task",
            env_fingerprint="env",
            target_messages=target,
        )
        assert decision.hit is True
        assert decision.match_kind == "relaxed"
        assert decision.relaxed_tier == "projection"
        assert decision.snapshot_id == "snapshot:source"
        assert decision.initial_messages == target_original
        assert decision.initial_messages != source

        # Both the caller's input and the decision's stored copy are detached.
        target[0]["content"] = "mutated outside"
        resumed = decision.initial_messages
        resumed[0]["content"] = "mutated copy"
        assert decision.initial_messages == target_original

        session = _Session()
        assert restore_cache_resume(session, decision, agent_id="target-branch") is True
        assert session.calls == [("snapshot:source", "target-branch")]
    finally:
        exact.close()
        relaxed.close()


def test_exact_miss_does_not_restore_any_environment(tmp_path):
    exact = ExactPrefixIndex(tmp_path / "exact.sqlite3")
    relaxed = RelaxedChangeIndex(tmp_path / "relaxed.sqlite3")
    cache = TrajectoryCache(exact, relaxed)
    try:
        decision = resolve_cache_resume(
            cache,
            mode="exact",
            task_id="task",
            env_fingerprint="env",
            target_messages=[{"role": "user", "content": "target"}],
        )
        assert decision.hit is False
        session = _Session()
        assert restore_cache_resume(session, decision, agent_id="branch") is False
        assert session.calls == []
    finally:
        exact.close()
        relaxed.close()
