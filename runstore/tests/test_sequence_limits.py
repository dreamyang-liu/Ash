import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from harness.rollout import RolloutControls
from runstore.native import NativePoint
from runstore import sequence_limits as limits


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        for message in messages:
            for call in message.get("tool_calls") or []:
                assert isinstance(call["function"]["arguments"], dict)
        return [0] * sum(len(message.get("content", "")) for message in messages)


@pytest.fixture
def tokenizer(monkeypatch):
    monkeypatch.setattr(limits, "_tokenizer", lambda path: CharacterTokenizer())


def point(tmp_path, messages, depth):
    path = tmp_path / f"prefix-{depth}.jsonl"
    data = "".join(json.dumps({"type": m["role"], "message": m}) + "\n" for m in messages).encode()
    path.write_bytes(data)
    return NativePoint(depth, depth, f"snapshot-{depth}", {
        "path": str(path), "byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "slot": "claude-code", "session_id": "session",
    })


def result(messages):
    return {
        "status": "completed", "final_snapshot_id": "later-full-state", "native_session_id": "session",
        "training_messages": messages, "training_tools": [],
    }


def test_overlength_export_keeps_exact_prefix_and_its_snapshot(tmp_path, monkeypatch, tokenizer):
    prefix = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "x" * 1100}]
    later = prefix + [{"role": "assistant", "content": "y" * 400}]
    points = [point(tmp_path, prefix, 1), point(tmp_path, later, 2)]
    monkeypatch.setattr(limits, "_prefixes", lambda *args: points)
    output = limits.apply_sequence_limit(
        result(later + [{"role": "assistant", "content": "z" * 500}]),
        tmp_path, "claude-code", [], None, {"max_sequence_tokens": 1300, "sequence_tokenizer_path": "/fixture"},
    )
    assert output["training_messages"] == prefix
    assert output["final_snapshot_id"] == "snapshot-1"
    assert output["raw_final_snapshot_id"] == "later-full-state"
    assert output["status"] == "truncated" and output["stop_reason"] == "max_sequence_tokens"
    assert output["training_token_count"] == 1104
    assert output["training_point_tokens"] == {"1": 1104, "2": 1504}
    assert output["sequence_truncation"]["tool_depth"] == 1


def test_no_gradable_prefix_is_not_replaced_by_arbitrary_token_slice(tmp_path, monkeypatch, tokenizer):
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "x" * 1500}]
    monkeypatch.setattr(limits, "_prefixes", lambda *args: [point(tmp_path, messages, 1)])
    original = result(messages)
    with pytest.raises(ValueError, match="No complete native checkpoint"):
        limits.apply_sequence_limit(
            original, tmp_path, "claude-code", [], None,
            {"max_sequence_tokens": 1300, "sequence_tokenizer_path": "/fixture"},
        )
    assert original["final_snapshot_id"] == "later-full-state"
    assert original["status"] == "completed"


def test_short_export_preserves_final_state_and_no_limit_needs_no_tokenizer(tmp_path, monkeypatch, tokenizer):
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "done"}]
    monkeypatch.setattr(limits, "_prefixes", lambda *args: [])
    original = result(messages)
    output = limits.apply_sequence_limit(
        original, tmp_path, "claude-code", [], None,
        {"max_sequence_tokens": 2048, "sequence_tokenizer_path": "/fixture"},
    )
    assert output["status"] == "completed" and output["final_snapshot_id"] == "later-full-state"
    assert output["training_token_count"] == 8
    monkeypatch.setattr(limits, "_tokenizer", lambda path: pytest.fail("No tokenizer for uncapped runs"))
    assert limits.apply_sequence_limit(original, tmp_path, "claude-code", [], None, None) is original


def test_tool_arguments_are_counted_with_same_mapping_as_miles(tokenizer):
    messages = [{"role": "assistant", "content": "inspect", "tool_calls": [{
        "id": "call", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
    }]}]
    assert limits.token_count(messages, [], "/fixture") == 7
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_live_request_caps_output_and_stops_before_another_model_call(monkeypatch):
    control = SimpleNamespace(reason=None, stop_reason=None)

    def stop(reason, *, stop_reason=None):
        control.reason, control.stop_reason = reason, stop_reason

    control.request_stop = stop
    journal = SimpleNamespace(emit=lambda *args, **kwargs: None)
    policy = RolloutControls({
        "message_export": True, "max_turns": 10, "deadline_at": time.time() + 100,
        "model_endpoint": "http://model", "sampling_params": {},
        "max_sequence_tokens": 8192, "sequence_tokenizer_path": "/fixture", "native_slot": "claude-code",
    }, journal, control)
    monkeypatch.setattr(limits, "remaining_output_budget", lambda *args: 4096)
    request = policy.prepare_model_request({"messages": [], "max_tokens": 32000}, "messages")
    assert request["max_tokens"] == 4096
    monkeypatch.setattr(limits, "remaining_output_budget", lambda *args: 0)
    with pytest.raises(ValueError, match="sequence token budget"):
        policy.prepare_model_request({"messages": [], "max_tokens": 32000}, "messages")
    assert control.stop_reason == "max_sequence_tokens"
