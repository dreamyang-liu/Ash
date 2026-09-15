from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

from rl_driver.ledger import Ledger
from rl_driver.profiling import (
    PROFILE_SCHEMA_VERSION,
    build_profile_records,
    export_jsonl,
    records_from_ledger,
)
from rl_driver.tests.test_message_rollout import request as message_request
from rl_driver.tests.test_miles import miles_request


def _v2_result(body: dict, *, group: str | None = None, slot: str | None = None) -> dict:
    group = group or body["rollout_job_id"]
    slot = slot or body["sample_slots"][0]["sample_slot_id"]
    prompt = body["prompt_token_ids"]
    return {
        "protocol_version": "ash-rollout-v2",
        "rollout_job_id": group,
        "prompt_group_id": body["prompt_group_id"],
        "status": "completed",
        "max_samples": 1,
        "actual_samples": 1,
        "stop_reason": None,
        "search_branches": 0,
        "consumed_budget": {"model_calls": 2, "tool_calls": 1},
        "trajectories": [{
            "sample_slot_id": slot,
            "branch_id": group + ":root",
            "parent_branch_id": None,
            "branch_point_token_count": None,
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call", "function": {"name": "shell"},
                }]},
                {"role": "tool", "tool_call_id": "call", "content": "ok"},
                {"role": "assistant", "content": "done"},
            ],
            "token_ids": [*prompt, 90, 91, 92],
            "prompt_length": len(prompt),
            "generated_spans": [
                {"response_id": "r1", "start": len(prompt), "end": len(prompt) + 1,
                 "input_token_ids": prompt, "output_token_ids": [90],
                 "weight_version": "42", "finish_reason": "tool_calls"},
                {"response_id": "r2", "start": len(prompt) + 2, "end": len(prompt) + 3,
                 "input_token_ids": [*prompt, 90, 91], "output_token_ids": [92],
                 "weight_version": "42", "finish_reason": "stop"},
            ],
            "response_text": "done",
            "reward": 1.0,
            "status": "completed",
            "prompt_token_alignment": "request_exact",
            "metadata": {"job_id": group + "-actor", "attempt_id": "attempt"},
        }],
    }


def _document(result: dict, *, completed_at: float = 14.5) -> dict:
    trajectory = result["trajectories"][0]
    return {
        "completed_at": completed_at,
        "samples": [{
            "sample_slot_id": "internal-slot",
            "actor": {"job_id": trajectory["metadata"]["job_id"],
                      "result": {"elapsed_s": 8.25, "rollout_usage": {
                          "model_calls": 2, "tool_calls": 1,
                      }}},
            "grade": {"result": {"resolved": True, "elapsed_s": 1.5}},
        }],
    }


def test_v2_profile_uses_exact_session_tokens_without_retokenizing():
    request = miles_request()
    request.update(max_samples=1, minimum_returned_samples=1)
    request["sample_slots"] = request["sample_slots"][:1]
    result = _v2_result(request)

    (record,) = build_profile_records(
        request, result, _document(result), created_at=4.0
    )

    assert record["schema_version"] == PROFILE_SCHEMA_VERSION
    assert record["prompt_tokens"] == len(request["prompt_token_ids"])
    assert record["assistant_generated_tokens"] == 2
    assert record["non_generated_response_tokens"] == 1
    assert record["final_transcript_tokens"] == len(request["prompt_token_ids"]) + 3
    assert record["model_call_count"] == 2
    assert record["tool_call_count"] == 1
    assert record["actor_time_seconds"] == 8.25
    assert record["verifier_time_seconds"] == 1.5
    assert record["wall_time_seconds"] == 10.5
    assert len(record["token_ids_sha256"]) == 64


def test_v3_profile_never_invents_missing_token_statistics():
    request = message_request()
    trajectory = {
        "sample_slot_id": request["sample_slots"][0]["sample_slot_id"],
        "branch_id": "actor-job",
        "parent_branch_id": None,
        "messages": [{"role": "user", "content": "fix"},
                     {"role": "assistant", "content": "done"}],
        "tools": [],
        "reward": 0.0,
        "status": "completed",
        "hints_removed": True,
        "metadata": {"job_id": "actor-job", "attempt_id": "attempt"},
    }
    result = {
        "protocol_version": "ash-rollout-v3",
        "rollout_job_id": request["rollout_job_id"],
        "prompt_group_id": request["prompt_group_id"],
        "status": "completed",
        "max_samples": 1,
        "actual_samples": 1,
        "stop_reason": None,
        "search_branches": 0,
        "consumed_budget": {"model_calls": 2, "tool_calls": 1},
        "trajectories": [trajectory],
    }

    (record,) = build_profile_records(
        request, result, _document(result), created_at=4.0
    )

    assert record["model_call_count"] == 2
    assert record["tool_call_count"] == 1
    assert record["message_count"] == 2
    assert record["requested_budget"] == {
        "max_turns": request["max_turns"],
        "max_wall_time_seconds": request["budgets"]["max_wall_time_seconds"],
    }
    for field in (
        "prompt_tokens", "assistant_generated_tokens", "final_transcript_tokens",
        "train_sequence_tokens", "peak_context_tokens",
        "all_model_calls_input_tokens", "token_ids_sha256", "generated_spans",
    ):
        assert record[field] is None


def _save_profile(ledger: Ledger, group: str, *, created_at_offset: int = 0) -> dict:
    request = miles_request()
    request.update(rollout_job_id=group, max_samples=1, minimum_returned_samples=1)
    request["sample_slots"] = [{"sample_slot_id": group + "-slot", "sample_index": 0}]
    result = _v2_result(request)
    document = _document(result)
    row = ledger.create(group, request, document)
    document["profiling_records"] = build_profile_records(
        request, result, document, created_at=row["created_at"] - created_at_offset
    )
    ledger.save(group, document, terminal=True)
    return document["profiling_records"][0]


def test_jsonl_export_is_stable_idempotent_and_survives_restart(tmp_path):
    path = tmp_path / "groups.sqlite3"
    ledger = Ledger(path)
    expected_b = _save_profile(ledger, "group-b")
    expected_a = _save_profile(ledger, "group-a")
    output = tmp_path / "profile.jsonl"

    assert export_jsonl(ledger, output) == 2
    first = output.read_bytes()
    assert export_jsonl(Ledger(path), output) == 2
    assert output.read_bytes() == first
    assert [json.loads(line) for line in first.splitlines()] == [expected_a, expected_b]
    assert records_from_ledger(Ledger(path)) == [expected_a, expected_b]


def test_concurrent_exporters_never_append_duplicates_or_partial_lines(tmp_path):
    ledger = Ledger(tmp_path / "groups.sqlite3")
    expected = [_save_profile(ledger, f"group-{index}") for index in range(12)]
    output = tmp_path / "profile.jsonl"

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(lambda _: export_jsonl(Ledger(ledger.path), output), range(24)))

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert counts == [len(expected)] * 24
    assert rows == sorted(expected, key=lambda row: (
        row["rollout_job_id"], row["sample_slot_id"], row["branch_id"]
    ))
    assert len({(row["rollout_job_id"], row["sample_slot_id"], row["branch_id"])
                for row in rows}) == len(expected)


def test_identical_profile_identity_is_deduplicated_but_conflict_fails(tmp_path):
    ledger = Ledger(tmp_path / "groups.sqlite3")
    record = _save_profile(ledger, "group")
    duplicate = deepcopy(record)
    ledger.create("copy", {"copy": True}, {"profiling_records": [duplicate]})
    assert records_from_ledger(ledger) == [record]

    conflict = deepcopy(record)
    conflict["reward"] = 0.0
    ledger.create("conflict", {"conflict": True}, {"profiling_records": [conflict]})
    try:
        records_from_ledger(ledger)
    except ValueError as error:
        assert "Conflicting profiling record identity" in str(error)
    else:
        raise AssertionError("conflicting profile identity was silently accepted")
