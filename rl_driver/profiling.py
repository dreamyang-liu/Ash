"""Durable profiling records and deterministic JSONL export.

Profiling is derived from the same terminal result that the driver persists.
It is deliberately not an append-side effect of execution completion: a
process may crash at any point and the JSONL file can be regenerated from the
ledger without losing or duplicating trajectories.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from rl_driver.ledger import Ledger, canonical


PROFILE_SCHEMA_VERSION = "ash-rollout-profile-v2"
_TERMINAL = {"completed", "early_stopped", "failed", "cancelled"}
_TOKEN_FIELDS = (
    "prompt_tokens",
    "assistant_generated_tokens",
    "non_generated_response_tokens",
    "final_transcript_tokens",
    "train_sequence_tokens",
    "peak_context_tokens",
    "all_model_calls_input_tokens",
    "token_ids_sha256",
    "generated_spans",
)


def _number(value: Any, *, digits: int = 3) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number >= 0:
            return round(number, digits)
    return None


def _stages_by_trajectory(document: dict, trajectory: dict, index: int) -> tuple[dict, dict | None]:
    job_id = (trajectory.get("metadata") or {}).get("job_id") or trajectory.get("branch_id")
    for sample in document.get("samples", []):
        actor = sample.get("actor") or {}
        if job_id and actor.get("job_id") == job_id:
            return actor, sample.get("grade")
    samples = document.get("samples") or []
    if index < len(samples):
        return samples[index].get("actor") or {}, samples[index].get("grade")
    return {}, None


def _tool_calls(messages: list[dict]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def _common_record(
    request: dict,
    result: dict,
    trajectory: dict,
    document: dict,
    index: int,
    *,
    group_wall_time_seconds: float | None,
) -> dict:
    actor, grade = _stages_by_trajectory(document, trajectory, index)
    actor_result = actor.get("result") or {}
    grade_result = (grade or {}).get("result") or {}
    metadata = trajectory.get("metadata") or {}
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "protocol_version": request["protocol_version"],
        "rollout_job_id": request["rollout_job_id"],
        "rollout_id": request["rollout_id"],
        "prompt_group_id": request["prompt_group_id"],
        "task_id": request["task_id"],
        "sample_slot_id": trajectory["sample_slot_id"],
        "branch_id": trajectory["branch_id"],
        "parent_branch_id": trajectory.get("parent_branch_id"),
        "branch_point_token_count": trajectory.get("branch_point_token_count"),
        "environment_ref": request["environment_ref"],
        "model": request.get("model"),
        "sampling_params": request.get("sampling_params", {}),
        "seed": request.get("sampling_params", {}).get("seed"),
        "expected_weight_version": request.get("expected_weight_version"),
        "message_count": len(trajectory.get("messages") or []),
        "tool_record_count": len(trajectory.get("tools") or []),
        "wall_time_seconds": _number(group_wall_time_seconds),
        "actor_time_seconds": _number(actor_result.get("elapsed_s")),
        "model_time_seconds": _number(metadata.get("model_time_seconds")),
        "tool_time_seconds": _number(metadata.get("tool_time_seconds")),
        "sandbox_setup_seconds": _number(metadata.get("sandbox_setup_seconds")),
        "verifier_time_seconds": (
            _number(grade_result.get("verifier_seconds"))
            if _number(grade_result.get("verifier_seconds")) is not None
            else _number(grade_result.get("elapsed_s"))
            if _number(grade_result.get("elapsed_s")) is not None
            else _number(metadata.get("verifier_seconds"))
        ),
        "patch_sha256": grade_result.get("patch_sha256"),
        "patch_chars": grade_result.get("patch_chars"),
        "patch_files_changed": grade_result.get("patch_files_changed"),
        "patch_lines_changed": grade_result.get("patch_lines_changed"),
        "patch_added_paths": grade_result.get("patch_added_paths"),
        "trajectory_status": trajectory.get("status"),
        "trajectory_stop_reason": trajectory.get("stop_reason"),
        "job_status": result.get("status"),
        "job_stop_reason": result.get("stop_reason"),
        "reward": trajectory.get("reward"),
        "search_branches": result.get("search_branches", 0),
        "consumed_budget": result.get("consumed_budget", {}),
        "trajectory_metadata": metadata,
    }


def _v2_record(request: dict, result: dict, trajectory: dict, document: dict,
               index: int, wall_time: float | None) -> dict:
    record = _common_record(
        request, result, trajectory, document, index,
        group_wall_time_seconds=wall_time,
    )
    tokens = trajectory["token_ids"]
    prompt_length = trajectory["prompt_length"]
    spans = trajectory["generated_spans"]
    generated = sum(len(span["output_token_ids"]) for span in spans)
    token_bytes = ",".join(str(token) for token in tokens).encode()
    messages = trajectory.get("messages") or []
    record.update({
        "requested_budget": request["budgets"],
        "prompt_token_alignment": trajectory.get("prompt_token_alignment", "request_exact"),
        "prompt_tokens": prompt_length,
        "assistant_generated_tokens": generated,
        "non_generated_response_tokens": len(tokens) - prompt_length - generated,
        "final_transcript_tokens": len(tokens),
        "train_sequence_tokens": len(tokens),
        "peak_context_tokens": max(len(span["input_token_ids"]) for span in spans),
        "all_model_calls_input_tokens": sum(len(span["input_token_ids"]) for span in spans),
        "model_call_count": len(spans),
        "tool_call_count": _tool_calls(messages),
        "generated_spans": [{
            "response_id": span["response_id"],
            "start": span["start"],
            "end": span["end"],
            "input_tokens": len(span["input_token_ids"]),
            "output_tokens": len(span["output_token_ids"]),
            "weight_version": span["weight_version"],
            "finish_reason": span["finish_reason"],
        } for span in spans],
        "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "finish_reasons": [span["finish_reason"] for span in spans],
    })
    return record


def _v3_record(request: dict, result: dict, trajectory: dict, document: dict,
               index: int, wall_time: float | None) -> dict:
    record = _common_record(
        request, result, trajectory, document, index,
        group_wall_time_seconds=wall_time,
    )
    actor, _ = _stages_by_trajectory(document, trajectory, index)
    usage = (actor.get("result") or {}).get("rollout_usage") or {}
    for field in _TOKEN_FIELDS:
        record[field] = None
    record.update({
        "requested_budget": {
            "max_turns": request.get("max_turns"),
            **request["budgets"],
        },
        "prompt_token_alignment": None,
        "model_call_count": (
            usage.get("model_calls")
            if type(usage.get("model_calls")) is int and usage["model_calls"] >= 0
            else None
        ),
        "tool_call_count": (
            usage.get("tool_calls")
            if type(usage.get("tool_calls")) is int and usage["tool_calls"] >= 0
            else len(trajectory["tools"])
            if isinstance(trajectory.get("tools"), list)
            else None
        ),
        "finish_reasons": None,
    })
    return record


def build_profile_records(
    request: dict,
    result: dict,
    document: dict,
    *,
    created_at: float | None,
) -> list[dict]:
    """Build one self-contained record per returned terminal trajectory."""
    if result.get("status") not in _TERMINAL:
        raise ValueError("Profiling requires a terminal rollout result")
    completed_at = document.get("completed_at")
    wall_time = (
        max(0.0, float(completed_at) - float(created_at))
        if isinstance(completed_at, (int, float))
        and isinstance(created_at, (int, float))
        else None
    )
    version = request.get("protocol_version")
    builder = _v2_record if version == "ash-rollout-v2" else _v3_record if version == "ash-rollout-v3" else None
    if builder is None:
        raise ValueError(f"Unsupported profiling protocol: {version!r}")
    return [
        builder(request, result, trajectory, document, index, wall_time)
        for index, trajectory in enumerate(result.get("trajectories") or [])
    ]


def _record_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("rollout_job_id", "")),
        str(record.get("sample_slot_id", "")),
        str(record.get("branch_id", "")),
    )


def records_from_ledger(ledger: Ledger) -> list[dict]:
    """Read a stable, deduplicated snapshot of all materialized profiles."""
    unique: dict[tuple[str, str, str], dict] = {}
    for row in ledger.rows():
        records = row["document"].get("profiling_records")
        if records is None:
            continue
        if not isinstance(records, list):
            raise ValueError(f"Malformed profiling_records in group {row['id']}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Malformed profiling record in group {row['id']}")
            key = _record_key(record)
            if not all(key):
                raise ValueError(f"Profiling record has an incomplete identity in group {row['id']}")
            previous = unique.setdefault(key, record)
            if canonical(previous) != canonical(record):
                raise ValueError(f"Conflicting profiling record identity: {key!r}")
    return [unique[key] for key in sorted(unique)]


@contextmanager
def _output_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def export_jsonl(ledger: Ledger, output: str | os.PathLike[str]) -> int:
    """Atomically replace ``output`` with the ledger's current profile view."""
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(path):
        records = records_from_ledger(ledger)
        descriptor, temporary = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_jsonl(Ledger(args.ledger), args.output)
    print(f"exported {count} profiling records to {args.output}")


if __name__ == "__main__":
    main()
