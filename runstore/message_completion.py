"""Finalize v3 episodes at a verified filesystem and native-message boundary."""

from runstore.failures import failure_kind
from runstore.files import journal_events
from runstore.message_export import export_messages, export_tools

LIMIT_STOP_REASONS = frozenset({"max_turns_reached", "timeout", "max_sequence_tokens"})
TERMINATION_GRACE_SECONDS = 120


def capture_final_snapshot(provisioned, directory, ledger, run_id: str) -> dict:
    """Capture after tool execution drains, before the owning VM is destroyed."""
    stop_server = getattr(provisioned, "stop_server", None)
    if stop_server is None:
        raise ValueError("Final snapshot requires an owned execution server")
    stop_server()
    events = journal_events(directory / "trajectory.jsonl")
    if any(event.get("reason") == "execution_uncertain" for event in events):
        raise ValueError("Uncertain tool execution has no gradable final state")
    session = getattr(provisioned, "session", None)
    if session is None:
        raise ValueError("Final snapshot requires an owned sandbox session")
    snapshot = session.snapshot(name=f"ash-final-{run_id}", disk_only=True)
    identifier = getattr(snapshot, "id", None)
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Could not capture the final grading snapshot")
    ledger._append("claim", run_id=run_id, kind="snapshot", id=identifier)
    return {"final_snapshot_id": identifier}


def is_truncated_result(result: dict) -> bool:
    return (result.get("status") == "truncated"
            and result.get("stop_reason") in LIMIT_STOP_REASONS
            and isinstance(result.get("final_snapshot_id"), str)
            and bool(result["final_snapshot_id"])
            and not result.get("training_snapshot_error")
            and failure_kind(result) not in {"configuration", "infrastructure"})


def complete_message_result(result: dict, directory, slot: str, recovery=None, *, contract=None) -> dict:
    result = dict(result)
    events = journal_events(directory / "trajectory.jsonl")
    stopped_by_limit = result.get("stop_reason") in LIMIT_STOP_REASONS
    unsafe = any(event.get("reason") == "execution_uncertain" for event in events)
    if (stopped_by_limit and not unsafe and result.get("final_snapshot_id")
            and not result.get("training_snapshot_error")
            and failure_kind(result) not in {"configuration", "infrastructure"}):
        result.update(agent_status=result["status"], status="truncated", failure_kind=None)
    references = [e["native_session_id"] for e in events
                  if e.get("type") == "session.ref" and e.get("native_session_id")]
    session_id = result.get("native_session_id") or (references[-1] if references else None)
    result["native_session_id"] = session_id
    try:
        result["training_messages"] = export_messages(
            directory, session_id, slot, events, allow_incomplete_tail=is_truncated_result(result))
        result["training_tools"] = export_tools(events)
        if contract and contract.get("max_sequence_tokens") is not None:
            from runstore.sequence_limits import apply_sequence_limit

            if (unsafe or result.get("training_snapshot_error") or not result.get("final_snapshot_id")
                    or result.get("status") not in {"completed", "truncated"}
                    or failure_kind(result) in {"configuration", "infrastructure"}):
                raise ValueError("Unsafe or failed execution cannot be converted into a length-truncated reward")
            result = apply_sequence_limit(result, directory, slot, events, recovery, contract)
        if recovery:
            result["training_origin"] = {
                "job_id": recovery.get("job_id"), "point_id": recovery["id"],
                "tool_depth": recovery["tool_depth"],
            }
        usage = [e for e in events if e.get("type") == "rollout.usage"]
        result["rollout_usage"] = {
            key: usage[-1][key] for key in ("model_calls", "tool_calls")
        } if usage else {}
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["training_export_error"] = str(error)
    return result
