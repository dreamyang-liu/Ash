"""Private branch-policy scheduling shared by rollout wire adapters.

Policies choose how already allocated sample slots are consumed.  This module
only reconciles those choices with durable driver state; it is deliberately not
another public admission API.
"""

from copy import deepcopy
import importlib


def load_branch_policy(reference: str | None):
    """Load one deployment-owned pure policy callable from ``module:object``."""
    if reference is None:
        return None
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("branch_policy must be a module:callable reference")
    module_name, attribute = reference.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("branch_policy must be a module:callable reference")
    policy = getattr(importlib.import_module(module_name), attribute)
    if not callable(policy):
        raise ValueError("branch_policy reference is not callable")
    return policy


def reconcile_branch_policy(driver, policy, request_types: dict[str, type], internal_id) -> None:
    """Consume deferred slots for every supported rollout protocol.

    ``request_types`` maps the private request-document key to its strict wire
    parser.  Both v2 token trajectories and v3 message trajectories therefore
    use exactly the same quota, recovery-point and idempotency rules.
    """
    if policy is None:
        return
    for row in driver.ledger.rows():
        document = row["document"]
        if row["terminal"] or row["cancel_requested"]:
            continue
        request_key = next((key for key in request_types if key in document), None)
        if request_key is None:
            continue
        deferred = [
            sample for sample in document["samples"]
            if sample["actor"]["state"] == "deferred"
        ]
        if not deferred:
            continue
        request = request_types[request_key].from_dict(document[request_key])
        slots = request.sample_slots[:request.max_samples]
        external_by_internal = {
            internal_id(slot.sample_slot_id): slot.sample_slot_id for slot in slots
        }
        samples = []
        for sample in document["samples"]:
            actor = sample["actor"]
            points = []
            if actor.get("job_id") and actor.get("attempt_id"):
                points = driver.client.points(
                    actor["job_id"], attempt_id=actor["attempt_id"]
                )
            samples.append({
                "sample_slot_id": external_by_internal[sample["sample_slot_id"]],
                "state": actor["state"],
                "job_id": actor.get("job_id"),
                "attempt_id": actor.get("attempt_id"),
                "recovery_points": points,
            })
        decisions = policy({
            "request": request.to_dict(),
            "samples": samples,
            "deferred_sample_slot_ids": [
                external_by_internal[sample["sample_slot_id"]]
                for sample in deferred
            ],
        })
        if decisions is None:
            continue
        if not isinstance(decisions, list):
            raise ValueError("Branch policy must return a list of decisions")
        seen = set()
        allowed = {slot.sample_slot_id for slot in slots}
        for item in decisions:
            if not isinstance(item, dict) or set(item) != {"sample_slot_id", "decision"}:
                raise ValueError(
                    "Branch policy decisions require sample_slot_id and decision"
                )
            slot_id = item["sample_slot_id"]
            if slot_id not in allowed:
                raise KeyError(slot_id)
            if slot_id in seen:
                raise ValueError("Branch policy returned the same sample slot twice")
            seen.add(slot_id)
            decision = deepcopy(item["decision"])
            if decision.get("kind") == "branch":
                source = decision.get("source_sample_slot_id")
                if source not in allowed:
                    raise KeyError(source)
                decision["source_sample_slot_id"] = internal_id(source)
            driver.decide_deferred(
                row["id"], internal_id(slot_id), decision
            )


def first_available_recovery(state: dict) -> list[dict] | None:
    """Reference mechanism test: branch every deferred slot from one point.

    This is intentionally not a search or training policy.  It exists so an
    end-to-end deployment can exercise checkpoint/fork deterministically before
    installing its own policy callable.
    """
    request = state.get("request") or {}
    session_id = None
    if request.get("session_server_endpoint"):
        # The stable Miles session id is derived by the adapter and therefore
        # is not exposed on the wire. A usable point is still identifiable by
        # requiring a complete, internally consistent model position; the
        # driver performs the exact session-id equality check before admission.
        session_id = True

    def usable(point):
        if point.get("available") is not True:
            return False
        if session_id:
            position = point.get("model_position")
            return (
                isinstance(position, dict)
                and isinstance(position.get("session_id"), str)
                and bool(position["session_id"])
                and isinstance(position.get("response_id"), str)
                and bool(position["response_id"])
                and isinstance(position.get("token_sha256"), str)
                and bool(position["token_sha256"])
            )
        return True

    sources = [
        {**sample, "recovery_points": [
            point for point in sample.get("recovery_points", []) if usable(point)
        ]}
        for sample in state.get("samples", [])
        if any(usable(point) for point in sample.get("recovery_points", []))
    ]
    if not sources:
        # A deferred slot cannot become a branch after every admitted source
        # has terminated without publishing a recovery point. Release it so a
        # failed parent makes the execution group terminal instead of leaving
        # the caller polling forever.
        admitted = [
            sample for sample in state.get("samples", [])
            if sample.get("sample_slot_id")
            not in set(state.get("deferred_sample_slot_ids", []))
        ]
        if admitted and all(
            sample.get("state") in {
                "succeeded", "failed", "quarantined", "cancelled", "skipped"
            }
            for sample in admitted
        ):
            return [
                {
                    "sample_slot_id": slot_id,
                    "decision": {"kind": "skip"},
                }
                for slot_id in state.get("deferred_sample_slot_ids", [])
            ]
        return None
    source = sources[0]
    point = next(
        iter(source["recovery_points"]),
        None,
    )
    if point is None:
        return None
    return [
        {
            "sample_slot_id": slot_id,
            "decision": {
                "kind": "branch",
                "source_sample_slot_id": source["sample_slot_id"],
                "point_id": point["id"],
                "overrides": {
                    "prompt": (
                        "Continue from the restored tool result and complete "
                        "the task without repeating completed actions."
                    )
                },
            },
        }
        for slot_id in state.get("deferred_sample_slot_ids", [])
    ]
