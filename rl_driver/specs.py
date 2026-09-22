"""Execution-driver contracts, deliberately distinct from Miles token samples."""

from copy import deepcopy
import hashlib
import re

from runstore.specs import JobSpec, no_credentials
from runstore.branch_guidance import validate_overrides
from rl_driver.ledger import canonical

PROTOCOL_VERSION = "ash-runstore-driver-v1"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")


def identifier(value, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a URL-safe identifier of 1..200 characters")
    return value


def object_fields(value, allowed: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"Invalid {name} fields; allowed: {sorted(allowed)}")


def validate_request(body: dict) -> dict:
    object_fields(body, {"protocol_version", "rollout_job_id", "prompt_group_id", "context", "samples"}, "group")
    if body.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ValueError(f"Use {PROTOCOL_VERSION}; ash-rollout-v2 requires a separate token-trace adapter")
    result = deepcopy(body)
    result["protocol_version"] = PROTOCOL_VERSION
    identifier(result.get("rollout_job_id"), "rollout_job_id")
    identifier(result.get("prompt_group_id"), "prompt_group_id")
    result.setdefault("context", {})
    if not isinstance(result["context"], dict):
        raise ValueError("context must be an object")
    samples = result.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 1000:
        raise ValueError("samples must contain 1..1000 entries")
    seen = set()
    for sample in samples:
        object_fields(sample, {"sample_slot_id", "run", "branch", "grade"}, "sample")
        slot = identifier(sample.get("sample_slot_id"), "sample_slot_id")
        if slot in seen:
            raise ValueError("sample_slot_id values must be unique")
        seen.add(slot)
        if ("run" in sample) == ("branch" in sample):
            raise ValueError("Each sample requires exactly one run or branch")
        if "run" in sample:
            if not isinstance(sample["run"], dict):
                raise ValueError("run must be a JobSpec object")
            job = JobSpec.from_dict({"kind": "rollout", **sample["run"]})
            if job.kind != "rollout" or job.parent_point:
                raise ValueError("run must be a rollout; use branch for continuation")
            sample["run"] = job.validate()
        else:
            branch = sample["branch"]
            object_fields(branch, {"job_id", "point_id", "overrides"}, "branch")
            identifier(branch.get("job_id"), "branch.job_id")
            identifier(branch.get("point_id"), "branch.point_id")
            branch.setdefault("overrides", {})
            validate_overrides(branch["overrides"])
        if "grade" in sample:
            grade = sample["grade"]
            if not isinstance(grade, dict) or not isinstance(grade.get("spec"), dict):
                raise ValueError("grade must be a GradeSpec job template")
            if "snapshot_id" in grade["spec"]:
                raise ValueError("grade snapshot_id is selected from the actor's exact final boundary")
            job = JobSpec.from_dict({**grade, "kind": "grade",
                                     "spec": {**grade["spec"], "snapshot_id": "driver-pending-snapshot"}})
            if grade.get("kind", "grade") != "grade" or job.parent_point:
                raise ValueError("grade must be an independent grading job")
            sample["grade"] = job.validate()
            del sample["grade"]["spec"]["snapshot_id"]
    no_credentials(result)
    canonical(result)
    return result


def submission_key(group_id: str, sample_id: str, phase: str) -> str:
    return "rl-driver/" + hashlib.sha256(canonical([group_id, sample_id, phase]).encode()).hexdigest()


def initial_document(request: dict) -> dict:
    samples = []
    for spec in request["samples"]:
        actor = {"state": "planned", "job_id": None, "attempt_id": None,
                 "operation": "branch" if "branch" in spec else "run"}
        if "branch" in spec:
            actor["submission"] = spec["branch"]
            actor["origin"] = {key: spec["branch"][key] for key in ("job_id", "point_id")}
        else:
            job = deepcopy(spec["run"])
            job["context"]["rl_driver"] = {
                "rollout_job_id": request["rollout_job_id"],
                "prompt_group_id": request["prompt_group_id"],
                "sample_slot_id": spec["sample_slot_id"], "context": request["context"],
            }
            actor["submission"] = job
        samples.append({"sample_slot_id": spec["sample_slot_id"], "actor": actor,
                        "grade": ({"state": "planned", "job_id": None, "attempt_id": None,
                                   "operation": "grade", "template": spec["grade"]}
                                  if "grade" in spec else None)})
    return {"protocol_version": PROTOCOL_VERSION, "rollout_job_id": request["rollout_job_id"],
            "prompt_group_id": request["prompt_group_id"], "status": "queued",
            "context": request["context"], "ready": False, "samples": samples}
