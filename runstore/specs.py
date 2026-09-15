"""JSON contracts; leases and results deliberately do not belong in RunSpec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
import re
from typing import Any

from harness.orchestrator.run import RunSpec


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def no_credentials(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            secret = any(part in normalized for part in (
                "api_key", "auth_token", "authorization", "password", "bearer_token", "secret"))
            reference = isinstance(item, dict) and set(item) == {"$env"}
            if secret and item and not reference and not normalized.endswith("_env"):
                raise ValueError("Use a worker-side $env reference instead of inline credentials")
            no_credentials(item)
    elif isinstance(value, list):
        for item in value:
            no_credentials(item)


def validate_continuation(parent: dict, child: dict) -> None:
    mutable = {"prompt", "model", "timeout_s", "budget_usd"}
    parent_spec = {key: value for key, value in parent["spec"].items() if key not in mutable}
    child_spec = {key: value for key, value in child["spec"].items() if key not in mutable}
    if (parent["kind"] != "rollout" or child["kind"] != "rollout"
            or parent["profile"] != child["profile"] or parent_spec != child_spec):
        raise ValueError("Continuation must retain its source environment, slot and tool configuration")


@dataclass(frozen=True)
class GradeSpec:
    benchmark: str
    instance_id: str
    snapshot_id: str
    dataset_path: str
    dataset_sha256: str
    grader_revision: str
    backend: dict = field(default_factory=dict)
    resources: dict = field(default_factory=lambda: {"cpu": 2, "memory_mb": 12288})
    timeout_s: float = 3600
    verifier_network: str = "deny"
    harness_repo: str | None = None
    parser_path: str | None = None
    baseline_untracked: list[str] | None = None
    repository_workdir: str | None = None
    repository_base_commit: str | None = None


@dataclass(frozen=True)
class JobSpec:
    kind: str
    spec: dict
    profile: str = "default"
    context: dict = field(default_factory=dict)
    max_infra_retries: int = 2
    parent_point: str | None = None
    profile_hash: str | None = None

    def validate(self) -> dict:
        if not isinstance(self.spec, dict) or not isinstance(self.context, dict):
            raise ValueError("spec and context must be JSON objects")
        body = asdict(self)
        canonical(body)
        no_credentials(self.spec)
        if type(self.max_infra_retries) is not int or not 0 <= self.max_infra_retries <= 2:
            raise ValueError("max_infra_retries must be 0..2")
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("profile must be nonempty")
        if self.kind == "rollout":
            allowed = {item.name for item in fields(RunSpec)}
            if set(self.spec) - allowed:
                raise ValueError("Unknown RunSpec fields")
            if not isinstance(self.spec.get("prompt"), str) or not self.spec["prompt"].strip():
                raise ValueError("RunSpec requires a nonempty prompt")
            if self.spec.get("slot", "claude-code") not in {"claude-code", "codex"}:
                raise ValueError("v1 slots are claude-code and codex SDK")
            for name in ("session", "journal_path", "run_id", "sandbox_id", "mcp_url", "mcp_stdio_args"):
                if self.spec.get(name) is not None:
                    raise ValueError(f"Queued runs cannot supply {name}")
            if self.spec.get("cwd", ".") != ".":
                raise ValueError("Queued runs use an isolated worker-owned cwd")
            if self.spec.get("keep_sandbox"):
                raise ValueError("Queued runs retain snapshots, not live sandboxes")
            if self.spec.get("resume_session_id") or self.spec.get("fork"):
                raise ValueError("Use a validated parent_point for continuation")
            if not isinstance(self.spec.get("extra", {}), dict):
                raise ValueError("RunSpec extra must be a JSON object")
            if any(key in self.spec.get("extra", {}) for key in (
                    "native_prefix", "resume_session_id", "fork", "checkpoint_identity",
                    "repository_baseline_untracked")):
                raise ValueError("Native restoration fields are worker-owned")
        elif self.kind == "grade":
            grade = GradeSpec(**self.spec)
            if grade.benchmark not in {"swebench-verified", "swebench-pro", "swe-rebench-v2", "deepswe"}:
                raise ValueError("Unsupported official grader")
            if grade.verifier_network not in {"allow", "deny"}:
                raise ValueError("verifier_network must be allow or deny")
            if not all((grade.instance_id, grade.snapshot_id, grade.dataset_path,
                        grade.dataset_sha256, grade.grader_revision)):
                raise ValueError("Grading requires frozen task, snapshot and grader references")
            if grade.benchmark == "swe-rebench-v2" and not grade.parser_path:
                raise ValueError("SWE-rebench grading requires a deployment parser_path")
            if grade.benchmark == "swe-rebench-v2" and grade.baseline_untracked is None:
                raise ValueError(
                    "SWE-rebench grading requires the actor's frozen baseline_untracked"
                )
            if grade.benchmark == "swe-rebench-v2" and (
                not isinstance(grade.repository_workdir, str)
                or not grade.repository_workdir.startswith("/")
                or grade.repository_workdir == "/"
                or not isinstance(grade.repository_base_commit, str)
                or not re.fullmatch(
                    r"[0-9a-fA-F]{40,64}", grade.repository_base_commit
                )
            ):
                raise ValueError(
                    "SWE-rebench grading requires the actor's frozen repository workdir/base commit"
                )
            if grade.baseline_untracked is not None and (
                not isinstance(grade.baseline_untracked, list)
                or any(
                    not isinstance(path, str)
                    or not path
                    or path.startswith("/")
                    or path in {".", ".."}
                    or ".." in path.split("/")
                    for path in grade.baseline_untracked
                )
                or grade.baseline_untracked != sorted(set(grade.baseline_untracked))
            ):
                raise ValueError(
                    "baseline_untracked must be sorted unique repository-relative paths"
                )
            if grade.benchmark == "deepswe" and not grade.harness_repo:
                raise ValueError("DeepSWE grading requires a deployment harness_repo")
        else:
            raise ValueError("kind must be rollout or grade")
        timeout = self.spec.get("timeout_s", 3600)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be finite and positive")
        return body

    @classmethod
    def from_dict(cls, body: dict) -> JobSpec:
        result = cls(**body)
        result.validate()
        return result
