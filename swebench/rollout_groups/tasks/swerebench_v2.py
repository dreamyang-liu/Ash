"""Private SWE-rebench-V2 task setup and official verifier adapter.

Miles sends only a public ``task_id`` and immutable ``environment_ref``.  This
module resolves them against a deployment-owned task catalog containing the
hidden test patch, expected tests and parser selection.  Hidden fields are
used only after the agent finishes and never enter model-visible messages.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...patch import select_added
from ..protocol import EnvironmentRef, RolloutGroupRequest, Trajectory
from ..runner import RolloutCancelled, RolloutContext, TaskEvaluation


_TEST_START = "SWEREBENCH_V2_TEST_OUTPUT_START"
_TEST_END = "SWEREBENCH_V2_TEST_OUTPUT_END"

_ENV = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/cargo/bin:/usr/local/go/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "CI": "1",
    "_JAVA_OPTIONS": "-Djava.net.preferIPv6Addresses=false",
}


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _required_text(value: Any, name: str) -> str:
    """Validate opaque text without changing byte-sensitive patch content."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must be a JSON array or list") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


def _install_config(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("install_config must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("install_config must be an object")
    return dict(value)


@dataclass(frozen=True)
class SWERebenchV2TaskRecord:
    task_id: str
    environment_ref: EnvironmentRef
    repo: str
    workdir: str
    base_commit: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    install_config: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> "SWERebenchV2TaskRecord":
        if not isinstance(value, dict):
            raise ValueError("SWE-rebench-V2 task entries must be objects")
        allowed = {
            "task_id",
            "instance_id",
            "environment_ref",
            "repo",
            "workdir",
            "base_commit",
            "test_patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "install_config",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"SWE-rebench-V2 task entry contains unknown fields: {sorted(unknown)}"
            )
        task_id = value.get("task_id", value.get("instance_id"))
        workdir = _required_string(value.get("workdir"), "workdir")
        if not workdir.startswith("/"):
            raise ValueError("workdir must be absolute")
        return cls(
            task_id=_required_string(task_id, "task_id"),
            environment_ref=EnvironmentRef.from_dict(value.get("environment_ref")),
            repo=_required_string(value.get("repo"), "repo"),
            workdir=workdir,
            base_commit=_required_string(value.get("base_commit"), "base_commit"),
            test_patch=_required_text(value.get("test_patch"), "test_patch"),
            fail_to_pass=_string_list(value.get("FAIL_TO_PASS"), "FAIL_TO_PASS"),
            pass_to_pass=_string_list(value.get("PASS_TO_PASS"), "PASS_TO_PASS"),
            install_config=_install_config(value.get("install_config")),
        )


class SWERebenchV2TaskCatalog:
    """Deployment-private task records keyed by public ``task_id``."""

    def __init__(self, records: list[SWERebenchV2TaskRecord]) -> None:
        if not records:
            raise ValueError("SWE-rebench-V2 task catalog must not be empty")
        self._records: dict[str, SWERebenchV2TaskRecord] = {}
        for record in records:
            if record.task_id in self._records:
                raise ValueError(f"duplicate SWE-rebench-V2 task_id: {record.task_id!r}")
            self._records[record.task_id] = record

    @classmethod
    def from_file(cls, path: str | Path) -> "SWERebenchV2TaskCatalog":
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            if source.suffix == ".jsonl":
                values = [
                    json.loads(line)
                    for line in handle
                    if line.strip()
                ]
            else:
                raw = json.load(handle)
                values = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(values, list):
            raise ValueError("task catalog must be a JSON array, {tasks: [...]}, or JSONL")
        return cls([SWERebenchV2TaskRecord.from_dict(item) for item in values])

    def get(self, task_id: str) -> SWERebenchV2TaskRecord:
        try:
            return self._records[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown SWE-rebench-V2 task_id: {task_id!r}") from exc


@dataclass(frozen=True)
class _PreparedTask:
    record: SWERebenchV2TaskRecord
    baseline_untracked: tuple[str, ...]


class SWERebenchV2TaskAdapter:
    """Prepare and grade SWE-rebench-V2 tasks in an active Ash sandbox."""

    def __init__(
        self,
        catalog: SWERebenchV2TaskCatalog,
        *,
        patch_dir: str | Path | None = None,
        parser_module: str = "swerebench_v2.log_parsers",
        parser_file: str | Path | None = None,
        max_test_output_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.catalog = catalog
        self.patch_dir = None if patch_dir is None else Path(patch_dir)
        self.parser_module = parser_module
        self.parser_file = None if parser_file is None else Path(parser_file)
        self.max_test_output_bytes = max_test_output_bytes
        if self.patch_dir is not None:
            self.patch_dir.mkdir(parents=True, exist_ok=True)

    def validate_request(self, request: RolloutGroupRequest) -> None:
        record = self.catalog.get(request.task_id)
        if request.environment_ref != record.environment_ref:
            raise ValueError(
                "environment_ref does not match the private task catalog for "
                f"task_id {request.task_id!r}"
            )

    def prepare(
        self,
        request: RolloutGroupRequest,
        sandbox: Any,
        context: RolloutContext | None = None,
    ) -> _PreparedTask:
        record = self.catalog.get(request.task_id)
        self.validate_request(request)
        workdir = shlex.quote(record.workdir)
        head = self._shell(
            sandbox, f"git -C {workdir} rev-parse HEAD", context=context
        )
        if head.strip() != record.base_commit:
            raise RuntimeError(
                f"task image has unexpected base commit for {record.task_id!r}"
            )
        if record.workdir != "/testbed":
            # Ash's default tool prompts and submission hooks use /testbed.
            # Official SWE-rebench images may place the checkout at /<repo>.
            target = shlex.quote(record.workdir)
            self._shell(
                sandbox,
                "if [ -e /testbed ] || [ -L /testbed ]; then "
                f"[ \"$(readlink -f /testbed)\" = \"$(readlink -f {target})\" ]; "
                f"else ln -s {target} /testbed; fi",
                context=context,
            )
        untracked = self._shell(
            sandbox,
            f"git -C {workdir} ls-files --others --exclude-standard",
            context=context,
        )
        return _PreparedTask(
            record=record,
            baseline_untracked=tuple(
                line.strip() for line in untracked.splitlines() if line.strip()
            ),
        )

    def evaluate(
        self,
        request: RolloutGroupRequest,
        sandbox: Any,
        trajectory: Trajectory,
        prepared: _PreparedTask,
        context: RolloutContext | None = None,
    ) -> TaskEvaluation:
        if context is not None:
            context.check_cancelled()
        started = time.monotonic()
        record = prepared.record
        patch = self._capture_patch(sandbox, prepared, context=context)
        patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
        self._save_patch(request, trajectory, patch)

        output = self._run_hidden_tests(sandbox, record, context=context)
        parser_name = _required_string(
            record.install_config.get("log_parser"),
            "install_config.log_parser",
        )
        parsers = self._load_parsers()
        parser = getattr(parsers, "NAME_TO_PARSER", {}).get(parser_name)
        if parser is None:
            parser = getattr(parsers, parser_name, None)
        if not callable(parser):
            raise RuntimeError(f"unknown official log parser: {parser_name!r}")
        status_map = parser(self._test_region(output)) or {}
        if not isinstance(status_map, dict):
            raise RuntimeError(f"official log parser {parser_name!r} returned a non-object")
        normalized = {
            self._normalize_test_name(name): status
            for name, status in status_map.items()
        }
        passed_value = getattr(parsers.TestStatus.PASSED, "value", "PASSED")
        f2p_passed, f2p_missing = self._counts(
            normalized, record.fail_to_pass, passed_value
        )
        p2p_passed, p2p_missing = self._counts(
            normalized, record.pass_to_pass, passed_value
        )
        expected = len(record.fail_to_pass) + len(record.pass_to_pass)
        reward = float(
            expected > 0
            and f2p_passed == len(record.fail_to_pass)
            and p2p_passed == len(record.pass_to_pass)
        )

        metadata: dict[str, Any] = {
            "benchmark": "SWE-rebench-V2",
            "verifier_status": "completed",
            "verifier_seconds": round(time.monotonic() - started, 3),
            "patch_sha256": patch_sha256,
            "patch_chars": len(patch),
            "patch_files_changed": sum(
                1 for line in patch.splitlines() if line.startswith("diff --git ")
            ),
            "patch_lines_changed": sum(
                1
                for line in patch.splitlines()
                if line.startswith(("+", "-"))
                and not line.startswith(("+++", "---"))
            ),
            "f2p_total": len(record.fail_to_pass),
            "f2p_passed": f2p_passed,
            "f2p_missing": len(f2p_missing),
            "p2p_total": len(record.pass_to_pass),
            "p2p_passed": p2p_passed,
            "p2p_missing": len(p2p_missing),
        }
        return TaskEvaluation(reward=reward, metadata=metadata)

    def _capture_patch(
        self,
        sandbox: Any,
        prepared: _PreparedTask,
        *,
        context: RolloutContext | None = None,
    ) -> str:
        record = prepared.record
        workdir = shlex.quote(record.workdir)
        current = self._shell(
            sandbox,
            f"git -C {workdir} ls-files --others --exclude-standard",
            context=context,
        )
        baseline = set(prepared.baseline_untracked)
        added = select_added(
            (line.strip() for line in current.splitlines() if line.strip()),
            baseline,
        )
        index = "/tmp/ash-swerebench-index"
        commands = [
            f"rm -f {index}",
            f"GIT_INDEX_FILE={index} git -C {workdir} read-tree {shlex.quote(record.base_commit)}",
            f"GIT_INDEX_FILE={index} git -C {workdir} add -u",
        ]
        if added:
            paths = " ".join(shlex.quote(path) for path in added)
            commands.append(
                f"GIT_INDEX_FILE={index} git -C {workdir} add -- {paths}"
            )
        commands.append(
            f"GIT_INDEX_FILE={index} git -C {workdir} diff --cached --binary "
            f"{shlex.quote(record.base_commit)}"
        )
        return self._shell(
            sandbox,
            " && ".join(commands),
            max_output_bytes=64 * 1024 * 1024,
            context=context,
        )

    def _run_hidden_tests(
        self,
        sandbox: Any,
        record: SWERebenchV2TaskRecord,
        *,
        context: RolloutContext | None = None,
    ) -> str:
        workdir = shlex.quote(record.workdir)
        patch_path = "/tmp/ash-swerebench-test.patch"
        self._shell(
            sandbox,
            f"cat > {patch_path}",
            stdin=record.test_patch,
            context=context,
        )
        restore = (
            "set -euo pipefail; "
            f"git -C {workdir} apply --numstat {patch_path} | cut -f3- | "
            "while IFS= read -r path; do "
            "path=${path#\\\"}; path=${path%\\\"}; "
            f"if git -C {workdir} cat-file -e {shlex.quote(record.base_commit)}:\"$path\" 2>/dev/null; "
            f"then git -C {workdir} checkout {shlex.quote(record.base_commit)} -- \"$path\"; "
            f"else rm -f -- {workdir}/\"$path\"; fi; done"
        )
        # AgentENV's shell tool executes commands through /bin/sh.  On
        # Debian/Ubuntu that is normally dash, which deliberately does not
        # implement `set -o pipefail`.  This verifier fragment needs Bash's
        # pipeline failure semantics, so select Bash explicitly instead of
        # relying on the sandbox's default shell.
        self._shell(
            sandbox, f"bash -lc {shlex.quote(restore)}", context=context
        )
        apply_commands = (
            f"cd {workdir} && "
            f"(git apply -v --3way --recount --ignore-space-change --whitespace=nowarn {patch_path} "
            f"|| patch --fuzz=5 -p1 -i {patch_path})"
        )
        self._shell(sandbox, apply_commands, context=context)

        raw_commands = record.install_config.get("test_cmd")
        if isinstance(raw_commands, str):
            test_commands = [raw_commands]
        elif isinstance(raw_commands, list):
            test_commands = [item for item in raw_commands if isinstance(item, str) and item.strip()]
        else:
            raise RuntimeError("install_config.test_cmd must be a string or list")
        if not test_commands:
            raise RuntimeError("install_config.test_cmd is empty")
        script = ["#!/bin/bash", "set -uo pipefail", f'echo "{_TEST_START}"', "FAIL=0"]
        script.extend(f"{command} || FAIL=1" for command in test_commands)
        script.extend([f'echo "{_TEST_END}"', 'exit "$FAIL"', ""])
        self._shell(
            sandbox,
            "cat > /tmp/ash-swerebench-eval.sh",
            stdin="\n".join(script),
            context=context,
        )
        self._shell(
            sandbox, "chmod +x /tmp/ash-swerebench-eval.sh", context=context
        )
        # A test failure is expected and is encoded by the parser.  Force the
        # shell transport to succeed so its complete output remains available.
        return self._shell(
            sandbox,
            f"cd {workdir} && bash /tmp/ash-swerebench-eval.sh 2>&1 || true",
            env=_ENV,
            max_output_bytes=self.max_test_output_bytes,
            context=context,
        )

    def _save_patch(
        self,
        request: RolloutGroupRequest,
        trajectory: Trajectory,
        patch: str,
    ) -> None:
        if self.patch_dir is None:
            return
        safe_job = hashlib.sha256(request.rollout_job_id.encode()).hexdigest()[:16]
        safe_slot = hashlib.sha256(trajectory.sample_slot_id.encode()).hexdigest()[:16]
        target = self.patch_dir / f"{safe_job}-{safe_slot}.patch"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(patch, encoding="utf-8")
        temporary.replace(target)

    def _load_parsers(self):
        if self.parser_file is None:
            return importlib.import_module(self.parser_module)
        if not self.parser_file.is_file():
            raise RuntimeError(f"official parser file does not exist: {self.parser_file}")
        name = "_ash_swerebench_v2_official_log_parsers"
        spec = importlib.util.spec_from_file_location(name, self.parser_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load official parser file: {self.parser_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _test_region(output: str) -> str:
        region = output
        if _TEST_START in region:
            region = region.split(_TEST_START, 1)[1]
        if _TEST_END in region:
            region = region.rsplit(_TEST_END, 1)[0]
        return region

    @staticmethod
    def _normalize_test_name(name: str) -> str:
        # Official parsers already normalize most formats.  These suffixes are
        # the remaining timing decorations normalized by the taskset itself.
        import re

        patterns = (
            r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$",
            r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b",
            r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$",
        )
        for pattern in patterns:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)
        return name.strip()

    @classmethod
    def _counts(
        cls,
        statuses: dict[str, str],
        expected: tuple[str, ...],
        passed_value: str,
    ) -> tuple[int, list[str]]:
        missing = [
            name
            for name in expected
            if statuses.get(cls._normalize_test_name(name)) != passed_value
        ]
        return len(expected) - len(missing), missing

    @staticmethod
    def _shell(
        sandbox: Any,
        command: str,
        *,
        stdin: str | None = None,
        env: dict[str, str] | None = None,
        max_output_bytes: int | None = None,
        context: RolloutContext | None = None,
    ) -> str:
        if context is not None:
            context.check_cancelled()
        args: dict[str, Any] = {"command": command}
        if stdin is not None:
            args["stdin"] = stdin
        if env is not None:
            args["env"] = env
        if max_output_bytes is not None:
            args["max_output_bytes"] = max_output_bytes
        transport_timeout = None
        if context is not None:
            remaining = context.remaining_wall_time_seconds
            if remaining is not None:
                if remaining <= 0:
                    context.check_cancelled()
                # Leave a small response window after the command timeout so
                # ash-runtime can return its structured CommandOutcome before
                # the transport itself reaches the rollout deadline.
                transport_timeout = remaining
                command_timeout = max(1, int(max(1.0, remaining - 5.0)))
                args["timeout"] = command_timeout
        result = (
            sandbox.execute("shell", args)
            if transport_timeout is None
            else sandbox.execute("shell", args, timeout=transport_timeout)
        )
        outcome = getattr(result, "outcome", None)
        exit_code = getattr(outcome, "exit_code", None)
        if not getattr(result, "success", False) or exit_code not in {None, 0}:
            detail = _shell_failure_detail(result, outcome)
            if outcome is not None and getattr(outcome, "timed_out", False):
                if context is not None:
                    context.cancel_event.set()
                raise RolloutCancelled(f"sandbox command timed out: {detail}")
            if context is not None and (
                context.cancel_event.is_set()
                or context.remaining_wall_time_seconds == 0
            ):
                context.cancel_event.set()
                raise RolloutCancelled(
                    f"sandbox command was cancelled at rollout deadline: {detail}"
                )
            raise RuntimeError(f"sandbox command failed: {detail}")
        if outcome is not None and getattr(outcome, "stdout", None) is not None:
            return outcome.stdout or ""
        return getattr(result, "output", "") or ""


def _shell_failure_detail(result: Any, outcome: Any) -> str:
    """Keep the command outcome useful even when the transport text is empty."""
    fields: list[str] = []
    error = getattr(result, "error", None)
    output = getattr(result, "output", None)
    if error:
        fields.append(f"error={error}")
    if outcome is not None:
        for name in ("exit_code", "timed_out", "running"):
            value = getattr(outcome, name, None)
            if value is not None:
                fields.append(f"{name}={value}")
        stdout = getattr(outcome, "stdout", "") or ""
        stderr = getattr(outcome, "stderr", "") or ""
        if stdout:
            fields.append(f"stdout={stdout[-500:]}")
        if stderr:
            fields.append(f"stderr={stderr[-500:]}")
    elif output:
        fields.append(f"output={str(output)[-500:]}")
    return "; ".join(fields) or "no error details returned by sandbox backend"
