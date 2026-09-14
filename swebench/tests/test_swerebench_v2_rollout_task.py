from __future__ import annotations

import json
import sys
import threading
import time
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from swebench.models import CommandOutcome, ToolResult
from swebench.rollout_groups.protocol import GeneratedSpan, RolloutGroupRequest, Trajectory
from swebench.rollout_groups.runner import RolloutCancelled, RolloutContext
from swebench.rollout_groups.tasks.swerebench_v2 import (
    SWERebenchV2TaskAdapter,
    SWERebenchV2TaskCatalog,
    SWERebenchV2TaskRecord,
    _PreparedTask,
)


def _record(**overrides):
    value = {
        "task_id": "owner__repo-1",
        "environment_ref": {
            "kind": "image",
            "id": "registry.example/owner/repo",
            "revision": "sha256:" + "a" * 64,
            "resource_profile": "standard",
        },
        "repo": "owner/repo",
        "workdir": "/repo",
        "base_commit": "abc123",
        "test_patch": "diff --git a/test.py b/test.py\n",
        "FAIL_TO_PASS": ["test.py::test_fix"],
        "PASS_TO_PASS": ["test.py::test_old"],
        "install_config": {"test_cmd": "pytest -q", "log_parser": "parse_fake"},
    }
    value.update(overrides)
    return value


def _request(**overrides):
    body = {
        "rollout_job_id": "job",
        "rollout_id": 0,
        "prompt_group_id": "group",
        "task_id": "owner__repo-1",
        "environment_ref": _record()["environment_ref"],
        "sample_slots": [{"sample_slot_id": "slot", "sample_index": 0}],
        "max_samples": 1,
        "minimum_returned_samples": 1,
        "prompt": [{"role": "user", "content": "public issue only"}],
        "prompt_token_ids": [1],
        "model_endpoint": "http://model",
        "expected_weight_version": "1",
        "return_rollout_logprobs": False,
        "sampling_params": {},
        "budgets": {
            "max_model_calls": 2,
            "max_tool_calls": 2,
            "max_wall_time_seconds": 10,
        },
    }
    body.update(overrides)
    return RolloutGroupRequest.from_dict(body)


def _trajectory():
    return Trajectory(
        sample_slot_id="slot",
        branch_id="root",
        messages=[{"role": "user", "content": "public issue only"}],
        token_ids=[1, 2],
        prompt_length=1,
        generated_spans=[GeneratedSpan("r", 1, 2, (1,), (2,), "1", "stop")],
        response_text="done",
    )


def test_catalog_rejects_unknown_task_and_environment_mismatch():
    catalog = SWERebenchV2TaskCatalog(
        [SWERebenchV2TaskRecord.from_dict(_record())]
    )
    adapter = SWERebenchV2TaskAdapter(catalog)

    with pytest.raises(ValueError, match="unknown"):
        adapter.validate_request(_request(task_id="missing"))

    mismatch = _record()["environment_ref"] | {"id": "registry.example/other"}
    with pytest.raises(ValueError, match="does not match"):
        adapter.validate_request(_request(environment_ref=mismatch))


def test_jsonl_catalog_and_hidden_fields_stay_server_side(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_record()) + "\n")
    catalog = SWERebenchV2TaskCatalog.from_file(path)
    adapter = SWERebenchV2TaskAdapter(catalog)
    request = _request()
    adapter.validate_request(request)

    wire = json.dumps(request.to_dict())
    assert "test_patch" not in wire
    assert "FAIL_TO_PASS" not in wire
    assert "pytest -q" not in wire


def test_catalog_preserves_patch_trailing_newline():
    value = _record(test_patch="diff --git a/a b/a\n+x\n")
    record = SWERebenchV2TaskRecord.from_dict(value)

    assert record.test_patch == value["test_patch"]


def test_evaluation_uses_official_parser_and_returns_aggregate_metadata(monkeypatch):
    module = ModuleType("fake_swerebench_parsers")

    class TestStatus(Enum):
        PASSED = "PASSED"

    module.TestStatus = TestStatus
    module.NAME_TO_PARSER = {
        "parse_fake": lambda _output: {
            "test.py::test_fix": "PASSED",
            "test.py::test_old": "PASSED",
        }
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    record = SWERebenchV2TaskRecord.from_dict(_record())
    adapter = SWERebenchV2TaskAdapter(
        SWERebenchV2TaskCatalog([record]), parser_module=module.__name__
    )
    monkeypatch.setattr(
        adapter,
        "_capture_patch",
        lambda *_args, **_kwargs: "diff --git a/a b/a\n+x\n",
    )
    monkeypatch.setattr(
        adapter,
        "_run_hidden_tests",
        lambda *_args, **_kwargs: "SWEREBENCH_V2_TEST_OUTPUT_START\nok\nSWEREBENCH_V2_TEST_OUTPUT_END",
    )

    result = adapter.evaluate(
        _request(), SimpleNamespace(), _trajectory(), _PreparedTask(record, ())
    )

    assert result.reward == 1.0
    assert result.metadata["f2p_passed"] == 1
    assert result.metadata["p2p_passed"] == 1
    assert "test_patch" not in result.metadata
    assert "test.py::test_fix" not in json.dumps(result.metadata)


def test_hidden_test_restore_command_preserves_shell_quote_escapes(monkeypatch):
    record = SWERebenchV2TaskRecord.from_dict(_record())
    adapter = SWERebenchV2TaskAdapter(SWERebenchV2TaskCatalog([record]))
    commands = []

    def shell(_sandbox, command, **_kwargs):
        commands.append(command)
        return ""

    monkeypatch.setattr(adapter, "_shell", shell)
    adapter._run_hidden_tests(SimpleNamespace(), record)

    restore = commands[1]
    assert restore.startswith("bash -lc ")
    assert 'path=${path#\\"}; path=${path%\\"};' in restore


def test_hidden_test_restore_runs_pipefail_under_bash(tmp_path):
    """Regression: /bin/sh may be dash and reject `set -o pipefail`."""
    workdir = tmp_path / "repo"
    workdir.mkdir()
    record = SWERebenchV2TaskRecord.from_dict(
        _record(
            workdir=str(workdir),
            test_patch="diff --git a/test.py b/test.py\n",
        )
    )
    adapter = SWERebenchV2TaskAdapter(SWERebenchV2TaskCatalog([record]))
    commands = []

    def shell(_sandbox, command, **_kwargs):
        commands.append(command)
        return ""

    adapter._shell = shell
    adapter._run_hidden_tests(SimpleNamespace(), record)

    assert commands[1].startswith("bash -lc ")
    # The outer shell only sees one quoted Bash command; pipefail is not
    # interpreted by /bin/sh.
    assert "'set -euo pipefail; " in commands[1]


def test_shell_caps_command_and_transport_timeout_to_rollout_deadline():
    seen = {}

    class Sandbox:
        def execute(self, name, args, timeout):
            seen.update(name=name, args=args, timeout=timeout)
            return ToolResult(success=True, output="ok")

    context = RolloutContext(
        cancel_event=threading.Event(),
        model_client=None,
        environment_provider=None,
        job_id="job",
        deadline=time.monotonic() + 20,
    )

    assert SWERebenchV2TaskAdapter._shell(
        Sandbox(), "pytest", context=context
    ) == "ok"
    assert 1 <= seen["timeout"] <= 20
    assert 1 <= seen["args"]["timeout"] <= seen["timeout"] - 4


def test_shell_timeout_preserves_structured_outcome_and_cancels_rollout():
    class Sandbox:
        def execute(self, _name, _args, timeout):
            assert timeout > 0
            return ToolResult(
                success=False,
                output="",
                outcome=CommandOutcome(
                    exit_code=137,
                    stderr="test exceeded deadline",
                    timed_out=True,
                ),
            )

    cancel_event = threading.Event()
    context = RolloutContext(
        cancel_event=cancel_event,
        model_client=None,
        environment_provider=None,
        job_id="job",
        deadline=time.monotonic() + 20,
    )

    with pytest.raises(
        RolloutCancelled,
        match="exit_code=137.*timed_out=True.*test exceeded deadline",
    ):
        SWERebenchV2TaskAdapter._shell(
            Sandbox(), "pytest", context=context
        )
    assert cancel_event.is_set()


def test_shell_failure_without_backend_text_is_diagnostic():
    class Sandbox:
        def execute(self, _name, _args):
            return ToolResult(success=False, output="")

    with pytest.raises(RuntimeError, match="no error details returned"):
        SWERebenchV2TaskAdapter._shell(Sandbox(), "pytest")


def test_shell_empty_failure_after_deadline_is_cancellation():
    class Sandbox:
        def execute(self, _name, _args, timeout):
            return ToolResult(success=False, output="")

    cancel_event = threading.Event()
    context = RolloutContext(
        cancel_event=cancel_event,
        model_client=None,
        environment_provider=None,
        job_id="job",
        deadline=time.monotonic() - 1,
    )

    with pytest.raises(RolloutCancelled, match="wall-time budget exhausted"):
        SWERebenchV2TaskAdapter._shell(Sandbox(), "pytest", context=context)
    assert cancel_event.is_set()
