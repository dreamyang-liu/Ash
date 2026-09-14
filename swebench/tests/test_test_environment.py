import json
from types import SimpleNamespace

import pytest

from swebench.test_environment import CONDA_INIT, test_command as command_for_test, validate_test_execution


def result(exit_code, stderr="", **extra):
    return SimpleNamespace(success=True, error=None, output=json.dumps({
        "stdout": "", "stderr": stderr, "exit_code": exit_code, **extra}))


@pytest.mark.parametrize("outcome", [
    result(1, "/opt/miniconda3/bin/python: No module named pytest"),
    result(86, "ASH_SWE_TEST_ENVIRONMENT_ERROR"), result(5), result(4),
    result(3), result(127), result(None), result(137, timed_out=True),
    SimpleNamespace(success=False, output="", error="connection lost"),
])
def test_invalid_execution_is_not_an_assertion_failure(outcome):
    with pytest.raises(RuntimeError):
        validate_test_execution(outcome)


def test_real_test_failure_stays_a_failure_not_an_infrastructure_error():
    validate_test_execution(result(1, "FAILED test_x - AssertionError"))
    failed = result(1, "FAILED test_x - AssertionError")
    failed.success = False
    validate_test_execution(failed)
    validate_test_execution(result(2, "ERROR collecting test_x: SyntaxError"))
    validate_test_execution(result(0))


def test_verifier_activates_existing_environment_without_installing():
    command = command_for_test("cd /testbed && python -m pytest test_x")
    assert CONDA_INIT in command
    assert "pytest --version" in command
    assert "install" not in command


def test_actor_backend_consumes_same_activation(tmp_path):
    from swebench.fork_eval import SweBench, backend_for
    args = SimpleNamespace(runtime_bin=str(tmp_path / "runtime"), timeout=1800)
    assert backend_for(args, SweBench())["microvm"]["runtime_init"] == CONDA_INIT
