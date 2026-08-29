

# --- bare-name test ids (sympy) ------------------------------------------------
def test_sympy_needs_the_file_runner_and_others_do_not():
    """sympy reports BARE test-function names for all 75 of its Verified
    instances. Handed to pytest as paths they collect nothing and the run fails
    whatever the agent did -- a live 7-attempt branching experiment scored every
    attempt zero on exactly that, and the images ship no pytest either."""
    from swebench.dataset import needs_file_runner

    assert needs_file_runner("sympy/sympy", ["test_equality"])
    assert not needs_file_runner("sympy/sympy", ["a/b.py::test_x"])
    assert not needs_file_runner("scikit-learn/scikit-learn", ["a/b.py::test_x"])
    assert not needs_file_runner("django/django", ["test_x (mod.Cls)"])


def test_the_batch_builder_refuses_ids_it_cannot_express():
    """Silently emitting a pytest command for bare names is what produced the
    false zeros; the caller must route them to the file runner instead."""
    import pytest

    from swebench.dataset import build_batch_test_command

    with pytest.raises(ValueError, match="file-based runner"):
        build_batch_test_command("sympy/sympy", ["test_equality"])


def test_the_sympy_runner_distinguishes_ran_from_matched_nothing():
    """The property every earlier attempt lacked. `bin/test -k` and
    `sympy.test(...)` both return success for a run that matched NOTHING, so a
    grader trusting them scores an untouched repository as resolved (verified
    live: a pre-edit snapshot "passed"). Exit 2 means the grader is broken, which
    is not the same answer as a failing test."""
    import subprocess
    import sys
    import textwrap

    from swebench.dataset import SYMPY_RUNNER, sympy_runner_spec

    def run(tmp, module_src, kw):
        (tmp / "fake_mod.py").write_text(module_src)
        spec = tmp / "spec.json"
        spec.write_text(json.dumps(sympy_runner_spec(kw, ["fake_mod.py"])))
        script = tmp / "runner.py"
        script.write_text(SYMPY_RUNNER)
        return subprocess.run([sys.executable, str(script), str(spec)],
                              cwd=str(tmp), capture_output=True, text=True)

    import json
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)
        passing = run(tmp, "def test_ok():\n    assert True\n", ["test_ok"])
        assert passing.returncode == 0
        assert "ran 1, failed 0" in passing.stdout

        failing = run(tmp, "def test_ok():\n    assert False\n", ["test_ok"])
        assert failing.returncode == 1, "a failing test is exit 1"

        absent = run(tmp, "def test_other():\n    pass\n", ["test_ok"])
        assert absent.returncode == 2, "matched nothing must NOT read as a pass"

        skipped = run(tmp, textwrap.dedent('''
            class Skipped(Exception):
                pass

            def test_ok():
                raise Skipped("nope")
            '''), ["test_ok"])
        assert skipped.returncode == 0, "a skip is not a failure of the change"
