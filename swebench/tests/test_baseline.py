from copy import deepcopy

import pytest

from swebench.baseline import BaselineAdmissionError, PROBE, inspect_baseline, validate_baseline


def evidence():
    base, head, tree = "1" * 40, "2" * 40, "3" * 40
    return {"base": base, "head": head, "tree": tree, "parents": [base],
            "status": "", "added": [], "removed": [],
            "content_changes": ["pyproject.toml"], "setuptools_pin_only": True,
            "mode_changes": [{"path": "astropy/modeling/separable.py", "before": "100644", "after": "100755"}]}


def test_reviewed_preparation():
    report = evidence()
    instance = {"instance_id": "astropy__astropy-12907", "base_commit": report["base"]}
    assert validate_baseline(report, instance) == "known-astropy-preparation"
    compile(PROBE, "baseline-probe", "exec")


@pytest.mark.parametrize("change", [
    {"status": " M source.py"}, {"added": ["answer.py"]}, {"removed": ["test.py"]},
    {"base": "f" * 40}, {"parents": []},
    {"content_changes": ["astropy/modeling/separable.py"]}, {"setuptools_pin_only": False},
    {"mode_changes": [{"before": "120000", "after": "100755"}]},
])
def test_unexplained_changes_still_stop(change):
    report = evidence()
    instance = {"instance_id": "astropy__astropy-12907", "base_commit": report["base"]}
    report.update(deepcopy(change))
    with pytest.raises(ValueError):
        validate_baseline(report, instance)


def test_identical_tree_and_exact_base():
    report = evidence()
    instance = {"instance_id": "other", "base_commit": report["base"]}
    report.update(content_changes=[], mode_changes=[])
    assert validate_baseline(report, instance) == "content-identical-preparation"
    report["head"] = report["base"]
    assert validate_baseline(report, instance) == "exact-base"


def test_known_delta_is_not_a_global_exception():
    report = evidence()
    with pytest.raises(ValueError, match="unreviewed"):
        validate_baseline(report, {"instance_id": "another-task", "base_commit": report["base"]})


def test_same_preparation_is_checked_by_content_not_task_sha():
    report = evidence()
    report.update(head="f" * 40, tree="e" * 40)
    assert validate_baseline(report, {"instance_id": "astropy__astropy-13236", "base_commit": report["base"]}) == "known-astropy-preparation"


def test_only_executable_bit_preparation_is_accepted():
    report = evidence()
    report["content_changes"] = []
    assert validate_baseline(report, {"instance_id": "other", "base_commit": report["base"]}) == "mode-only-preparation"


def test_rejected_probe_keeps_evidence():
    import json
    from types import SimpleNamespace
    report = evidence()
    report["content_changes"] = ["source.py"]
    session = SimpleNamespace(execute=lambda *args: SimpleNamespace(success=True, output=json.dumps(report)))
    with pytest.raises(BaselineAdmissionError) as caught:
        inspect_baseline(session, {"instance_id": "other", "base_commit": report["base"]})
    assert caught.value.report["content_changes"] == ["source.py"]


def test_stderr_wrapper_is_not_mistaken_for_baseline_json():
    import json
    from types import SimpleNamespace
    report = evidence()
    report.update(content_changes=[], mode_changes=[])
    warning = "Error processing distutils-precedence.pth: No module named '_distutils_hack'"
    envelope = {"stdout": json.dumps(report), "stderr": warning, "exit_code": 0}
    session = SimpleNamespace(execute=lambda *args: SimpleNamespace(success=True, output=json.dumps(envelope)))
    result = inspect_baseline(session, {"instance_id": "pylint-dev__pylint-8898", "base_commit": report["base"]})
    assert result["admission"] == "content-identical-preparation"
    assert result["probe_stderr"] == warning


def test_native_structured_outcome_is_used():
    from types import SimpleNamespace
    from harness.core.result import CommandOutcome
    from swebench.baseline import probe_output
    result = SimpleNamespace(success=True, output="not the stdout", outcome=CommandOutcome(exit_code=0, stdout="json", stderr="warning"))
    assert probe_output(result) == ("json", "warning")
    result.outcome.timed_out = True
    with pytest.raises(ValueError):
        probe_output(result)


def repository(tmp_path, files):
    import subprocess
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    for command in [["init", "-q"], ["config", "user.name", "Test"], ["config", "user.email", "test@example.invalid"],
                    ["add", "."], ["commit", "-qm", "base"]]:
        subprocess.run(["git", *command], cwd=tmp_path, check=True, capture_output=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path).decode().strip()


def probe_repository(tmp_path, base):
    import json
    import subprocess
    import sys
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "preparation"], cwd=tmp_path, check=True)
    return json.loads(subprocess.check_output([sys.executable, "-c", PROBE, base], cwd=tmp_path))


def test_sphinx_full_content_transform_and_url_side_effect(tmp_path):
    setup = b"install_requires = ['Jinja2>=2.3', 'packaging',]\n"
    tox = b"commands = pytest --durations 25\npy311: git+https://github.com/pytest-dev/py\n"
    base = repository(tmp_path, {"setup.py": setup, "tox.ini": tox})
    (tmp_path / "setup.py").write_bytes(setup.replace(b"Jinja2>=2.3", b"Jinja2<3.0").replace(b"'packaging',", b"'packaging', 'markupsafe<=2.0.1',"))
    (tmp_path / "tox.ini").write_bytes(tox.replace(b"pytest", b"pytest -rA"))
    report = probe_repository(tmp_path, base)
    assert report["sphinx_py311_url_rewrite"]
    assert validate_baseline(report, {"instance_id": "sphinx-doc__sphinx-9698", "base_commit": base}) == "known-sphinx-preparation"


def test_sphinx_file_names_do_not_allow_arbitrary_changes(tmp_path):
    base = repository(tmp_path, {"setup.py": b"install_requires = []\n", "tox.ini": b"pytest\n"})
    (tmp_path / "setup.py").write_bytes(b"install_requires = []\nprint('unrelated change')\n")
    (tmp_path / "tox.ini").write_bytes(b"pytest -rA\n")
    report = probe_repository(tmp_path, base)
    with pytest.raises(ValueError, match="unexpected Sphinx"):
        validate_baseline(report, {"instance_id": "sphinx-doc__sphinx-1", "base_commit": base})


def test_requests_build_copies_must_match_tracked_blobs(tmp_path):
    import json
    import subprocess
    import sys
    base = repository(tmp_path, {"requests/api.py": b"value = 1\n"})
    probe_repository(tmp_path, base)
    build = tmp_path / "build/lib/requests/api.py"
    build.parent.mkdir(parents=True)
    build.write_bytes(b"value = 1\n")
    instance = {"instance_id": "psf__requests-1142", "base_commit": base}
    report = json.loads(subprocess.check_output([sys.executable, "-c", PROBE, base], cwd=tmp_path))
    assert report["build_copies_verified"]
    assert validate_baseline(report, instance) == "content-identical-preparation"
    build.write_bytes(b"value = 2\n")
    report = json.loads(subprocess.check_output([sys.executable, "-c", PROBE, base], cwd=tmp_path))
    with pytest.raises(ValueError, match="dirty"):
        validate_baseline(report, instance)
