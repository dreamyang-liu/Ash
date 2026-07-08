"""Unit tests for the best-of-n harness (swebench.harnesses.best_of_n).

Pure functions only — no Docker, no model calls:
- FAIL_TO_PASS parsing (JSON list string, list passthrough, garbage)
- changed-file extraction from unified diffs
- per-candidate temperature laddering (jitter + clamp)
- test-command mapping (pytest default, django runner + id conversion)
- selection: tests mode (score, then fewer-files / shorter-patch tiebreaks)
- selection: heuristic mode (majority file agreement, then shortest)
- selection-report shape and overall exit status
- wiring: registry entry, config flattening, harness instantiation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swebench.harnesses import HARNESSES, get_harness
from swebench.harnesses.best_of_n import (
    BestOfNHarness,
    Candidate,
    MAX_TEMPERATURE,
    build_test_command,
    candidate_temperature,
    changed_files,
    overall_status,
    parse_test_list,
    select_by_heuristic,
    select_by_tests,
    selection_report,
)
from swebench.__main__ import _flatten_config, _load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "best-of-n.yaml"


def _diff(*files: str, pad: int = 0) -> str:
    """Minimal unified diff touching ``files``; ``pad`` controls patch length."""
    hunks = [
        f"diff --git a/{f} b/{f}\n--- a/{f}\n+++ b/{f}\n@@ -1 +1 @@\n-old\n+new\n"
        for f in files
    ]
    return "".join(hunks) + ("#" * pad + "\n" if pad else "")


def _cand(index: int, patch: str = "", status: str = "completed",
          passed: int | None = None, total: int | None = None,
          cost: float = 0.0) -> Candidate:
    return Candidate(index=index, patch=patch, exit_status=status,
                     cost=cost, tests_passed=passed, tests_total=total)


# --------------------------------------------------------------------------- #
#  FAIL_TO_PASS parsing
# --------------------------------------------------------------------------- #

def test_parse_test_list_decodes_json_list_string():
    raw = '["tests/test_a.py::test_x", "tests/test_a.py::test_y"]'
    assert parse_test_list(raw) == ["tests/test_a.py::test_x",
                                    "tests/test_a.py::test_y"]


def test_parse_test_list_passes_lists_through():
    assert parse_test_list(["a", "b"]) == ["a", "b"]


def test_parse_test_list_rejects_garbage():
    assert parse_test_list("not json at all") == []
    assert parse_test_list('{"a": 1}') == []      # JSON but not a list
    assert parse_test_list("") == []
    assert parse_test_list(None) == []


# --------------------------------------------------------------------------- #
#  Patch introspection
# --------------------------------------------------------------------------- #

def test_changed_files_reads_diff_git_headers():
    patch = _diff("src/a.py", "src/pkg/b.py")
    assert changed_files(patch) == frozenset({"src/a.py", "src/pkg/b.py"})
    assert changed_files("") == frozenset()


# --------------------------------------------------------------------------- #
#  Temperature ladder
# --------------------------------------------------------------------------- #

def test_candidate_temperature_without_jitter_passes_base_through():
    assert candidate_temperature(0.7, None, 2) == 0.7
    assert candidate_temperature(None, 0, 1) is None


def test_candidate_temperature_ladders_and_clamps():
    assert candidate_temperature(0.3, 0.2, 0) == 0.3
    assert candidate_temperature(0.3, 0.2, 2) == 0.7
    assert candidate_temperature(None, 0.2, 1) == 0.2   # no base -> start at 0.0
    assert candidate_temperature(0.9, 0.2, 3) == MAX_TEMPERATURE


# --------------------------------------------------------------------------- #
#  Test-command mapping
# --------------------------------------------------------------------------- #

def test_build_test_command_defaults_to_pytest():
    cmd = build_test_command("astropy/astropy", "astropy/io/tests/test_x.py::test_y")
    assert cmd.startswith("python -m pytest")
    assert "astropy/io/tests/test_x.py::test_y" in cmd


def test_build_test_command_converts_django_ids():
    cmd = build_test_command(
        "django/django",
        "test_ordering (migrations.test_operations.OperationTests)")
    assert cmd.startswith("./tests/runtests.py")
    assert "migrations.test_operations.OperationTests.test_ordering" in cmd


# --------------------------------------------------------------------------- #
#  Selection: tests mode (research mode)
# --------------------------------------------------------------------------- #

def test_select_by_tests_most_passing_wins():
    candidates = [
        _cand(0, _diff("a.py"), passed=0, total=2),
        _cand(1, _diff("b.py"), passed=2, total=2),
        _cand(2, _diff("c.py"), passed=1, total=2),
    ]
    assert select_by_tests(candidates) == 1


def test_select_by_tests_tiebreak_fewer_files_then_shorter_patch():
    two_files = _cand(0, _diff("a.py", "b.py"), passed=1, total=1)
    one_file_long = _cand(1, _diff("a.py", pad=100), passed=1, total=1)
    one_file_short = _cand(2, _diff("a.py"), passed=1, total=1)
    assert select_by_tests([two_files, one_file_long]) == 1     # fewer files
    assert select_by_tests([one_file_long, one_file_short]) == 2  # shorter patch


def test_select_by_tests_ignores_empty_patches():
    candidates = [
        _cand(0, "", passed=1, total=1),          # empty patch never wins
        _cand(1, _diff("a.py"), passed=0, total=1),
    ]
    assert select_by_tests(candidates) == 1
    assert select_by_tests([_cand(0, ""), _cand(1, "  \n")]) is None
    assert select_by_tests([]) is None


# --------------------------------------------------------------------------- #
#  Selection: heuristic mode
# --------------------------------------------------------------------------- #

def test_select_by_heuristic_majority_file_agreement_wins():
    candidates = [
        _cand(0, _diff("core/fix.py", pad=50)),
        _cand(1, _diff("core/fix.py")),           # same file set, shortest
        _cand(2, _diff("other.py")),              # unique set, shortest overall
    ]
    assert select_by_heuristic(candidates) == 1   # majority set, then shortest


def test_select_by_heuristic_falls_back_to_shortest():
    candidates = [
        _cand(0, _diff("a.py", pad=200)),
        _cand(1, _diff("b.py")),                  # all sets unique -> shortest
        _cand(2, _diff("c.py", pad=100)),
    ]
    assert select_by_heuristic(candidates) == 1


def test_select_by_heuristic_prefers_any_patch_over_none():
    assert select_by_heuristic([_cand(0, ""), _cand(1, _diff("a.py"))]) == 1
    assert select_by_heuristic([_cand(0, ""), _cand(1, "")]) is None


# --------------------------------------------------------------------------- #
#  Report shape + overall status
# --------------------------------------------------------------------------- #

def test_selection_report_shape():
    candidates = [
        _cand(0, _diff("a.py"), passed=1, total=2, cost=0.1234),
        _cand(1, "", status="error: boom"),
    ]
    report = selection_report("astropy__astropy-1", "tests", candidates, winner=0)

    assert report["instance_id"] == "astropy__astropy-1"
    assert report["selection"] == "tests"
    assert report["winner"] == 0
    assert len(report["candidates"]) == 2
    first = report["candidates"][0]
    assert first == {
        "candidate": 0,
        "exit_status": "completed",
        "patch_bytes": len(_diff("a.py")),
        "changed_files": ["a.py"],
        "tests_passed": 1,
        "tests_total": 2,
        "cost": 0.1234,
    }
    assert report["candidates"][1]["changed_files"] == []


def test_overall_status_modes():
    ok = _cand(0, _diff("a.py"))
    empty = _cand(1, "")
    failed = _cand(2, "", status="session_failed")
    errored = _cand(3, "", status="error: boom")

    assert overall_status([ok, empty], winner=0) == "completed"
    assert overall_status([empty, failed], winner=None) == "no_patch"
    assert overall_status([failed, errored], winner=None) == "session_failed"
    assert overall_status([], winner=None) == "no_patch"


# --------------------------------------------------------------------------- #
#  Wiring: registry, config, instantiation
# --------------------------------------------------------------------------- #

def test_harness_is_registered():
    assert HARNESSES["best-of-n"] is BestOfNHarness
    assert get_harness("best-of-n") is BestOfNHarness


def test_harness_instantiates_and_validates_selection():
    harness = BestOfNHarness({})                  # defaults: selection=tests
    assert harness.name == "BestOfNHarness"
    with pytest.raises(ValueError, match="unknown selection"):
        BestOfNHarness({"selection": "llm-judge"})


def test_config_flattens_best_of_n_section():
    flat = _flatten_config(_load_config(str(CONFIG_PATH)))
    assert flat["harness"] == "best-of-n"
    assert flat["n_candidates"] == 3
    assert flat["selection"] == "tests"
    assert flat["temperature_jitter"] == 0.2
    assert flat["output"] == "results/best-of-n"
    assert flat["runtime_bin"]
    assert flat["cost_limit"] == 3.0              # per-candidate budget
    BestOfNHarness(flat)                          # config instantiates cleanly
