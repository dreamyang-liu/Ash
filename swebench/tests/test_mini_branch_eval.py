from __future__ import annotations

import json

import pytest

from swebench import fork_eval, mini_branch_eval, structured_review


def options(tmp_path, benchmark):
    runtime = tmp_path / "runtime"
    runtime.touch()
    return ["--benchmark", benchmark, "--instance", "fixture",
            "--out", str(tmp_path / "result"), "--source-commit", "pinned",
            "--model", "fixture-model", "--model-endpoint", "http://127.0.0.1:18252",
            "--model-key-env", "TEST_MODEL_KEY", "--runtime-bin", str(runtime),
            *(["--tasks-dir", str(tmp_path / "tasks")] if benchmark == "deepswe" else
              ["--pro-repo", str(tmp_path / "upstream"), "--pro-runtime-port", "34122"])]


@pytest.mark.parametrize("benchmark", ["deepswe", "swebench-pro"])
def test_shared_wrapper_freezes_the_same_mini_branch_policy(tmp_path, benchmark):
    args = mini_branch_eval.parse_args(options(tmp_path, benchmark))
    forwarded = mini_branch_eval._fork_args(args)
    for key, value in [("--slot", "mini-swe-agent"),
                       ("--branch-guidance", "assistant-turn"),
                       ("--branches", "4,3"), ("--rounds", "2"),
                       ("--branch-count-mode", "adaptive")]:
        assert forwarded[forwarded.index(key) + 1] == value
    if benchmark == "swebench-pro":
        assert forwarded[forwarded.index("--pro-runtime-port") + 1] == "34122"
        assert forwarded[forwarded.index("--timeout") + 1] == "3600"
    else:
        assert forwarded[forwarded.index("--tasks-dir") + 1] == str((tmp_path / "tasks").resolve())
        assert forwarded[forwarded.index("--timeout") + 1] == "10800"


def test_shared_wrapper_rejects_wrong_source_before_creating_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(mini_branch_eval, "_source_identity", lambda: ("different", {}))
    monkeypatch.setenv("TEST_MODEL_KEY", "local-test")
    with pytest.raises(ValueError, match="differs"):
        mini_branch_eval.main(options(tmp_path, "deepswe"))
    assert not (tmp_path / "result").exists()


def test_shared_wrapper_wires_model_route_and_strict_reviewer(tmp_path, monkeypatch):
    monkeypatch.setattr(mini_branch_eval, "_source_identity", lambda: ("pinned", {"source": "digest"}))
    monkeypatch.setenv("TEST_MODEL_KEY", "local-test")
    original = fork_eval.Orchestrator
    observed = []

    def fake_eval(argv):
        observed.append({"argv": argv, "orchestrator": fork_eval.Orchestrator,
                         "analyst": fork_eval.ask_analyst,
                         "parser": fork_eval.extract_branch_plan})
        return 0

    monkeypatch.setattr(fork_eval, "main", fake_eval)
    assert mini_branch_eval.main(options(tmp_path, "swebench-pro")) == 0
    assert observed[0]["orchestrator"] is not original
    assert observed[0]["parser"] is structured_review.extract_branch_plan
    assert observed[0]["analyst"].model == "fixture-model"
    assert fork_eval.Orchestrator is original
    manifest = json.loads((tmp_path / "result/shared-source.json").read_text())
    assert manifest["branch_caps"] == [4, 3] and manifest["source_commit"] == "pinned"
    assert "local-test" not in (tmp_path / "result/shared-source.json").read_text()
