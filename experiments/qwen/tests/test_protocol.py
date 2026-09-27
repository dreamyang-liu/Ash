import json
from pathlib import Path
from types import SimpleNamespace

from experiments.qwen import run
from experiments.qwen.report import phase_metrics, usage
from taskwise.policies import bpo_select, shepherd_select, token_entropy


def args(tmp_path, method="all"):
    return SimpleNamespace(output=tmp_path, baseline_root=None, method=method,
                           dataset="deepswe", max_rollouts=8)


def test_success_still_runs_four_and_skips_all_branches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "stage", lambda a, t, o, m, *rest, **kw:
                        calls.append(m) or [{"resolved": True}])
    result = run.run_task(args(tmp_path), "task")
    assert result["complete"] and calls == ["baseline"] * 4
    assert all(s["status"] == "skipped_initial_success" for s in result["methods"].values())


def test_initial_failure_stays_shared_even_if_later_baseline_passes(tmp_path, monkeypatch):
    calls = []
    def stage(a, t, output, method, *rest, **kw):
        calls.append(method)
        if method == "baseline":
            return [{"resolved": output.name != "attempt-1"}]
        return ([{"resolved": False}] if method == "sprout" else []) + [{"resolved": True}]
    def retain(out):
        run.write(out / "retained-checkpoints.json", {"keep": True})
    monkeypatch.setattr(run, "stage", stage)
    monkeypatch.setattr(run, "retain_parent", retain)
    monkeypatch.setattr(run, "snapshot_ids", lambda *a: {"shared-first"})
    result = run.run_task(args(tmp_path), "task")
    assert result["complete"] and calls == ["baseline"] * 4 + ["sprout", "bpo", "shepherd"]
    assert (tmp_path / "tasks/task/baseline/attempt-1/retained-checkpoints.json").exists()


def test_failure_never_deletes_shared_parent(tmp_path, monkeypatch):
    def stage(a, t, out, method, *rest, **kw):
        if method != "baseline":
            raise RuntimeError("network error")
        return [{"resolved": False}]
    monkeypatch.setattr(run, "stage", stage)
    monkeypatch.setattr(run, "retain_parent", lambda out: run.write(out / "retained-checkpoints.json", {}))
    monkeypatch.setattr(run, "snapshot_ids", lambda *a: {"parent"})
    result = run.run_task(args(tmp_path), "task")
    assert not result["complete"]
    assert (tmp_path / "tasks/task/baseline/attempt-1/retained-checkpoints.json").exists()
    assert not hasattr(run, "cleanup_owned")


def test_unknown_cost_is_not_zero_and_cached_subset_preserved():
    u = usage({"prompt_tokens": 100, "completion_tokens": 20,
               "prompt_tokens_details": {"cached_tokens": 75}})
    assert u["cost_usd"] is None and u["cached_input_tokens"] == 75


def test_deepswe_negative_exit_is_graded_but_infrastructure_error_is_not(tmp_path, monkeypatch):
    import pytest
    a = SimpleNamespace(dataset="deepswe", source_commit="pinned", server_url="http://localhost",
                        api_key_file=None, max_rollouts=8)
    monkeypatch.setattr(run, "command", lambda *args: ["fixture"])
    output = tmp_path / "attempt"
    def completed_negative(*args, **kwargs):
        run.write(output / "summary.json", {"instances": [{"attempts": [{"status": "timeout", "resolved": False}]}]})
        return 1
    monkeypatch.setattr(run.subprocess, "call", completed_negative)
    assert run.stage(a, "task", output, "baseline")[0]["resolved"] is False
    output = tmp_path / "broken"
    def broken(*args, **kwargs):
        run.write(output / "summary.json", {"instances": [{"attempts": [{"status": "error", "resolved": False}]}]})
        return 1
    monkeypatch.setattr(run.subprocess, "call", broken)
    with pytest.raises(ValueError, match="infrastructure"):
        run.stage(a, "task", output, "baseline")


def test_metrics_include_failed_calls_critic_and_no_inherited_steps(tmp_path):
    initial = tmp_path / "initial"
    extra = tmp_path / "extra"
    initial.mkdir(); extra.mkdir()
    rows = [{"type": "run.started"}, {"type": "tool.started"},
            {"type": "raw.mini-swe-agent", "response": {"usage": {
                "prompt_tokens": 100, "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0}, "cost": 0.1}}}]
    content = "\n".join(json.dumps(r) for r in rows)
    (initial / "parent.jsonl").write_text(content)
    (extra / "parent.jsonl").write_text(content)
    (extra / "child.jsonl").write_text(content + '\n{"type":"model.request.retry"}')
    run.write(extra / "meta-request.json", {"request": {}, "response": {"usage": {
        "prompt_tokens": 10, "completion_tokens": 2, "cost": 0.02}}})
    first = phase_metrics([initial])
    additional = phase_metrics([extra], set(first["journal_hashes"]))
    assert additional["actual_new_steps"] == 1
    assert additional["usage"]["known_input_tokens"] == 110
    assert additional["usage"]["cost_usd"] is None
    assert additional["model_overhead_calls"] == 2


def test_exact_selectors_reject_invalid_and_do_not_invent_entropy():
    import pytest
    with pytest.raises(ValueError):
        token_entropy({})
    with pytest.raises(ValueError):
        shepherd_select({"checkpoint_step": 6, "reason": "test"}, [1, 2], 7)
    selected = bpo_select([{"step": 1, "token_position": 10, "entropy_lower_bound_nats": 1}], 7)
    assert len(selected) == 7 and {s["step"] for s in selected} == {1}


def test_failed_selector_response_usage_survives_retry(tmp_path, monkeypatch):
    from taskwise import policy_eval
    responses = iter([
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        {"choices": [{"message": {"content": "ok"}}],
         "usage": {"prompt_tokens": 11, "completion_tokens": 5}},
    ])
    monkeypatch.setattr(policy_eval.httpx, "post", lambda *args, **kwargs:
                        SimpleNamespace(status_code=200, json=lambda: next(responses)))
    monkeypatch.setattr(policy_eval.time, "sleep", lambda _: None)
    policy_eval.ModelClient("http://localhost:9000", "test-only").complete(
        {"model": "fixture"}, tmp_path / "entropy/step-1.request.json")
    metrics = phase_metrics([tmp_path])
    assert metrics["model_overhead_calls"] == 2
    assert metrics["usage"]["input_tokens"] == 18
    assert metrics["usage"]["output_tokens"] == 8


def test_commands_have_bounded_shared_parent_and_high(tmp_path):
    a = SimpleNamespace(model="qwen3.8-27b", model_endpoint="http://localhost:9000",
        model_key_env="QWEN_API_KEY", runtime_bin=tmp_path / "runtime", max_turns=300,
        max_output_tokens=64000, dataset="terminalbench21", max_rollouts=8,
        preparation=tmp_path / "prep.json", tasks_dir=tmp_path, digest="sha256:test",
        server_url="http://localhost:8000", api_key_file=None, image_registry=None)
    for method in run.METHODS:
        cmd = run.command(a, "task", tmp_path / method, method, tmp_path / "parent")
        assert cmd[cmd.index("--reasoning-effort") + 1] == "high"
        if method in ("bpo", "shepherd"):
            assert cmd[cmd.index("--max-rollouts") + 1] == "8"
        if method == "sprout":
            assert cmd[cmd.index("--branches") + 1:cmd.index("--branches") + 3] == ["4", "3"]


def test_five_row_metrics_resolve_recovery_and_actual_tokens(tmp_path):
    from experiments.qwen.report import build_report
    run.write(tmp_path / "manifest-all.json", {"method": "all", "tasks": ["a", "b"],
        "max_total_method_rollouts": 8})
    for name in ("a", "b"):
        root = tmp_path / "tasks" / name
        for i in range(1, 5):
            output = root / "baseline" / f"attempt-{i}"
            run.write(output / "summary.json", {"attempts": [{"resolved": name == "a" or i == 3}]})
            events = [{"type": "run.started", "identity": f"{name}-{i}"}, {"type": "tool.started"},
                      {"type": "raw.mini-swe-agent", "response": {"usage": {
                          "prompt_tokens": 100, "completion_tokens": 20,
                          "prompt_tokens_details": {"cached_tokens": 50}}}}]
            (output / "journal.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        methods = {}
        for method in ("bpo", "shepherd", "sprout"):
            methods[method] = {"status": "skipped_initial_success" if name == "a" else "complete"}
            if name == "b":
                rows = ([{"resolved": False}] if method == "sprout" else []) + [{"resolved": method != "bpo"}]
                run.write(root / method / "summary.json", {"attempts": rows})
        run.write(root / "all.task.json", {"methods": methods})
    rows = build_report(tmp_path, "all")["comparison"]
    assert len(rows) == 5
    assert rows[0]["Resolve Rate"] == 0.5 and rows[0]["Recovery"] == 0
    assert rows[0]["Steps"] == 2 and rows[0]["Tokens (M)"] == 240 / 1e6
    assert rows[1]["Resolve Rate"] == 1 and rows[1]["Recovery"] == 1
    assert rows[1]["Steps"] == 8 and rows[1]["Tokens (M)"] == 960 / 1e6
    assert rows[2]["Resolve Rate"] == 0.5 and rows[2]["Recovery"] == 0
    assert rows[3]["Resolve Rate"] == rows[4]["Resolve Rate"] == 1
