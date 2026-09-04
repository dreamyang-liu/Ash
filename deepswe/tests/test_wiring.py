"""The seam: what fork_eval hands the orchestrator when the benchmark is DeepSWE.

These assert *wiring*, in this repository's tradition: a no-network flag that
exists but never reaches the pool would grade an offline benchmark online.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deepswe.bench import DeepSWE, PRIMER
from deepswe.tests.test_tasks import make_task
from swebench import fork_eval


def args(**kw):
    base = dict(runtime_bin="runtime/ash-runtime", benchmark="deepswe",
                tasks_dir=None, subset="verified")
    base.update(kw)
    return SimpleNamespace(**base)


def test_backend_turns_egress_off_for_deepswe_and_not_for_swebench(tmp_path):
    deep = fork_eval.backend_for(args(), DeepSWE(tmp_path))
    assert deep["microvm"]["allow_internet"] is False
    swe = fork_eval.backend_for(args(benchmark="swebench"), fork_eval.SweBench())
    assert "allow_internet" not in swe["microvm"]
    # No benchmark given: the historical SWE-bench payload, byte for byte.
    assert fork_eval.backend_for(args()) == swe


def test_gate_uses_the_same_backend_as_a_real_attempt(tmp_path):
    # The gate is evidence about the runs only if it runs on the runs' backend:
    # same egress policy, same image_env templates. A private copy drifted once.
    from deepswe.gate import backend_for as gate_backend
    make_task(tmp_path)
    assert gate_backend("runtime/ash-runtime", str(tmp_path)) == \
        fork_eval.backend_for(args(), DeepSWE(tmp_path))
    assert gate_backend("runtime/ash-runtime", str(tmp_path))["microvm"]["image_env"] is True


def test_select_benchmark_defaults_to_swebench_and_needs_tasks_dir_for_deepswe():
    assert isinstance(fork_eval.select_benchmark(args(benchmark="swebench")),
                      fork_eval.SweBench)
    with pytest.raises(SystemExit, match="tasks-dir"):
        fork_eval.select_benchmark(args(benchmark="deepswe", tasks_dir=None))


def test_instance_prompt_and_resources_come_from_the_task(tmp_path):
    make_task(tmp_path)
    bench = DeepSWE(tmp_path)
    catalogue = bench.catalogue(args(tasks_dir=str(tmp_path)))
    instance = bench.instance(catalogue["demo-task"])
    assert instance["instance_id"] == "demo-task"
    assert instance["image"].startswith("public.ecr.aws/")
    assert instance["f2p"] == ["pkg.TestA", "pkg.TestB"]
    assert bench.resources(instance) == {"cpu": 2, "memory_mb": 8192}

    prompt = bench.prompt(instance)
    assert "Add `Query` to orderedmap." in prompt          # instruction verbatim
    assert "/app" in prompt and "/testbed" not in prompt   # re-pointed primer
    assert "NO internet" in prompt and "COMMITTED" in prompt
    assert PRIMER in prompt

    branch = bench.branch_prompt(instance, verdict="resolved=False", hint="try X")
    assert "resolved=False" in branch and "try X" in branch and "/testbed" not in branch


def test_branch_note_for_a_truncated_fork_reads_as_environment_feedback(tmp_path):
    from swebench.fork_eval import Grade
    make_task(tmp_path)
    bench = DeepSWE(tmp_path)
    instance = bench.instance(bench.catalogue(None)["demo-task"])
    grade = Grade(patch="diff", detail=(
        "repo state: branch=feat, 1 commit(s) since base, 0 uncommitted path(s)\n"
        'reward.json: {"f2p_passed": 5, "f2p_total": 6, "p2p_passed": 6, "p2p_total": 6, "reward": 0}\n'
        "non-passing tests (1 shown): f2p:pkg.TestStep\n\nverifier output (tail):\nblah"))
    analysis = {"failure_reason": "the step parser dropped the sign", "lesson": "negative steps exist",
                "salvage": "everything on disk at step 60"}
    note = bench.branch_prompt(instance, verdict="IGNORED RAW VERDICT", hint="handle negative steps",
                               truncated=True, step=25, grade=grade, analysis=analysis)
    assert note.startswith("<system-reminder>") and note.rstrip().endswith("</system-reminder>")
    assert "step 25" in note
    assert "target tests: 5/6" in note and "pkg.TestStep" in note
    assert "regression tests: 6/6" in note
    assert "committed and applied cleanly" in note
    assert "the step parser dropped the sign" in note and "negative steps exist" in note
    assert "handle negative steps" in note
    # not restated: the task, the tool primer, the raw verdict, the salvage note
    assert "Add `Query`" not in note and "Your tools" not in note
    assert "IGNORED RAW VERDICT" not in note and "everything on disk" not in note
    # an untruncated fork still gets the full prompt that explains the rolled-back disk
    full = bench.branch_prompt(instance, verdict="V", hint="H")
    assert "ITS EDITS ARE ON DISK" in full and "Add `Query`" in full


def test_run_attempt_passes_shape_and_offline_backend_to_the_orchestrator(tmp_path):
    make_task(tmp_path)
    bench = DeepSWE(tmp_path)
    instance = bench.instance(bench.catalogue(None)["demo-task"])
    seen = {}

    class Orch:
        def run(self, spec):
            seen["spec"] = spec
            return SimpleNamespace(status="completed", journal_path=None,
                                   checkpoints=0, error=None)

    fork_eval.run_attempt(Orch(), args(slot="claude-code", model="m", timeout=10800.0),
                          instance, name="parent", prompt="p", image=instance["image"],
                          out_dir=tmp_path / "out", resources=bench.resources(instance),
                          bench=bench)
    spec = seen["spec"]
    assert spec.sandbox_resources == {"cpu": 2, "memory_mb": 8192}
    assert spec.backend["microvm"]["allow_internet"] is False
    assert spec.timeout_s == 10800.0
    assert spec.sandbox_image == instance["image"]
