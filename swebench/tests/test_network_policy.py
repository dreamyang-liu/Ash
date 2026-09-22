"""Independent actor/verifier egress, from CLI selection to backend wiring."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.execution import backends
from harness.orchestrator.run import RunOutcome
from swebench import fork_eval


def options(**changes):
    return SimpleNamespace(runtime_bin="runtime/ash-runtime", timeout=30,
                           slot="fixture", model="fixture", **changes)


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("actor", [None, "allow", "deny"])
@pytest.mark.parametrize("verifier", [None, "allow", "deny"])
def test_network_phases_are_independent_and_preserve_defaults(offline, actor, verifier):
    bench = fork_eval.Benchmark()
    bench.no_network = offline
    args = options(agent_network=actor, verifier_network=verifier)
    for phase, requested in (("agent", actor), ("verifier", verifier)):
        backend = fork_eval.backend_for(args, bench, phase)
        expected = requested == "allow" if requested is not None else False if offline else None
        assert backend["microvm"].get("allow_internet") is expected
        assert ("allow_internet" in backend["microvm"]) is (expected is not None)
    assert bench.no_network is offline


@pytest.mark.parametrize("allow", [True, False])
def test_phase_policy_reaches_microvm_pool_constructor(monkeypatch, allow):
    seen = {}
    monkeypatch.setattr(backends, "MicroVMPool", lambda **kwargs: seen.update(kwargs) or seen)
    config = fork_eval.backend_for(options(verifier_network="allow" if allow else "deny"),
                                  fork_eval.SweBench(), "verifier")
    config["microvm"].update(server_url="http://offline-fixture", api_key="fixture")
    backends.build_pool(config)
    assert seen["allow_internet"] is allow
    assert seen["server_url"] == "http://offline-fixture"


@pytest.mark.parametrize("resume", [False, True])
def test_actor_and_branch_specs_use_actor_policy_only(tmp_path, resume):
    seen = []
    orch = SimpleNamespace(run=lambda spec: seen.append(spec))
    args = options(agent_network="deny", verifier_network="allow")
    fork_eval.run_attempt(orch, args, {}, name="branch" if resume else "parent", prompt="Continue.",
                          image="snapshot", out_dir=tmp_path, resume="session" if resume else None,
                          fork=resume, bench=fork_eval.SweBench())
    assert seen[0].backend["microvm"]["allow_internet"] is False
    assert seen[0].resume_session_id == ("session" if resume else None)
    assert "internet access for this attempt: deny" in seen[0].prompt


def test_pro_legacy_shared_default_can_be_overridden_per_phase(tmp_path):
    from swebench_pro.bench import SWEbenchPro

    args = options(pro_repo=tmp_path, pro_block_network=True, agent_network="allow")
    bench = SWEbenchPro(args)
    assert fork_eval.backend_for(args, bench)["microvm"]["allow_internet"] is True
    assert fork_eval.backend_for(args, bench, "verifier")["microvm"]["allow_internet"] is False
    assert fork_eval.network_summary(args, bench)["no_network"] is None


@pytest.mark.parametrize("policy", ["allow", "deny"])
def test_network_aware_prompts_follow_actor_override(tmp_path, policy):
    from deepswe.bench import DeepSWE
    from swebench_pro.bench import SWEbenchPro

    instance = {"repo": "repo", "problem": "Fix it.", "agent_network": policy}
    deep = DeepSWE(tmp_path).prompt(instance)
    pro = SWEbenchPro(options(pro_repo=tmp_path, pro_block_network=True)).prompt(instance)
    swe = fork_eval.SweBench().prompt(instance)
    assert ("NO internet" in deep) is (policy == "deny")
    assert ("no internet access" in pro) is (policy == "deny")
    assert ("no internet access" in swe) is (policy == "deny")


@pytest.fixture
def fake_benchmark(monkeypatch):
    seen = {"prepare": [], "actor": [], "verifier": []}

    class Bench(fork_eval.Benchmark):
        name = "fixture"

        def catalogue(self, args):
            return {"task": "raw"}

        def instance(self, raw):
            return {"instance_id": "task", "repo": "fixture", "image": "image", "f2p": [], "p2p": []}

        def prompt(self, instance):
            return "Fix the task."

        def prepare_image(self, instance, backend, directory):
            seen["prepare"].append(backend)
            return "prepared"

        def grade(self, snapshot_id, instance, backend):
            seen["verifier"].append(backend)
            return fork_eval.Grade(resolved=True, f2p_pass=True, p2p_ran=True, p2p_pass=True)

    class Orchestrator:
        def __init__(self, **kwargs):
            pass

        def run(self, spec):
            seen["actor"].append(spec.backend)
            events = [
                {"type": "run.started", "slot": spec.slot},
                {"type": "checkpoint.captured", "step": 1, "snapshot_id": "saved", "reason": "captured"},
            ]
            Path(spec.journal_path).write_text("".join(json.dumps(event) + "\n" for event in events))
            return RunOutcome(run_id=spec.run_id, journal_path=spec.journal_path, status="completed")

    monkeypatch.setattr(fork_eval, "select_benchmark", lambda args: Bench())
    monkeypatch.setattr(fork_eval, "Orchestrator", Orchestrator)
    return seen


@pytest.mark.parametrize("actor,verifier", [("deny", "allow"), ("allow", "deny")])
def test_cli_network_flags_reach_execution_grading_and_summary(tmp_path, fake_benchmark, actor, verifier):
    arguments = ["--instance", "task", "--rounds", "0", "-o", str(tmp_path), "--volatile-ok",
                 "--agent-network", actor, "--verifier-network", verifier]
    assert fork_eval.main(arguments) == 0
    assert fake_benchmark["prepare"] == fake_benchmark["actor"]
    assert fake_benchmark["actor"][0]["microvm"]["allow_internet"] is (actor == "allow")
    assert fake_benchmark["verifier"][0]["microvm"]["allow_internet"] is (verifier == "allow")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["slot"] == "mini-swe-agent"
    assert summary["branch_guidance"] == "assistant-turn"
    assert summary["network_policy"] == {"agent": actor, "verifier": verifier}
    assert summary["network_requested"] == summary["network_policy"]
    assert summary["no_network"] is None
    assert summary["instances"][0]["attempts"][0]["network_policy"] == summary["network_policy"]
    assert summary["instances"][0]["attempts"][0]["grading_snapshot"]["network_policy"] == verifier


def test_regrade_changes_only_verifier_policy_and_preserves_actor_summary(tmp_path, fake_benchmark):
    arguments = ["--instance", "task", "--rounds", "0", "-o", str(tmp_path), "--volatile-ok",
                 "--agent-network", "deny", "--verifier-network", "deny"]
    assert fork_eval.main(arguments) == 0
    original = (tmp_path / "summary.json").read_bytes()
    assert fork_eval.main(["--regrade", "-o", str(tmp_path), "--verifier-network", "allow"]) == 0
    assert len(fake_benchmark["actor"]) == 1
    assert fake_benchmark["verifier"][-1]["microvm"]["allow_internet"] is True
    assert (tmp_path / "summary.json").read_bytes() == original
    regrade = json.loads((tmp_path / "regrade.json").read_text())
    assert regrade["network_policy"] == {"agent": "not-rerun", "verifier": "allow"}


def test_default_summary_does_not_claim_measured_network_access():
    summary = fork_eval.network_summary(options(), fork_eval.SweBench())
    assert summary["network_policy"] == {"agent": "backend-default", "verifier": "backend-default"}
    assert summary["network_requested"] == {"agent": None, "verifier": None}
    assert summary["no_network"] is False


def test_imported_parent_is_not_relabelled_with_requested_actor_policy(tmp_path, fake_benchmark):
    source = tmp_path / "source"
    output = tmp_path / "branching"
    assert fork_eval.main(["--instance", "task", "--rounds", "0", "-o", str(source), "--volatile-ok",
                           "--agent-network", "allow", "--verifier-network", "allow"]) == 0
    assert fork_eval.main(["--instance", "task", "--rounds", "0", "-o", str(output), "--volatile-ok",
                           "--parent-from", str(source), "--agent-network", "deny",
                           "--verifier-network", "deny"]) == 0
    assert len(fake_benchmark["actor"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["network_requested"]["agent"] == "deny"
    assert summary["instances"][0]["attempts"][0]["network_policy"] == {
        "agent": "recorded", "verifier": "deny"}


def test_invalid_network_values_are_rejected():
    with pytest.raises(SystemExit):
        fork_eval.main(["--agent-network", "maybe"])
    with pytest.raises(ValueError, match="Invalid verifier"):
        fork_eval.backend_for(options(verifier_network=True), fork_eval.SweBench(), "verifier")
    with pytest.raises(ValueError, match="Network phase"):
        fork_eval.backend_for(options(), phase="inference")
