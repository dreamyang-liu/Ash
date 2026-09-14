import os

from harness.core.slot import TaskSpec
from harness.orchestrator.run import Orchestrator, RunOutcome
from harness.slots.codex_sdk import CodexSdkSlot
from runstore.child import execute


def test_worker_credentials_are_hidden_and_gateway_wins_over_agent_env(tmp_path, monkeypatch):
    environment = {**os.environ, "RS_BACKEND": "backend-fixture-secret",
                   "RS_PROVIDER": "provider-fixture-secret"}
    monkeypatch.setattr(os, "environ", environment)
    captured = {}

    def run(orchestrator, spec):
        task = TaskSpec(prompt=spec.prompt, cwd=spec.cwd, extra={})
        gateway = orchestrator._wire_gateway(spec, None, task, spec.run_id)
        try:
            captured.update(CodexSdkSlot()._child_env(task))
            assert spec.backend["microvm"]["api_key"] == "backend-fixture-secret"
            assert task.extra["config_overrides"]["model_provider"] == '"ash-gateway"'
            assert task.env["ANTHROPIC_BASE_URL"] == gateway.base_url
        finally:
            gateway.stop()
        return RunOutcome(spec.run_id, spec.journal_path, "completed")

    monkeypatch.setattr(Orchestrator, "run", run)
    execute({"kind": "rollout", "attempt_id": "fixture", "effective_spec": {
        "prompt": "fixture", "slot": "codex", "use_gateway": True,
        "backend": {"microvm": {"api_key": {"$env": "RS_BACKEND"}}}},
        "profile_config": {"worker_env": {"UPSTREAM_API_KEY": {"$env": "RS_PROVIDER"}},
                           "env": {"ANTHROPIC_BASE_URL": "http://must-not-bypass-gateway"}}}, tmp_path)
    assert captured["RS_BACKEND"] == ""
    assert captured["RS_PROVIDER"] == ""
    assert captured["UPSTREAM_API_KEY"] == ""
    assert captured["ASH_GATEWAY_TOKEN"].startswith("ash-slot-")
    assert "backend-fixture-secret" not in captured.values()
    assert "provider-fixture-secret" not in captured.values()
