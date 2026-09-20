"""BPO probability provenance, entropy selection and branch budgets."""

import json
import math
from types import SimpleNamespace

import pytest

from deepswe.branching.policies import bpo_select, token_entropy
from deepswe.branching.provider import candidates_from_audit, score_candidate, ProbabilityUnavailable
from deepswe.branching.storage import save


def token(*probabilities):
    return {"top_logprobs": [{"token": str(i), "logprob": math.log(p)}
                            for i, p in enumerate(probabilities)]}


def test_entropy_preserves_tail_and_uses_natural_log():
    result = token_entropy(token(.5, .25))
    assert result.tail_mass == .25
    assert result.lower_bound_nats == pytest.approx(1.0397207708399179)
    assert token_entropy(token(.5, .25, .125, .125)).lower_bound_nats > result.lower_bound_nats


@pytest.mark.parametrize("entry", [{}, {"top_logprobs": []},
    {"top_logprobs": [{"token": "x", "logprob": float("nan")}]},
    {"top_logprobs": [{"token": "x", "logprob": .1}]},
    {"top_logprobs": [{"token": "x", "logprob": float("-inf")}]}, token(.8, .8),
    {"top_logprobs": [{"token": "x", "logprob": -1}, {"token": "x", "logprob": -1}]}])
def test_invalid_probability_never_becomes_fallback(entry):
    with pytest.raises(ValueError):
        token_entropy(entry)


def test_bpo_spacing_and_sibling_budget():
    scores = [{"step": s, "token_position": pos, "entropy_lower_bound_nats": h}
              for s, pos, h in [(1, 0, .4), (2, 20, .9), (3, 84, .8), (4, 100, .7)]]
    result = bpo_select(scores, 7)
    assert [r["step"] for r in result] == [2, 3, 2, 3, 2, 3, 2]
    assert [r["sibling_index"] for r in result] == [2, 2, 3, 3, 4, 4, 5]
    assert scores[0]["step"] == 1  # no mutation of caller's backbone


def test_empty_bpo_and_zero_budget():
    assert bpo_select([], 0) == []
    with pytest.raises(ValueError):
        bpo_select([], 7)


def test_provider_audit_maps_after_tool_boundary(tmp_path):
    point = SimpleNamespace(call_id="tool1", snapshot_id="snap1")
    usage = []
    for i in range(2):
        request = {"model": "m", "messages": [{"role": "user", "content": "task"}]}
        if i:
            request["messages"].append({"role": "tool", "tool_call_id": "tool1", "content": "ok"})
        save(tmp_path / "provider-responses" / (str(i) + ".json"), {
            "owner": "owner", "request": request,
            "native_response": {"usage": {"completion_tokens": 80}, "choices": [{}]}})
        usage.append(json.dumps({"owner": "owner", "status": 200, "time": i, "request_id": str(i)}))
    (tmp_path / "actor-usage.jsonl").write_text("\n".join(usage))
    candidates = candidates_from_audit(tmp_path, "owner", {4: point}, "m")
    assert len(candidates) == 1
    assert candidates[0]["step"] == 4
    assert candidates[0]["token_position"] == 80
    with pytest.raises(ValueError, match="owner/model"):
        candidates_from_audit(tmp_path, "owner", {4: point}, "other-model")


def test_rescore_preserves_original_prefix_and_marks_approximation(tmp_path):
    original = {"model": "m", "messages": [{"role": "assistant", "reasoning_content": "retain"}], "max_tokens": 100}
    candidate = {"step": 2, "token_position": 64, "snapshot_id": "s", "request_id": "id",
                 "request": original, "native_response": {"choices": [{}]}}
    sent = []
    def complete(request, audit):
        sent.append(request)
        return {"choices": [{"logprobs": {"content": [token(.5, .25)]}}], "usage": {"completion_tokens": 1}}
    client = SimpleNamespace(complete=complete)
    result = score_candidate(candidate, client, tmp_path)
    assert sent[0]["messages"] == original["messages"]
    assert "max_tokens" not in sent[0]
    assert original["max_tokens"] == 100
    assert result["source"] == "historical-prefix-rescore"
    assert result["tail_mass"] == .25
    score_candidate(candidate, client, tmp_path)
    assert len(sent) == 1
    with pytest.raises(ValueError, match="cache"):
        score_candidate(candidate, client, tmp_path, top_k=10)


def test_provider_omits_logprobs_fails_closed(tmp_path):
    candidate = {"step": 2, "token_position": 64, "snapshot_id": "s", "request_id": "id",
                 "request": {"model": "m"}, "native_response": {"choices": [{}]}}
    client = SimpleNamespace(complete=lambda *args: {"choices": [{}]})
    with pytest.raises(ProbabilityUnavailable):
        score_candidate(candidate, client, tmp_path)
    assert not (tmp_path / "step-2.json").exists()


def test_original_probabilities_do_not_spend_a_rescoring_request(tmp_path):
    candidate = {"step": 2, "token_position": 64, "snapshot_id": "s", "request_id": "id",
                 "request": {"model": "m"}, "native_response": {
                     "choices": [{"logprobs": {"content": [token(.5, .25)]}}]}}
    client = SimpleNamespace(complete=lambda *args: pytest.fail("Original probabilities exist"))
    result = score_candidate(candidate, client, tmp_path)
    assert result["source"] == "original-response"
    assert result["scoring_usage"] is None
    assert result["entropy_lower_bound_nats"] == pytest.approx(1.0397207708399179)
    assert not (tmp_path / "step-2.request.json").exists()


@pytest.mark.parametrize("status", [400, 422, 403, 502])
def test_http_rejection_is_audited_without_credentials(tmp_path, monkeypatch, status):
    import httpx
    from deepswe.branching import provider

    original_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"error": "rejected"}))
    monkeypatch.setattr(provider.httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs))
    client = provider.ChatClient(base_url="https://example.invalid/v1", api_key="private-test-key")
    audit = tmp_path / "request.json"
    error = ProbabilityUnavailable if status in (400, 422) else RuntimeError
    with pytest.raises(error):
        client.complete({"model": "m", "logprobs": True}, audit)
    record = json.loads(audit.read_text())
    assert record["http_status"] == status
    assert record["error_type"] == error.__name__
    assert "private-test-key" not in audit.read_text()
    assert "Authorization" not in audit.read_text()


def bpo_runner(tmp_path, monkeypatch):
    from deepswe.branching.runner import BenchmarkRunner, Config
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    runner = BenchmarkRunner(Config("m", str(tmp_path), str(tmp_path / "output"),
                                   "/runtime", methods=("bpo",), max_rollouts=3))
    journal = tmp_path / "parent.jsonl"
    journal.write_text('{"type":"run.finished"}\n')
    monkeypatch.setattr(runner, "initial", lambda task: ({"grade": {"resolved": False}}, journal))
    return runner, journal


def test_bpo_runner_wires_ranked_checkpoints_to_same_parent(tmp_path, monkeypatch):
    from deepswe.branching import runner as module
    runner, journal = bpo_runner(tmp_path, monkeypatch)
    points = {step: SimpleNamespace(step=step, snapshot_id="snapshot-%d" % step)
              for step in (2, 4)}
    runner.ev = SimpleNamespace(available_branch_points=lambda path: points)
    scores = [{"step": 2, "token_position": 64, "entropy_lower_bound_nats": .2},
              {"step": 4, "token_position": 128, "entropy_lower_bound_nats": .8}]
    monkeypatch.setattr(module, "candidates_from_audit", lambda *args: scores)
    monkeypatch.setattr(module, "score_candidate", lambda candidate, *args, **kwargs: candidate)
    calls = []
    def branch(task, method, index, choice, checkpoint, parent):
        calls.append((method, choice["step"], checkpoint.snapshot_id, parent))
        return {"name": "branch-%d" % index, "grade": {"resolved": index == 2}}
    monkeypatch.setattr(runner, "branch", branch)
    summary = runner.run_task(SimpleNamespace(task_id="task"))
    assert calls == [("bpo", 4, "snapshot-4", journal), ("bpo", 2, "snapshot-2", journal)]
    assert summary["methods"]["bpo"]["resolved"]
    assert summary["methods"]["bpo"]["extra_rollouts"] == 2


def test_bpo_missing_probabilities_blocks_before_any_branch(tmp_path, monkeypatch):
    runner, _ = bpo_runner(tmp_path, monkeypatch)
    def unavailable(*args):
        raise ProbabilityUnavailable("No logprobs")
    monkeypatch.setattr(runner, "plan", unavailable)
    monkeypatch.setattr(runner, "branch", lambda *args: pytest.fail("Must not launch a fallback branch"))
    summary = runner.run_task(SimpleNamespace(task_id="task"))
    assert summary["methods"]["bpo"]["status"] == "blocked"
    assert summary["methods"]["bpo"]["error_type"] == "ProbabilityUnavailable"
