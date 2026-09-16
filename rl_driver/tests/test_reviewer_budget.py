import json

import httpx
import pytest

from rl_driver.branch_review import BranchingConfig, ask_reviewer, validate_plan


def evidence():
    return {
        "target_resolved": True, "branch_limit": 4, "task": "Fix the parser",
        "attempts": [{"job_id": "parent", "available_points": [{"id": "point"}]}],
    }


@pytest.mark.parametrize("strict", [True, False])
def test_budget_is_sent_only_to_an_enforcing_backend(monkeypatch, strict):
    calls = []

    def transport(request):
        calls.append(request)
        if request.url.path == "/get_server_info":
            return httpx.Response(200, json={"server_args": {"enable_strict_thinking": strict}})
        body = json.loads(request.content)
        assert body["max_tokens"] == 32768
        assert body["custom_params"] == {"thinking_budget": 16384}
        assert body["chat_template_kwargs"]["enable_thinking"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["branches"]["maxItems"] == 4
        return httpx.Response(200, json={"choices": [{
            "message": {"content": '{"synthesis":"No useful route","branches":[]}'},
            "finish_reason": "stop",
        }], "usage": {"completion_tokens": 30, "reasoning_tokens": 10}})

    original = httpx.Client
    monkeypatch.setattr(
        "rl_driver.branch_review.httpx.Client",
        lambda **kwargs: original(transport=httpx.MockTransport(transport), **kwargs),
    )
    config = BranchingConfig(
        reviewer_model="model", reviewer_endpoint="http://reviewer",
        reviewer_max_tokens=32768, reviewer_thinking_budget=16384,
    )
    if strict:
        assert validate_plan(ask_reviewer(config, evidence()), evidence())["branches"] == []
        assert [request.method for request in calls] == ["GET", "POST"]
    else:
        with pytest.raises(ValueError, match="enable_strict_thinking"):
            ask_reviewer(config, evidence())
        assert [request.method for request in calls] == ["GET"]


@pytest.mark.parametrize("budget", [-1, True, 32768, 32769])
def test_invalid_thinking_budgets_fail_before_requests(budget):
    with pytest.raises(ValueError, match="reviewer_thinking_budget"):
        BranchingConfig.from_dict({
            "reviewer_endpoint": "http://reviewer", "reviewer_max_tokens": 32768,
            "reviewer_thinking_budget": budget,
        })


def test_thinking_budget_requires_local_transport():
    with pytest.raises(ValueError, match="local reviewer_endpoint"):
        BranchingConfig.from_dict({"reviewer_thinking_budget": 100})
